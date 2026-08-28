"""视频字幕生成 Web 服务（优化版）。

相对原版的性能关键改进：
- 支持多线程并发（threaded=True），大文件上传不再阻塞整个服务。
- 转写 + 烧字幕放到后台线程执行，接口立即返回，前端轮询状态，彻底告别"卡死"。
- 采用基于任务目录(uuid)的沙箱，避免文件名拼接/路径穿越与全局变量踩踏。
- 预转写出下载文件名编码等常见坑。
"""

import glob
import os
import shutil
import threading
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

from addSubtitles import RealizeAddSubtitles
from transcribe import TranscribeArgs, Transcribe

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'upload')

# 上传大小上限：2GB（单位字节）
MAX_CONTENT_LENGTH_BYTES = 2 * 1024 * 1024 * 1024
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv')

# 任务结果保留时长（秒），超过后由清理线程删除
TASK_RETENTION_SECONDS = 24 * 3600
CLEANUP_INTERVAL_SECONDS = 1800

app = Flask(__name__)
app.config['UPLOAD_DIR'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH_BYTES
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _safe_filename(name):
    """保留一个安全的文件名：去掉路径成分，只留合法文件名字符。"""
    name = os.path.basename(name).strip().replace('\\', '').replace('/', '')
    name = ''.join(c for c in name if c not in '<>:"|?*')
    return name or 'video'


# ---------------------------------------------------------------------------
# 后台任务管理
# ---------------------------------------------------------------------------
class Task:
    def __init__(self, task_id, src_path, out_path, src_name):
        self.id = task_id
        self.src_path = src_path
        self.out_path = out_path
        self.src_name = src_name
        self.folder = os.path.dirname(src_path)
        self.stage = 'pending'      # pending/uploaded/transcribing/burning/done/failed
        self.progress = 0           # 0-100
        self.message = '排队中'
        self.error = None

    _LOCK = threading.Lock()

    def update(self, stage=None, progress=None, message=None):
        with self._LOCK:
            if stage is not None:
                self.stage = stage
            if progress is not None:
                self.progress = int(max(0, min(100, progress)))
            if message is not None:
                self.message = message

    def as_dict(self):
        with self._LOCK:
            return {
                'task_id': self.id,
                'stage': self.stage,
                'progress': self.progress,
                'message': self.message,
                'error': self.error,
                'download_url': url_for('download', task_id=self.id)
                if self.stage == 'done' else None,
            }


class TaskRegistry:
    def __init__(self):
        self._tasks = {}
        self._lock = threading.Lock()

    def create(self, task):
        with self._lock:
            self._tasks[task.id] = task
        return task

    def get(self, task_id):
        with self._lock:
            return self._tasks.get(task_id)

    def all(self):
        with self._lock:
            return list(self._tasks.values())


REGISTRY = TaskRegistry()

# 把 whisper 转写参数固定在服务上，前端总是用这一套默认值，
# 避免每条上传请求从系统参数里重新解析。
# 可用环境变量在不动代码的情况下调整（例如小内存/无 GPU 的机器建议用 tiny/base）：
#   WHISPER_MODEL=small   WHISPER_LANG=auto|zh|en   WHISPER_DEVICE=None   WHISPER_VAD=0|1
# 注：WHISPER_LANG 默认 auto —— whisper 自动识别音频语言，中英文均可。
# 注：VAD 会通过 torch.hub 从 GitHub 下载 silero-vad 模型，网络受限/下载失败可能拖慢或挂起，
# 因此默认关闭；需要时可开启，开启后失败也会自动回退为整段转写。
def _env_bool(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ('1', 'true', 'yes', 'on')


def _resolve_lang():
    # 'auto' 或未设置 -> None（whisper 自动检测）；显式 'zh'/'en' 则为指定语言
    v = os.environ.get('WHISPER_LANG', 'auto')
    v = (v or '').strip().lower()
    return None if v in ('', 'auto') else v


TRANSCRIBE_OPTS = TranscribeArgs(
    lang=_resolve_lang(),
    whisper_model=os.environ.get('WHISPER_MODEL', 'small'),
    vad=_env_bool('WHISPER_VAD'),
    device=os.environ.get('WHISPER_DEVICE'),
    delay_seconds=1.0)


def _run_pipeline(task):
    """在后台线程中执行：语音转写 -> 烧录字幕 -> 产出最终视频。"""
    try:
        # 1) 语音转写
        # 注：首次使用某个模型时需要联网下载，可能耗时数分钟。
        task.update(stage='transcribing', progress=5,
                    message='正在加载语音识别模型(首次运行需下载，可能较慢)')
        # 转写子模块上报的 pct 是 0~1 的小数：映射到总进度 5%~50%。
        # （原实现写成 int(pct * 0.45)，对 0~1 的小数取整恒为 0，
        #   导致转写全程进度永远停在 5% —— 这就是"卡在5%"的显示层根因。）
        transcribe = Transcribe(
            TRANSCRIBE_OPTS,
            progress_cb=lambda stage, pct, msg: task.update(
                stage, 5 + int(pct * 45), msg))
        srt_path = transcribe.run_single(task.src_path)

        # 2) 字幕烧录（ffmpeg，只重编码字幕叠加后的视频，不整片 Python 逐帧处理）
        task.update(stage='burning', progress=50, message='语音识别完成，正在烧录字幕')
        burner = RealizeAddSubtitles(task.src_path, srt_path, out_path=task.out_path)
        # ffmpeg 子进程上报的 pct 是 0~100：映射到总进度 50%~100%
        burner.burn(progress_cb=lambda pct: task.update(
            stage='burning', progress=50 + int(pct * 0.5),
            message=f'正在烧录字幕 {pct:.0f}%'))

        if not os.path.isfile(task.out_path) or os.path.getsize(task.out_path) == 0:
            raise RuntimeError('字幕烧录未生成输出文件')

        task.update(stage='done', progress=100, message='已完成，可以下载')
    except Exception as e:  # noqa: BLE001 - 后台线程统一上报失败原因
        task.update(stage='failed', message='处理失败')
        task.error = str(e)
    finally:
        # 视频源文件 + 中间 srt 不必暴露给下载，保留最终产物即可；
        # 但为便于排查先保留任务目录，由清理线程定期回收。
        pass


def _dispatch_pipeline(task):
    threading.Thread(target=_run_pipeline, args=(task,), daemon=True).start()


def _cleanup_loop():
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS)
        now = time.time()
        for task in REGISTRY.all():
            try:
                mtime = os.path.getmtime(task.folder)
            except OSError:
                mtime = time.time() + TASK_RETENTION_SECONDS  # 目录已异常，尽快清理
            if now - mtime > TASK_RETENTION_SECONDS:
                with REGISTRY._lock:
                    REGISTRY._tasks.pop(task.id, None)
                shutil.rmtree(task.folder, ignore_errors=True)


# ---------------------------------------------------------------------------
# 页面与接口
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/api/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if f is None or not f.filename:
        return jsonify({'error': '未选择文件'}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        return jsonify({'error': f'不支持的文件类型，仅支持 {" ".join(VIDEO_EXTENSIONS)}'}), 400

    task_id = uuid.uuid4().hex
    folder = os.path.join(UPLOAD_DIR, task_id)
    os.makedirs(folder, exist_ok=True)

    src_name = _safe_filename(f.filename)
    src_path = os.path.join(folder, 'input' + ext)
    out_name = os.path.splitext(src_name)[0] + '_srt' + ext
    out_path = os.path.join(folder, out_name)

    task = Task(task_id, src_path, out_path, src_name)
    REGISTRY.create(task)

    task.update(stage='uploaded', progress=2, message='正在保存上传文件')
    # 分块写入磁盘，避免整个大文件长时间驻留内存
    try:
        f.save(src_path)
    except Exception as e:  # noqa: BLE001 - 保存失败要反馈给前端
        task.update(stage='failed', message='上传文件保存失败', error=str(e))
        shutil.rmtree(folder, ignore_errors=True)
        return jsonify({'error': f'保存上传文件失败：{e}'}), 500

    _dispatch_pipeline(task)
    return jsonify({'task_id': task_id})


@app.route('/api/status/<task_id>')
def status(task_id):
    task = REGISTRY.get(task_id)
    if task is None:
        return jsonify({'error': '任务不存在或已过期'}), 404
    return jsonify(task.as_dict())


@app.route('/api/download/<task_id>')
def download(task_id):
    task = REGISTRY.get(task_id)
    if task is None or task.stage != 'done' or not os.path.isfile(task.out_path):
        return jsonify({'error': '文件不存在或尚未就绪'}), 404
    out_name = os.path.basename(task.out_path)
    return send_from_directory(task.folder, out_name, as_attachment=True)


if __name__ == '__main__':
    threading.Thread(target=_cleanup_loop, args=(), daemon=True).start()
    # 多线程并发，单次大上传不再阻塞其它请求。
    # 端口经环境变量 PORT 可配（默认 8081）。
    # 生产环境建议改用 waitress：`pip install waitress`，改调用 wsgi 服务器。
    port = int(os.environ.get('PORT', '8081'))
    app.run(host='0.0.0.0', port=port, threaded=True)