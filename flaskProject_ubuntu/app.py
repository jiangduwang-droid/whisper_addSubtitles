"""视频字幕生成 Web 服务（优化版）。

相对原版的性能关键改进：
- 支持多线程并发（threaded=True），大文件上传不再阻塞整个服务。
- 转写 + 烧字幕放到后台线程执行，接口立即返回，前端轮询状态，彻底告别"卡死"。
- 采用基于任务目录(uuid)的沙箱，避免文件名拼接/路径穿越与全局变量踩踏。
- 预转写出下载文件名编码等常见坑。
"""

import glob
import json as _json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import timedelta

import srt
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

from addSubtitles import FFMPEG, RealizeAddSubtitles
from transcribe import TranscribeArgs, Transcribe

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'upload')

# 上传大小上限（单位字节）：默认 4GB，可用环境变量 MAX_UPLOAD_MB 调整（单位 MB）。
# 过大上限受限于服务器磁盘（当前约 15GB 可用）与内存；请按需设置。
MAX_UPLOAD_MB = int(os.environ.get('MAX_UPLOAD_MB', '4096'))
MAX_CONTENT_LENGTH_BYTES = MAX_UPLOAD_MB * 1024 * 1024
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv')

# 任务结果保留时长（秒），超过后由清理线程删除
TASK_RETENTION_SECONDS = 24 * 3600
CLEANUP_INTERVAL_SECONDS = 1800
# 等待人工校对的超时（秒）：转写完成后暂停在这里，用户确认或超时后继续烧录。
# 超时按「当前字幕原样继续」，避免已完成的转写被浪费。
EDIT_WAIT_TIMEOUT = float(os.environ.get('EDIT_WAIT_TIMEOUT', '7200'))
# 校对接口的输入上限，防御异常客户端
MAX_SUBTITLE_COUNT = 3000
MAX_SUBTITLE_TEXT = 500
# 校对页预览视频的转码超时（秒）：超时则回退用原始视频
PREVIEW_TRANSCODE_TIMEOUT = int(os.environ.get('PREVIEW_TRANSCODE_TIMEOUT', '1800'))
# ffprobe 与 ffmpeg 同目录约定（FFMPEG_BIN 指定完整路径时同样适用）
FFPROBE = os.environ.get('FFPROBE_BIN') or FFMPEG.replace('ffmpeg', 'ffprobe')

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
        self.stage = 'pending'      # pending/uploaded/transcribing/awaiting_edit/burning/done/failed
        self.progress = 0           # 0-100
        self.message = '排队中'
        self.error = None
        self.srt_path = None                  # 转写产出的 srt，人工校对会原地改写
        self.style = None                     # 字幕样式（校对预览/烧录用），upload 时填入
        self.edit_confirmed = threading.Event()   # 人工校对完成信号，置位后流水线继续烧录
        # 校对页预览视频：none / generating / ready / failed
        self.preview_path = None
        self.preview_state = 'none'

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


# ---------------------------------------------------------------------------
# 前端可配置项：语言 / 模型 / 字幕样式（大小、颜色）
# ---------------------------------------------------------------------------
ALLOWED_MODELS = ('tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3')
ALLOWED_LANGS = ('auto', 'zh', 'en')
HEX_COLOR_RE = re.compile(r'^#[0-9a-fA-F]{6}$')


def _to_int(value, default, lo, hi):
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _parse_options(form):
    """解析并校验上传附带的配置，任何非法输入都回退到安全默认值。"""
    lang = (form.get('language') or 'auto').strip().lower()
    if lang not in ALLOWED_LANGS:
        lang = 'auto'
    lang = None if lang == 'auto' else lang

    model = (form.get('model') or 'base').strip().lower()
    if model not in ALLOWED_MODELS:
        model = 'base'

    font_size = _to_int(form.get('font_size'), 32, 8, 100)

    text_color = (form.get('text_color') or '#FFFFFF').upper()
    if not HEX_COLOR_RE.match(text_color):
        text_color = '#FFFFFF'

    outline_color = (form.get('outline_color') or '#000000').upper()
    if not HEX_COLOR_RE.match(outline_color):
        outline_color = '#000000'

    return {
        'lang': lang,
        'model': model,
        'style': {
            'font_size': font_size,
            'text_color': text_color,
            'outline_color': outline_color,
        },
    }


def _run_pipeline(task, opts):
    """在后台线程中执行：语音转写 -> 烧录字幕 -> 产出最终视频。"""
    try:
        # 1) 语音转写
        # 注：首次使用某个模型时需要联网下载，可能耗时数分钟。
        task.update(stage='transcribing', progress=5,
                    message='正在加载语音识别模型(首次运行需下载，可能较慢)')
        transcribe_args = TranscribeArgs(
            lang=opts['lang'],
            whisper_model=opts['model'],
            vad=TRANSCRIBE_OPTS.vad,
            device=TRANSCRIBE_OPTS.device,
            delay_seconds=TRANSCRIBE_OPTS.delay_seconds)
        # 转写子模块上报的 pct 是 0~1 的小数：映射到总进度 5%~50%。
        # （原实现写成 int(pct * 0.45)，对 0~1 的小数取整恒为 0，
        #   导致转写全程进度永远停在 5% —— 这就是"卡在5%"的显示层根因。）
        transcribe = Transcribe(
            transcribe_args,
            progress_cb=lambda stage, pct, msg: task.update(
                stage, 5 + int(pct * 45), msg))
        srt_path = transcribe.run_single(task.src_path)
        task.srt_path = srt_path

        # 2) 人工校对：转写完成后暂停流水线，等待用户在页面上编辑字幕
        #    （调整时间轴/文本，与画面对齐），点击「确认」后继续烧录。
        #    超时未确认则按当前字幕原样继续，避免已完成的转写被浪费。
        task.update(stage='awaiting_edit', progress=48,
                    message='字幕已生成，请校对时间轴与文本')
        # 后台生成浏览器可流畅播放的预览副本（源视频编码不兼容时必须转码）
        threading.Thread(target=_generate_preview, args=(task,), daemon=True,
                         name=f'preview-{task.id[:8]}').start()
        if not task.edit_confirmed.wait(EDIT_WAIT_TIMEOUT):
            task.update(message='校对超时，按当前字幕继续')

        # 3) 字幕烧录（ffmpeg，只重编码字幕叠加后的视频，不整片 Python 逐帧处理）
        task.update(stage='burning', progress=50, message='校对完成，正在烧录字幕')
        burner = RealizeAddSubtitles(
            task.src_path, srt_path, out_path=task.out_path, style=opts['style'])
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


def _dispatch_pipeline(task, opts):
    threading.Thread(target=_run_pipeline, args=(task, opts), daemon=True).start()


# ---------------------------------------------------------------------------
# 校对页预览视频：浏览器兼容副本
#
# 直接把原始视频给 <video> 常见两类问题：
# 1. 编码不兼容（HEVC/H.265、10bit、MKV/AVI 容器等）—— 浏览器只解得出
#    声音、画面黑屏，即用户反馈的「只有声音」；
# 2. MP4 的 moov 索引在文件尾（无 faststart）+ 文件大 —— 播放器要等
#    索引才能起播/拖动，表现为长时间转圈、卡顿。
# 解决：探测编码，能直用就直用/快速重封装，否则后台转码出 H.264+AAC
# 的 720p 预览副本（+faststart），前端轮询就绪后再播放。
# ---------------------------------------------------------------------------
def _probe_video(path):
    """ffprobe 探测视频编码信息；失败返回 None。"""
    try:
        proc = subprocess.run(
            [FFPROBE, '-v', 'error', '-print_format', 'json',
             '-show_streams', '-show_format', path],
            capture_output=True, text=True, timeout=120)
        info = _json.loads(proc.stdout or '{}')
    except Exception:  # noqa: BLE001 - 探测失败按不兼容处理
        return None
    v = next((s for s in info.get('streams', [])
              if s.get('codec_type') == 'video'), None)
    a = next((s for s in info.get('streams', [])
              if s.get('codec_type') == 'audio'), None)
    if not v:
        return None
    try:
        height = int(v.get('height') or 0)
    except (TypeError, ValueError):
        height = 0
    return {
        'vcodec': v.get('codec_name'),
        'pix_fmt': v.get('pix_fmt'),
        'height': height,
        'acodec': a.get('codec_name') if a else None,
    }


def _has_faststart(path):
    """检查 MP4 的 moov 索引是否在文件头（faststart）。

    moov 在 mdat 之后时，浏览器必须先摸到文件尾才能起播/拖动 ——
    大文件经 HTTP 播放就会表现为卡顿。读头部 256KB 足够判断。
    """
    try:
        with open(path, 'rb') as f:
            head = f.read(256 * 1024)
    except OSError:
        return True  # 读不了就不折腾，按可用处理
    i_moov, i_mdat = head.find(b'moov'), head.find(b'mdat')
    if i_moov == -1 and i_mdat == -1:
        return True  # 头部都未见索引，交给转码分支兜底
    if i_mdat == -1:
        return True
    if i_moov == -1:
        return False
    return i_moov < i_mdat


def _is_browser_friendly(info, ext):
    """源视频是否能被主流浏览器直接解码播放。"""
    return (info is not None
            and ext in ('.mp4', '.m4v')
            and info['vcodec'] == 'h264'
            and info['pix_fmt'] in ('yuv420p', 'yuvj420p')
            and info['acodec'] in ('aac', 'mp3', None)
            and info['height'] > 0)


def _generate_preview(task):
    """生成校对页预览副本（后台线程）。状态写入 task.preview_state。"""
    out_path = os.path.join(task.folder, 'preview.mp4')
    task.preview_state = 'generating'
    src_ext = os.path.splitext(task.src_path)[1].lower()
    try:
        info = _probe_video(task.src_path)
        if (_is_browser_friendly(info, src_ext)
                and info['height'] <= 1080
                and _has_faststart(task.src_path)):
            # 已是浏览器友好且索引在头部：直接用原片，零等待
            task.preview_path = task.src_path
            task.preview_state = 'ready'
            logging.getLogger(__name__).info(
                f'preview: {task.id} 直接使用原片（h264/aac, faststart）')
            return
        if _is_browser_friendly(info, src_ext):
            # 编码可播但缺 faststart：流拷贝重封装，秒级完成
            cmd = [FFMPEG, '-y', '-loglevel', 'error', '-i', task.src_path,
                   '-c', 'copy', '-movflags', '+faststart', out_path]
        else:
            # 编码不兼容（HEVC/MKV/AVI/10bit 等）：转码 H.264+AAC，最高 720p
            cmd = [FFMPEG, '-y', '-loglevel', 'error', '-i', task.src_path,
                   '-vf', "scale=-2:'min(720,ih)'",
                   '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                   '-pix_fmt', 'yuv420p',
                   '-c:a', 'aac', '-b:a', '128k',
                   '-movflags', '+faststart', out_path]
        subprocess.run(cmd, check=True, timeout=PREVIEW_TRANSCODE_TIMEOUT)
        if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError('预览视频未生成')
        task.preview_path = out_path
        task.preview_state = 'ready'
    except Exception as e:  # noqa: BLE001 - 预览失败不阻断校对，回退原片
        task.preview_state = 'failed'
        logging.getLogger(__name__).warning(
            f'preview: {task.id} 生成失败（前端将回退原始视频）：{e}')


def _warmup_model():
    """启动时后台预热默认模型：把「首次下载/加载」的成本移到服务启动阶段，
    避免第一个用户任务长时间停在 5%（加载模型）。
    预热失败只记日志，不影响服务启动。"""
    model_name = (os.environ.get('WHISPER_WARMUP_MODEL')
                  or os.environ.get('WHISPER_MODEL')
                  or 'base')
    try:
        args = TranscribeArgs(whisper_model=model_name)
        # noqa 以下 import 仅为复用 _get_model 的超时保护
        from transcribe import _get_model
        _get_model(args)
        logging.getLogger(__name__).info(f'Warmup model {model_name} ready')
    except Exception as e:  # noqa: BLE001 - 预热失败不影响服务
        logging.getLogger(__name__).warning(
            f'模型 {model_name} 预热失败（首个任务会再次尝试）：{e}')


_WARMUP_ONCE_LOCK = threading.Lock()
_warmup_started = False


def _start_warmup():
    """只触发一次的模型预热（python app.py 与 waitress 等 WSGI 启动方式都覆盖：
    主入口启动时调用一次；首个上传请求时兜底再调用一次，已启动则直接返回）。"""
    global _warmup_started
    with _WARMUP_ONCE_LOCK:
        if _warmup_started:
            return
        _warmup_started = True
    threading.Thread(target=_warmup_model, daemon=True, name='model-warmup').start()


def _cleanup_loop():
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS)
        now = time.time()
        for task in REGISTRY.all():
            try:
                mtime = os.path.getmtime(task.folder)
            except OSError:
                mtime = 0.0  # 目录已异常（可能已被删），视为过期尽快清理
            if now - mtime > TASK_RETENTION_SECONDS:
                with REGISTRY._lock:
                    REGISTRY._tasks.pop(task.id, None)
                shutil.rmtree(task.folder, ignore_errors=True)

        # 兜底清理孤儿目录：服务重启后 REGISTRY 清空，旧任务目录不扫就永远留下。
        # 只要目录超过保留时长（无论是否在内存任务表中）都回收，防止磁盘被吃满。
        try:
            for name in os.listdir(UPLOAD_DIR):
                folder = os.path.join(UPLOAD_DIR, name)
                if not os.path.isdir(folder):
                    continue
                try:
                    mtime = os.path.getmtime(folder)
                except OSError:
                    continue
                if now - mtime > TASK_RETENTION_SECONDS:
                    shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 页面与接口
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    # 把上传大小上限注入模板，供前端在上传前做本地预检
    return render_template('upload.html', max_upload_bytes=MAX_CONTENT_LENGTH_BYTES)


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': f'文件超过大小限制（上限 {MAX_UPLOAD_MB}MB），请压缩后再上传'
    }), 413


@app.route('/api/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if f is None or not f.filename:
        return jsonify({'error': '未选择文件'}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        return jsonify({'error': f'不支持的文件类型，仅支持 {" ".join(VIDEO_EXTENSIONS)}'}), 400

    # 解析并校验前端配置（语言/模型/字幕样式），全部有安全默认值
    opts = _parse_options(request.form)

    task_id = uuid.uuid4().hex
    folder = os.path.join(UPLOAD_DIR, task_id)
    os.makedirs(folder, exist_ok=True)

    src_name = _safe_filename(f.filename)
    src_path = os.path.join(folder, 'input' + ext)
    out_name = os.path.splitext(src_name)[0] + '_srt' + ext
    out_path = os.path.join(folder, out_name)

    task = Task(task_id, src_path, out_path, src_name)
    task.style = opts['style']
    REGISTRY.create(task)

    # 兜底触发一次模型预热（若主入口未启动/被跳过，这里补上）
    _start_warmup()

    task.update(stage='uploaded', progress=2, message='正在保存上传文件')
    # 分块写入磁盘，避免整个大文件长时间驻留内存
    try:
        f.save(src_path)
    except Exception as e:  # noqa: BLE001 - 保存失败要反馈给前端
        task.update(stage='failed', message='上传文件保存失败', error=str(e))
        shutil.rmtree(folder, ignore_errors=True)
        return jsonify({'error': f'保存上传文件失败：{e}'}), 500

    _dispatch_pipeline(task, opts)
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


# ---------------------------------------------------------------------------
# 人工校对：字幕查看 / 编辑 / 确认 / 视频预览
# ---------------------------------------------------------------------------
def _load_srt_subtitles(task):
    """读取任务 srt 并解析为 srt.Subtitle 列表。"""
    if not task.srt_path or not os.path.isfile(task.srt_path):
        raise FileNotFoundError('字幕尚未生成')
    with open(task.srt_path, 'r', encoding='utf-8') as f:
        return srt.parse(f.read())


@app.route('/api/subtitles/<task_id>', methods=['GET'])
def get_subtitles(task_id):
    """返回字幕列表（秒为单位的浮点数，前端直接做时间轴运算）。"""
    task = REGISTRY.get(task_id)
    if task is None:
        return jsonify({'error': '任务不存在或已过期'}), 404
    try:
        subs = _load_srt_subtitles(task)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:  # noqa: BLE001 - 解析失败要反馈给前端
        return jsonify({'error': f'字幕文件解析失败：{e}'}), 500
    return jsonify({
        'task_id': task_id,
        'stage': task.stage,
        'style': task.style or {},
        'subtitles': [{
            'index': s.index,
            'start': round(s.start.total_seconds(), 3),
            'end': round(s.end.total_seconds(), 3),
            'text': s.content.strip(),
        } for s in subs],
    })


@app.route('/api/subtitles/<task_id>', methods=['PUT'])
def put_subtitles(task_id):
    """保存人工校对结果：严格校验后重写 srt（原地覆盖，烧录时读取的就是它）。"""
    task = REGISTRY.get(task_id)
    if task is None:
        return jsonify({'error': '任务不存在或已过期'}), 404
    # 已确认（事件置位）后即使 stage 尚未翻转也拒绝写入，避免与烧录抢文件
    if task.stage != 'awaiting_edit' or task.edit_confirmed.is_set():
        return jsonify({'error': '当前阶段不能编辑字幕（请在校对页操作）'}), 409

    data = request.get_json(silent=True) or {}
    raw = data.get('subtitles')
    if not isinstance(raw, list):
        return jsonify({'error': '请求数据格式错误'}), 400
    if not raw:
        return jsonify({'error': '字幕列表不能为空'}), 400
    if len(raw) > MAX_SUBTITLE_COUNT:
        return jsonify({'error': f'字幕条数超过上限（{MAX_SUBTITLE_COUNT}）'}), 400

    items = []
    for i, it in enumerate(raw):
        if not isinstance(it, dict):
            return jsonify({'error': f'第 {i + 1} 条字幕格式错误'}), 400
        try:
            start = float(it.get('start'))
            end = float(it.get('end'))
        except (TypeError, ValueError):
            return jsonify({'error': f'第 {i + 1} 条字幕时间格式错误'}), 400
        text = str(it.get('text') or '').strip()
        if not (0 <= start < end) or end > 24 * 3600:
            return jsonify({'error': f'第 {i + 1} 条字幕时间无效（需 0 ≤ 开始 < 结束）'}), 400
        if not text:
            continue  # 空文本的条目直接丢弃（渲染不出来）
        items.append((start, end, text[:MAX_SUBTITLE_TEXT]))

    if not items:
        return jsonify({'error': '字幕文本不能全部为空'}), 400
    # 按开始时间排序并重新编号，保证时间轴单调
    items.sort(key=lambda x: x[0])
    subs = [srt.Subtitle(index=i + 1,
                         start=timedelta(seconds=s),
                         end=timedelta(seconds=e),
                         content=t)
            for i, (s, e, t) in enumerate(items)]
    with open(task.srt_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subs))
    # 保存动作给任务目录续命（清理线程按目录 mtime 回收）
    try:
        os.utime(task.folder, None)
    except OSError:
        pass
    return jsonify({'ok': True, 'count': len(subs)})


@app.route('/api/confirm/<task_id>', methods=['POST'])
def confirm_edit(task_id):
    """用户确认校对完成：置位信号，后台流水线从等待点继续烧录。"""
    task = REGISTRY.get(task_id)
    if task is None:
        return jsonify({'error': '任务不存在或已过期'}), 404
    if task.stage != 'awaiting_edit':
        return jsonify({'error': '当前阶段无需确认'}), 409
    task.edit_confirmed.set()
    task.update(message='已确认，准备烧录字幕')
    return jsonify({'ok': True})


@app.route('/api/video/<task_id>')
def task_video(task_id):
    """提供原始视频给校对页预览（支持 Range，可拖动进度条）。"""
    task = REGISTRY.get(task_id)
    if task is None or not os.path.isfile(task.src_path):
        return jsonify({'error': '视频不存在或已过期'}), 404
    return send_from_directory(
        task.folder, os.path.basename(task.src_path), conditional=True)


@app.route('/api/preview/<task_id>')
def preview_status(task_id):
    """预览副本生成状态：none/generating/ready/failed。"""
    task = REGISTRY.get(task_id)
    if task is None:
        return jsonify({'error': '任务不存在或已过期'}), 404
    return jsonify({
        'task_id': task_id,
        'state': task.preview_state,
        'url': f'/api/preview/{task_id}/file'
        if task.preview_state == 'ready' else None,
    })


@app.route('/api/preview/<task_id>/file')
def preview_file(task_id):
    """预览视频文件（浏览器兼容副本；支持 Range）。"""
    task = REGISTRY.get(task_id)
    if (task is None or task.preview_state != 'ready'
            or not task.preview_path or not os.path.isfile(task.preview_path)):
        return jsonify({'error': '预览尚未就绪'}), 404
    return send_from_directory(
        os.path.dirname(task.preview_path),
        os.path.basename(task.preview_path),
        conditional=True, mimetype='video/mp4')


if __name__ == '__main__':
    threading.Thread(target=_cleanup_loop, args=(), daemon=True).start()
    # 启动时后台预热默认模型，避免第一个用户任务长时间停在「加载模型」
    _start_warmup()
    # 多线程并发，单次大上传不再阻塞其它请求。
    # 端口经环境变量 PORT 可配（默认 8081）。
    # 生产环境建议改用 waitress：`pip install waitress`，改调用 wsgi 服务器。
    port = int(os.environ.get('PORT', '8081'))
    app.run(host='0.0.0.0', port=port, threaded=True)