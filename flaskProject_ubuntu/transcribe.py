import datetime
import gc
import logging
import os
import threading
import time

import opencc
import srt

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# 国内服务器直连 HuggingFace 不通，默认走镜像下载模型；已显式设置时保留用户配置。
# 同时禁用 xet 协议（镜像不支持），改走传统 HTTP 下载。
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')


class TranscribeArgs:
    """用于 web 服务的转写参数对象，替代 app 里直接 parse_args。"""

    # lang: 指定语言（如 'zh'/'en'），None 表示自动检测，中英文均可
    def __init__(self, lang=None, whisper_model='base', vad=False,
                 device=None, prompt='', encoding='utf-8', delay_seconds=1.0):
        self.inputs = []
        self.lang = lang
        self.whisper_model = whisper_model
        self.vad = vad
        self.device = device
        self.prompt = prompt
        self.encoding = encoding
        self.delay_seconds = delay_seconds


# ---------------------------------------------------------------------------
# faster-whisper（CTranslate2）引擎：CPU 上比 openai/whisper 快约 4~5 倍、
# 内存占用大幅降低，配合 int8 量化后 medium/large 也能在低内存服务器运行。
#
# 修复「任务卡在 5% 不动」的关键点：
# 1) 模型下载（huggingface_hub）默认没有总超时，网络停滞时会无限等待 ——
#    显式设置 HF_HUB_DOWNLOAD_TIMEOUT，让网络问题以异常形式暴露。
# 2) 原实现把「下载+加载」放在全局锁内：一旦某次加载挂起，锁被永久持有，
#    后续所有任务（无论什么模型）都会在锁上排队，永远停在 5%。
#    现在改为：缓存锁只保护字典读写；加载放到子线程并带总超时，
#    超时立即抛错，任务标记 failed 并给出可操作的提示，而不是永久挂起。
# ---------------------------------------------------------------------------
_CACHE_LOCK = threading.Lock()
_MODEL_CACHE = {}
# 同一模型的并发加载互斥（不同模型互不影响）
_LOAD_LOCKS_LOCK = threading.Lock()
_LOAD_LOCKS = {}
# 最多同时驻留的模型数量：切换不同模型时防止多个模型同时占用内存
_MODEL_CACHE_MAX = int(os.environ.get('WHISPER_MODEL_CACHE_MAX', '2'))
# 量化精度：int8 最快最省内存；float32 更准但更慢。CPU 推荐 int8。
_COMPUTE_TYPE = os.environ.get('WHISPER_COMPUTE_TYPE', 'int8')
# 模型加载（含首次下载）总超时（秒）：超时即失败报错，不再无限等待
_MODEL_LOAD_TIMEOUT = float(os.environ.get('WHISPER_MODEL_LOAD_TIMEOUT', '600'))

# huggingface_hub 的文件下载默认无总超时：连接停滞会永久挂起。
# 显式给下载设置超时（秒），网络异常会正常抛错并让任务失败。
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')


def _load_model_blocking(args):
    """实际执行模型下载 + 加载（可能较慢），失败抛异常。"""
    # 延迟 import，避免非转写路径也强制拉起 ctranslate2
    from faster_whisper import WhisperModel
    logging.info(f'Loading faster-whisper model {args.whisper_model} '
                 f'device={args.device or "cpu"} compute_type={_COMPUTE_TYPE}')
    tic = time.time()
    model = WhisperModel(
        args.whisper_model,
        device=args.device or 'cpu',
        compute_type=_COMPUTE_TYPE)
    logging.info(f'faster-whisper model loaded in {time.time() - tic:.1f} sec')
    return model


def _get_model(args):
    key = (args.whisper_model, args.device)
    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
    if model is not None:
        return model

    with _LOAD_LOCKS_LOCK:
        key_lock = _LOAD_LOCKS.setdefault(key, threading.Lock())
    # 同一模型的并发任务在这里排队（等待时间受 _MODEL_LOAD_TIMEOUT 约束）；
    # 不同模型互不阻塞 —— 一个模型加载挂起不再拖死整个服务。
    with key_lock:
        # 双重检查：排队期间可能已被其它任务加载完成
        with _CACHE_LOCK:
            model = _MODEL_CACHE.get(key)
        if model is not None:
            return model

        result = {}

        def _worker():
            try:
                result['model'] = _load_model_blocking(args)
            except Exception as e:  # noqa: BLE001 - 加载异常带回调用方
                result['error'] = e

        t = threading.Thread(
            target=_worker, daemon=True, name=f'whisper-load-{args.whisper_model}')
        t.start()
        t.join(_MODEL_LOAD_TIMEOUT)

        if t.is_alive():
            raise RuntimeError(
                f'语音识别模型 {args.whisper_model} 加载超时'
                f'（超过 {int(_MODEL_LOAD_TIMEOUT)} 秒）。'
                f'首次运行需从 {os.environ.get("HF_ENDPOINT", "https://huggingface.co")} '
                f'下载模型，请检查服务器外网连通性后重试；'
                f'或在服务器上手动预下载模型后重启服务：'
                f'python -c "from faster_whisper import WhisperModel; '
                f'WhisperModel(\'{args.whisper_model}\')"')
        if 'error' in result:
            raise RuntimeError(
                f'语音识别模型 {args.whisper_model} 加载失败：{result["error"]}')

        model = result['model']
        with _CACHE_LOCK:
            _MODEL_CACHE[key] = model
            # 超出上限时淘汰最早的模型，及时释放内存
            while len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
                _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
                gc.collect()
        return model


# opencc 繁转简实例全局复用，避免每条任务重复加载词典
_OPENCC = None
_OPENCC_LOCK = threading.Lock()


def _get_opencc():
    global _OPENCC
    with _OPENCC_LOCK:
        if _OPENCC is None:
            _OPENCC = opencc.OpenCC('t2s')
        return _OPENCC


class Transcribe:
    def __init__(self, args, progress_cb=None):
        self.args = args
        self.sampling_rate = 16000
        self.whisper_model = None
        self.progress_cb = progress_cb or (lambda *_: None)

    def _report(self, stage, pct, msg):
        try:
            self.progress_cb(stage, pct, msg)
        except Exception:  # 进度上报失败不应影响主流程
            pass

    def run(self):
        for input_path in self.args.inputs:
            self.run_single(input_path)

    def run_single(self, input_path):
        """转写单个文件并写 srt，返回 srt 路径。"""
        name, _ = os.path.splitext(input_path)
        logging.info(f'Transcribing {input_path}')

        segments, info = self._transcribe(input_path)

        output = name + '.srt'
        self._save_srt(output, segments)
        logging.info(f'Transcribed {input_path} -> {output} (detected: {info.language})')
        return output

    def _transcribe(self, input_path):
        tic = time.time()
        self.whisper_model = _get_model(self.args)

        # 中文场景用简短初始提示，引导简体输出、稳定标点
        prompt = self.args.prompt
        if not prompt and self.args.lang in ('zh', 'zh-CN', 'chinese'):
            prompt = '以下是普通话的句子。'

        self._report('transcribing', 0.1, '正在识别语音')
        segments, info = self.whisper_model.transcribe(
            input_path,
            language=self.args.lang,      # None 时自动检测
            beam_size=5,
            vad_filter=self.args.vad,     # 内置 Silero VAD，跳过静音段
            initial_prompt=prompt,
        )

        # segments 是惰性生成器，迭代时逐句产出并上报进度
        results = []
        total_dur = info.duration or 1.0
        for seg in segments:
            results.append(seg)
            # 用当前句子结束时间占全片时长的比例，反映真实进度
            pct = max(0.1, min(0.95, seg.end / total_dur))
            self._report('transcribing', pct, f'已识别 {len(results)} 句')
        logging.info(f'Done transcription in {time.time() - tic:.1f} sec')
        return results, info

    def _save_srt(self, output, segments):
        subs = []
        cc = _get_opencc()
        for s in segments:
            text = s.text.strip()
            if not text:
                continue
            subs.append(srt.Subtitle(
                index=0,
                start=datetime.timedelta(seconds=s.start),
                end=datetime.timedelta(seconds=s.end),
                content=cc.convert(text)))

        with open(output, 'wb') as f:
            f.write(srt.compose(subs).encode(self.args.encoding, 'replace'))