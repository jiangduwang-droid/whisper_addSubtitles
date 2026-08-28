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
# ---------------------------------------------------------------------------
_MODEL_LOCK = threading.Lock()
_MODEL_CACHE = {}
# 最多同时驻留的模型数量：切换不同模型时防止多个模型同时占用内存
_MODEL_CACHE_MAX = int(os.environ.get('WHISPER_MODEL_CACHE_MAX', '2'))
# 量化精度：int8 最快最省内存；float32 更准但更慢。CPU 推荐 int8。
_COMPUTE_TYPE = os.environ.get('WHISPER_COMPUTE_TYPE', 'int8')


def _get_model(args):
    key = (args.whisper_model, args.device)
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            # 延迟 import，避免非转写路径也强制拉起 ctranslate2
            from faster_whisper import WhisperModel
            logging.info(f'Loading faster-whisper model {key} compute_type={_COMPUTE_TYPE}')
            tic = time.time()
            model = WhisperModel(
                args.whisper_model,
                device=args.device or 'cpu',
                compute_type=_COMPUTE_TYPE)
            _MODEL_CACHE[key] = model
            logging.info(f'faster-whisper model loaded in {time.time() - tic:.1f} sec')
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