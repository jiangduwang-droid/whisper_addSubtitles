import datetime
import logging
import os
import threading
import time

import opencc
import srt

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


class TranscribeArgs:
    """用于 web 服务的转写参数对象，替代 app 里直接 parse_args。"""

    # lang: 指定语言（如 'zh'/'en'），None 表示自动检测，中英文均可
    def __init__(self, lang=None, whisper_model='small', vad=True,
                 device=None, prompt='', encoding='utf-8', delay_seconds=1.0):
        self.inputs = []
        self.lang = lang
        self.whisper_model = whisper_model
        self.vad = vad
        self.device = device
        self.prompt = prompt
        self.encoding = encoding
        # 相邻语音片段间隔小于该秒数时合并，减少 whisper 调用次数、加快整体速度
        self.delay_seconds = delay_seconds


# ---------------------------------------------------------------------------
# 全局 whisper 模型缓存：模型加载一次、全进程复用。
# small 模型约 1GB 内存/加载耗时以分钟计，原始代码每个请求都重载，是主要瓶颈之一。
#
# 修复「任务卡在 5% 不动」的关键点：
# 原实现把「下载+加载」放在全局锁内，一旦某次加载（含模型下载）挂起，
# 锁被永久持有，后续所有任务都会在锁上排队，永远停在 5%。
# 现在改为：缓存锁只保护字典读写；加载放到子线程并带总超时（默认 600 秒），
# 超时立即抛错，任务标记 failed 并给出可操作的提示，而不是永久挂起。
# ---------------------------------------------------------------------------
_CACHE_LOCK = threading.Lock()
_MODEL_CACHE = {}
# 模型加载（含首次下载）总超时（秒）：超时即失败报错，不再无限等待
_MODEL_LOAD_TIMEOUT = float(os.environ.get('WHISPER_MODEL_LOAD_TIMEOUT', '600'))


def _load_model_blocking(args):
    """实际执行模型下载 + 加载（可能较慢），失败抛异常。"""
    import whisper
    logging.info(f'Loading whisper model {args.whisper_model}')
    tic = time.time()
    model = whisper.load_model(args.whisper_model, args.device)
    logging.info(f'whisper model loaded in {time.time() - tic:.1f} sec')
    return model


def _get_model(args):
    key = (args.whisper_model, args.device)
    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
    if model is not None:
        return model

    # 加载放到子线程并带超时等待：下载挂起时任务会明确报错，而不是永久卡住
    result = {}

    def _worker():
        try:
            result['model'] = _load_model_blocking(args)
        except Exception as e:  # noqa: BLE001 - 加载异常带回调用方
            result['error'] = e

    t = threading.Thread(target=_worker, daemon=True,
                         name=f'whisper-load-{args.whisper_model}')
    t.start()
    t.join(_MODEL_LOAD_TIMEOUT)

    if t.is_alive():
        raise RuntimeError(
            f'语音识别模型 {args.whisper_model} 加载超时'
            f'（超过 {int(_MODEL_LOAD_TIMEOUT)} 秒）。'
            f'首次运行需联网下载模型，请检查网络连通性后重试；'
            f'或先在本机手动预下载一次模型，再重启服务。')
    if 'error' in result:
        raise RuntimeError(
            f'语音识别模型 {args.whisper_model} 加载失败：{result["error"]}')

    model = result['model']
    with _CACHE_LOCK:
        _MODEL_CACHE[key] = model
    return model


class Transcribe:
    def __init__(self, args, progress_cb=None):
        self.args = args
        self.sampling_rate = 16000
        self.whisper_model = None
        self.vad_model = None
        self.detect_speech = None
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
        """对单个文件完成 加载/去噪/VAD/转写/写srt/写md的整条流水线，返回 srt 路径。"""
        name, _ = os.path.splitext(input_path)
        logging.info(f'Transcribing {input_path}')

        audio = self._load_audio(input_path)
        speech_timestamps = self._detect_voice_activity(audio)
        transcribe_results = self._transcribe(audio, speech_timestamps)

        output = name + '.srt'
        self._save_srt(output, transcribe_results)
        logging.info(f'Transcribed {input_path} to {output}')

        # web 场景不需要生成用于人工挑选的 .md，省掉一次 IO
        return output

    def _load_audio(self, input_path):
        import whisper
        return whisper.load_audio(input_path, sr=self.sampling_rate)

    def _detect_voice_activity(self, audio):
        if not self.args.vad:
            return [{'start': 0, 'end': len(audio)}]

        try:
            tic = time.time()
            if self.vad_model is None or self.detect_speech is None:
                import torch
                self.vad_model, funcs = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad', model='silero_vad',
                    trust_repo=True)
                self.detect_speech = funcs[0]
            speeches = self.detect_speech(audio, self.vad_model,
                                          sampling_rate=self.sampling_rate)

            # 过滤过短的片段、合并相邻片段，减少 whisper 推理次数
            from utils import (expand_segments, merge_adjacent_segments,
                               remove_short_segments)
            speeches = remove_short_segments(speeches, 0.25 * self.sampling_rate)
            speeches = merge_adjacent_segments(
                speeches, self.args.delay_seconds * self.sampling_rate)
            speeches = expand_segments(speeches, 0.2 * self.sampling_rate,
                                       0.0 * self.sampling_rate, audio.shape[0])
            logging.info(f'Done voice activity detection in {time.time() - tic:.1f} sec')
            return speeches
        except Exception as e:  # noqa: BLE001
            # VAD 依赖 torch.hub 从 GitHub 下载模型，可能因网络/环境失败。
            # 失败时优雅回退为整段转写，绝不让一条上传任务挂在 VAD 上。
            logging.warning(f'VAD 不可用，回退为整段转写：{e}')
            return [{'start': 0, 'end': len(audio)}]

    def _transcribe(self, audio, speech_timestamps):
        import whisper
        tic = time.time()
        self.whisper_model = _get_model(self.args)

        res = []
        total = len(speech_timestamps)
        for i, seg in enumerate(speech_timestamps):
            r = self.whisper_model.transcribe(
                audio[int(seg['start']):int(seg['end'])],
                task='transcribe', language=self.args.lang,
                initial_prompt=self.args.prompt)
            r['origin_timestamp'] = seg
            res.append(r)
            self._report('transcribing', (i + 1) / total,
                         f'识别中 {i + 1}/{total}')
        logging.info(f'Done transcription in {time.time() - tic:.1f} sec')
        return res

    def _save_srt(self, output, transcribe_results):
        subs = []
        # whisper 偶尔输出繁体中文，显式转简
        cc = opencc.OpenCC('t2s')
        sample_rate = self.sampling_rate

        def _add_sub(start, end, text):
            subs.append(srt.Subtitle(index=0,
                                     start=datetime.timedelta(seconds=start),
                                     end=datetime.timedelta(seconds=end),
                                     content=cc.convert(text.strip())))

        prev_end = 0
        for r in transcribe_results:
            origin = r['origin_timestamp']
            for s in r['segments']:
                start = s['start'] + origin['start'] / sample_rate
                end = min(s['end'] + origin['start'] / sample_rate,
                          origin['end'] / sample_rate)
                if start > end:
                    continue
                if start > prev_end + 1.0:
                    # 替过长的静音段打上占位，避免字幕时间轴跳变
                    _add_sub(prev_end, start, '< No Speech >')
                _add_sub(start, end, s['text'])
                prev_end = end

        with open(output, 'wb') as f:
            f.write(srt.compose(subs).encode(self.args.encoding, 'replace'))