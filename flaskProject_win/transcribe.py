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
# ---------------------------------------------------------------------------
_MODEL_LOCK = threading.Lock()
_MODEL_CACHE = {}


def _get_model(args):
    key = (args.whisper_model, args.device)
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            # 延迟 import，避免在只有批处理/其它用法时也强制拉起 torch
            import whisper
            logging.info(f'Loading whisper model {key}')
            tic = time.time()
            model = whisper.load_model(args.whisper_model, args.device)
            _MODEL_CACHE[key] = model
            logging.info(f'whisper model loaded in {time.time() - tic:.1f} sec')
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