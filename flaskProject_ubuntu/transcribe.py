import copy
import datetime
import gc
import logging
import os
import subprocess
import tempfile
import threading
import time

import opencc
import srt

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ffprobe 与 ffmpeg 同目录的约定（与 app.py 保持一致；环境变量优先）
FFPROBE_BIN = os.environ.get('FFPROBE_BIN') or os.path.join(
    os.path.dirname(os.path.abspath(subprocess.run(
        ['which', 'ffmpeg'], capture_output=True, text=True).stdout.strip()
        or '/usr/bin/ffmpeg')), 'ffprobe')

# 国内服务器直连 HuggingFace 不通，默认走镜像下载模型；已显式设置时保留用户配置。
# 同时禁用 xet 协议（镜像不支持），改走传统 HTTP 下载。
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

# 中英混合模式标识：上传页选择「中英混合」时 lang 传 'zh+en'。
# Whisper 单次调用只支持一种语言（language 是全局解码约束），自动检测
# 或指定语言后，非主导语言的句子会被直接跳过——实测 28 秒中英交替
# 音频在 auto/zh 下英文句全部丢失（CER 69.6%）。因此混合模式按
# 「VAD 切句 -> 每句独立检测语言(限定 zh/en) -> 分语言转写 -> 拼回全片
# 时间轴」处理，实测同样音频 7 句全部正确识别。
BILINGUAL_LANG = 'zh+en'


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


def _hf_hub_cache_dir():
    """本地 HF 缓存目录（与 huggingface_hub 的解析规则一致，无需 import 它）。"""
    root = os.environ.get('HF_HUB_CACHE')
    if not root:
        home = os.environ.get('HF_HOME', os.path.join(
            os.path.expanduser('~'), '.cache', 'huggingface'))
        root = os.path.join(home, 'hub')
    return root


def _model_is_cached(model_name):
    """判断 faster-whisper 模型是否已有完整本地缓存（含 refs/main 版本指针）。

    已缓存时直接走 local_files_only 加载，完全跳过 huggingface_hub 的
    联网版本校验 —— 外网不通的服务器上，正是那次联网校验导致加载永久挂起。
    """
    cache_dir = os.path.join(
        _hf_hub_cache_dir(), f'models--Systran--faster-whisper-{model_name}')
    return os.path.isfile(os.path.join(cache_dir, 'refs', 'main'))


def _load_model_blocking(args):
    """实际执行模型下载 + 加载（可能较慢），失败抛异常。"""
    # 延迟 import，避免非转写路径也强制拉起 ctranslate2
    from faster_whisper import WhisperModel
    local_only = _model_is_cached(args.whisper_model)
    logging.info(f'Loading faster-whisper model {args.whisper_model} '
                 f'device={args.device or "cpu"} compute_type={_COMPUTE_TYPE} '
                 f'local_files_only={local_only}')
    tic = time.time()
    model = WhisperModel(
        args.whisper_model,
        device=args.device or 'cpu',
        compute_type=_COMPUTE_TYPE,
        local_files_only=local_only)
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


class _FakedInfo:
    """双语路径拼装的转写信息（run_single 只用到 language/duration）。"""

    def __init__(self, language, duration):
        self.language = language
        self.duration = duration
        self.language_probability = 1.0


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

        if (self.args.lang or '') == BILINGUAL_LANG:
            # 中英混合：分段检测语言 + 分段转写（见 BILINGUAL_LANG 注释）
            return self._transcribe_bilingual(input_path)

        # 中文场景用简短初始提示，引导简体输出、稳定标点
        prompt = self.args.prompt
        if not prompt and self.args.lang in ('zh', 'zh-CN', 'chinese'):
            prompt = '以下是普通话的句子。'
        elif not prompt and self.args.lang is None:
            # auto 模式：先探一次语言（只看前 30 秒，代价很小）。
            # 不加引导时 Whisper 中文常输出繁体且标点不稳
            # （实测 base 模型纯中文 CER 23.1%，加引导后 0%）。
            if self._probe_language(input_path) == 'zh':
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

    def _probe_language(self, input_path):
        """探测音频语言（只取前 30 秒），auto 模式决定是否加中文引导。"""
        try:
            from faster_whisper import decode_audio
            tmp = None
            try:
                # ffmpeg 切前 30 秒成临时 wav，避免整片解码占用内存
                fd, tmp = tempfile.mkstemp(suffix='.wav')
                os.close(fd)
                r = subprocess.run(
                    [FFPROBE_BIN, '-v', 'error', '-t', '30', '-i', input_path,
                     '-vn', '-ar', '16000', '-ac', '1', '-y', tmp],
                    capture_output=True, timeout=120)
                if r.returncode != 0:
                    return None
                wave = decode_audio(tmp, sampling_rate=16000)
                if len(wave) < 1600:   # 不足 0.1s 不判
                    return None
                lang, _, _ = self.whisper_model.detect_language(wave)
                return lang
            finally:
                if tmp:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
        except Exception as e:  # 探测失败不影响主流程（退回无引导）
            logging.warning(f'language probe failed: {e}')
            return None

    def _transcribe_bilingual(self, input_path):
        """中英混合转写：VAD 切句 -> 逐句检测语言 -> 分语言转写。

        时间轴 = 各段在全片中的偏移 + 段内相对时间，直接拼回原视频。
        """
        from faster_whisper import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        tic = time.time()
        self._report('transcribing', 0.02, '正在读取音频')
        wave = decode_audio(input_path, sampling_rate=16000)
        total_samples = len(wave)

        self._report('transcribing', 0.05, '正在切分语音段')
        # speech_pad 关键：默认 400ms 会把两侧各 0.4s 的句间静音填满，
        # 相邻句被粘连成整段、语言检测退回主导语言（实测丢英文句）。
        # 收窄到 120ms 并保留句子边界，双语视频才能逐句检测语言。
        vad_opts = VadOptions(min_silence_duration_ms=350, speech_pad_ms=120)
        ts = get_speech_timestamps(wave, vad_opts)
        if not ts:
            # VAD 没切出语音：退回整片按 zh 转写，至少能出结果
            return self._transcribe_monolingual(input_path, 'zh')

        # 合并被 VAD 撕裂的碎片（间隔 <0.2s 视为同句），并限制单段最长 30s
        # （whisper 窗口）。间隔必须收得很小——双语视频句间停顿就是语言切换点。
        merged = []
        for t in ts:
            s, e = t['start'], t['end']
            if merged and s - merged[-1][1] < 16000 * 0.2 \
                    and e - merged[-1][0] <= 16000 * 30:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        chunks = []
        for s, e in merged:
            if e - s <= 16000 * 30:
                chunks.append((s, e))
            else:
                # 超长段按 30s 窗口切（简单等分尾部容差）
                step = 16000 * 30
                c = s
                while c < e:
                    chunks.append((c, min(c + step, e)))
                    c += step

        # ---- 阶段 1：逐段语言检测（仅 encoder 前向，代价小） ----
        # 相邻同语言的段合并成大段后再转写：真实双语视频里语言是
        # 「大块连续」的（整段中文讲解 + 整段英文对白），合并后
        # 转写调用次数从「句数级」降到「语言切换次数级」，长视频提速明显。
        detected = []   # [(start, end, lang)]
        for i, (s, e) in enumerate(chunks):
            if e - s < 8000:   # <0.5s 的碎片跳过
                continue
            try:
                det, _prob, _ = self.whisper_model.detect_language(wave[s:e])
            except Exception:
                det = 'zh'
            if det not in ('zh', 'en'):
                det = 'zh'
            detected.append([s, e, det])
            self._report('transcribing',
                         max(0.05, min(0.4, e / max(total_samples, 1))),
                         f'正在检测语言 {i + 1}/{len(chunks)} 段')

        # ---- 阶段 2：相邻同语言段合并（总长 <=30s whisper 窗口） ----
        lang_groups = []
        for s, e, lang in detected:
            if lang_groups and lang_groups[-1][2] == lang \
                    and e - lang_groups[-1][0] <= 16000 * 30:
                lang_groups[-1][1] = e
            else:
                lang_groups.append([s, e, lang])

        # ---- 阶段 3：逐大段转写，时间轴平移拼回 ----
        results = []
        info_lang, info = 'zh', None
        for i, (s, e, det) in enumerate(lang_groups):
            prompt = '以下是普通话的句子。' if det == 'zh' else ''
            self._report('transcribing',
                         max(0.4, min(0.95, e / max(total_samples, 1))),
                         f'正在识别第 {i + 1}/{len(lang_groups)} 段'
                         f'（{"中文" if det == "zh" else "英文"}）')
            segs_iter, seg_info = self.whisper_model.transcribe(
                wave[s:e], language=det, beam_size=5,
                vad_filter=False, initial_prompt=prompt)
            for seg in segs_iter:
                text = seg.text.strip()
                if not text:
                    continue
                # 拼回全片时间轴（浅拷贝保留全部字段后平移 start/end；
                # Segment 非 NamedTuple，没有 _replace）
                offset = s / 16000.0
                shifted = copy.copy(seg)
                shifted.start = seg.start + offset
                shifted.end = seg.end + offset
                results.append(shifted)
            if i == 0:
                info = seg_info
                info_lang = det

        if not results:
            logging.warning('bilingual transcription produced nothing, '
                            'fallback to monolingual zh')
            return self._transcribe_monolingual(input_path, 'zh')

        # 构造一个与 _transcribe 兼容的 info（仅用到 language/duration 字段）
        info_lang = info_lang or 'zh'
        total_dur = total_samples / 16000.0
        self._report('transcribing', 0.95, f'已识别 {len(results)} 句')
        logging.info(f'Done bilingual transcription: {len(results)} segments '
                     f'in {time.time() - tic:.1f} sec')
        return results, _FakedInfo(info_lang, total_dur)

    def _transcribe_monolingual(self, input_path, lang):
        """单语言兜底转写（双语路径 VAD 无结果/无产出时使用）。"""
        prompt = '以下是普通话的句子。' if lang == 'zh' else ''
        segments, info = self.whisper_model.transcribe(
            input_path, language=lang, beam_size=5,
            vad_filter=self.args.vad, initial_prompt=prompt)
        results = list(segments)
        total_dur = info.duration or 1.0
        for seg in results:
            pct = max(0.1, min(0.95, seg.end / total_dur))
            self._report('transcribing', pct, f'已识别 {len(results)} 句')
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