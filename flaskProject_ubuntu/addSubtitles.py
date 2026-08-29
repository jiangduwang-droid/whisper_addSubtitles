"""用 ffmpeg 把 srt 字幕烧录进视频（Ubuntu 版）。

替换 moviepy 的原因：
- moviepy 的 TextClip 每条字幕都走 ImageMagick 渲染并 Python 逐帧拼合，
  最后 write_videofile 整片重编码，大视频慢到不可用且吃内存。
- ffmpeg subtitles 滤镜在 C 层叠加，只重编码画面、音频 copy 直通，快一个数量级。

约定：源视频、srt、输出必须位于同一任务目录，文件名不含特殊字符
（由 app.py 的任务沙箱保证），避免 ffmpeg 滤镜对路径特殊字符的解析问题。
Linux 上如无 SimHei，请先安装 CJK 字体（如 Noto Sans CJK）并设置
环境变量 SUBTITLE_FONT，例如：
   export SUBTITLE_FONT="Noto Sans CJK SC"
"""

import os
import re
import subprocess

FFMPEG = os.environ.get('FFMPEG_BIN', 'ffmpeg')
FONT_NAME = os.environ.get('SUBTITLE_FONT', 'Noto Sans CJK SC')


def _hex_to_ass(color_hex, alpha='00'):
    """把 '#RRGGBB' 转成 ASS 颜色 '&H00BBGGRR&'（BGR 反序 + 前导 alpha）。"""
    c = (color_hex or '#FFFFFF').lstrip('#')
    if len(c) != 6:
        c = 'FFFFFF'
    r, g, b = c[0:2], c[2:4], c[4:6]
    return f'&H{alpha}{b}{g}{r}&'


# ---------------------------------------------------------------------------
# 自定义字幕位置（人工校对页拖拽/预设得到）：
#
# 位置用 pos_x / pos_y（0~1 浮点）表示字幕文本块中心在画面中的相对坐标。
# force_style 只能表达 Alignment + Margin，无法精确落点，
# 因此带位置时改走 ASS 文件：每条 Dialogue 用 \pos(x,y) 显式定位，
# Style 的 Alignment=5（中心锚点）保证 \pos 锚定的是文本块中心，
# 与前端 CSS left/top + translate(-50%,-50%) 的预览口径完全一致。
#
# PlayRes 沿用 384x288（ffmpeg 把 srt 转 ass 的同一基准），
# FontSize 数值口径不变，字号观感与无位置时的 force_style 路径一致。
# （2026-08-29 实测标定：两条路径文字中心偏差 < 0.3% 画面高度）
# ---------------------------------------------------------------------------
ASS_PLAY_X, ASS_PLAY_Y = 384, 288
_SRT_TS_RE = re.compile(
    r'(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)')

# ---------------------------------------------------------------------------
# 长文本预换行 + 分区对齐：
#
# 实测（2026-08-29，ffmpeg 4.4 libass + Noto Sans CJK SC）：libass 对无空格的
# 中文长句不会自动换行——\pos 定位下单行向两侧溢出画面且锚点漂移到画面中心；
# 默认路径（无 \pos）同样单行满宽。因此带自定义位置时在 Python 侧按字符
# 宽度估算预换行（\N），并按位置所在区域选择对齐锚点，保证文本块完整落在
# 画面内、且预览与烧录口径一致：
#   pos_x <= 0.35 左区：Alignment=4，\pos 锚文本块左边中点，文字向右延伸
#   pos_x >= 0.65 右区：Alignment=6，\pos 锚文本块右边中点，文字向左延伸
#   其余中区：Alignment=5，\pos 锚文本块中心，两侧对称延伸
# （三者垂直方向都锚文本块中心，与前端 translate(?,-50%) 口径一致）
#
# 字符宽度标定：字号 32 时实测中文每字 21.6 PlayRes 单位（系数 0.675），
# 英文小写 10.0、大写 13.6、数字 11.9。取 0.70 为全宽系数并高估西文，
# 宁可早换行也不让文字溢出画面。
# ---------------------------------------------------------------------------
ADVANCE_FACTOR = 0.70
_CJK_RANGES = (
    (0x1100, 0x11FF),   # 谚文字母
    (0x2E80, 0x9FFF),   # CJK 部首/汉字/注音
    (0xA000, 0xA4CF),   # 彝文
    (0xAC00, 0xD7FF),   # 谚文音节
    (0xF900, 0xFAFF),   # CJK 兼容表意
    (0xFF01, 0xFF60),   # 全角形式
    (0xFFE0, 0xFFE6),
)


def _is_cjk(ch):
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _char_units(ch):
    """单字符宽度，单位 = ADVANCE_FACTOR * font_size。

    西文按保守高估（实测小写 0.31、大写 0.42、数字 0.37），
    换行偏早不留溢出风险；全角标点与汉字同为 1.0。
    """
    if ch == ' ':
        return 0.35
    if _is_cjk(ch):
        return 1.0
    if ch.isupper() or ch.isdigit():
        return 0.72
    return 0.55


def _split_tokens(text):
    """token 化：英文单词整体保留（不在词中间断行），CJK 逐字，空格独立。"""
    tokens, word = [], []
    for ch in text:
        if _is_cjk(ch) or ch.isspace():
            if word:
                tokens.append(''.join(word))
                word = []
            tokens.append(ch)
        else:
            word.append(ch)
    if word:
        tokens.append(''.join(word))
    return tokens


def _zone_of(pos_x):
    """位置所在区域：'left' / 'center' / 'right'。"""
    if pos_x <= 0.35:
        return 'left'
    if pos_x >= 0.65:
        return 'right'
    return 'center'


def _zone_alignment(pos_x):
    """位置区域 -> ASS Alignment（numpad 布局，垂直均为居中）：
    4=左边中点锚，5=中心锚，6=右边中点锚。与前端 transform 口径一一对应。"""
    return {'left': 4, 'center': 5, 'right': 6}[_zone_of(pos_x)]


def wrap_text(text, pos_x, font_size):
    """按位置区域可用宽度预换行，返回 '\\N' 连接的多行文本。

    可用宽度（PlayRes 单位）：
      左区：位置点 -> 画面右缘（右边距 12）
      右区：画面左缘 -> 位置点（左边距 12）
      中区：位置点向两侧对称延伸（两侧边距各 12）
    单个超宽 token（超长英文单词）只能整词成行，属可接受的边界情况。
    """
    x = pos_x * ASS_PLAY_X
    zone = _zone_of(pos_x)
    if zone == 'left':
        max_w = ASS_PLAY_X - x - 12
    elif zone == 'right':
        max_w = x - 12
    else:
        max_w = 2 * min(x, ASS_PLAY_X - x) - 24
    unit = ADVANCE_FACTOR * font_size
    max_units = max(4.0, max_w / unit)

    lines = []
    # 用户在编辑器里手动输入的换行保留为独立段落，段内再预换行
    for seg in text.split('\n'):
        cur, cur_units = '', 0.0
        for tok in _split_tokens(seg):
            u = sum(_char_units(c) for c in tok)
            if cur and cur_units + u > max_units:
                lines.append(cur.rstrip())
                cur, cur_units = '', 0.0
                if tok.isspace():
                    continue
            if not cur and tok.isspace():
                continue
            cur += tok
            cur_units += u
        if cur:
            lines.append(cur.rstrip())

    lines = [l for l in lines if l.strip()]
    return '\\N'.join(lines) if lines else text


def _parse_srt(path):
    """轻量 SRT 解析：[(start秒, end秒, 文本多行), ...]，容错损坏块。"""
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
        content = f.read()
    out = []
    for block in re.split(r'\r?\n\s*\r?\n', content.strip()):
        lines = [l for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        m = None
        idx = 0
        for i, line in enumerate(lines):
            m = _SRT_TS_RE.search(line)
            if m:
                idx = i
                break
        if m is None:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        text = '\n'.join(l.strip() for l in lines[idx + 1:] if l.strip())
        if text and end > start:
            out.append((start, end, text))
    return out


def _ass_timestamp(sec):
    """秒 -> ASS 时间戳 'H:MM:SS.cc'（厘秒）。"""
    cs = int(round(sec * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f'{h}:{m:02d}:{s:02d}.{c:02d}'


def _sanitize_ass_text(t):
    """花括号是 ASS override 语法，替换成全角防注入；换行转 \\N。"""
    t = t.replace('{', '｛').replace('}', '｝')
    return t.replace('\n', r'\N')


def _write_ass(srt_path, style):
    """按 style 的位置（pos_x/pos_y）把 srt 转成带 \\pos 的 ass，返回路径。

    分区对齐：左区 Alignment=4（\pos 锚文本块左边中点）、右区 6（右边中点）、
    中区 5（中心），与前端 overlay 的 transform 锚点一一对应；
    长文本先 wrap_text 预换行，避免 libass 不换行导致溢出画面。
    """
    pos_x = float(style['pos_x'])
    pos_y = float(style['pos_y'])
    font_size = int(style.get('font_size', 32))
    text_color = _hex_to_ass(style.get('text_color', '#FFFFFF')).rstrip('&')
    outline_color = _hex_to_ass(style.get('outline_color', '#000000')).rstrip('&')
    alignment = _zone_alignment(pos_x)
    x = round(pos_x * ASS_PLAY_X, 2)
    y = round(pos_y * ASS_PLAY_Y, 2)

    events = []
    for s, e, t in _parse_srt(srt_path):
        wrapped = wrap_text(t, pos_x, font_size)
        events.append(
            f'Dialogue: 0,{_ass_timestamp(s)},{_ass_timestamp(e)},Default,,'
            f'0,0,0,,{{\\pos({x},{y})}}{_sanitize_ass_text(wrapped)}'
        )
    if not events:
        raise RuntimeError('字幕文件解析后没有可烧录的条目')

    header = '\n'.join([
        '[Script Info]',
        'ScriptType: v4.00+',
        f'PlayResX: {ASS_PLAY_X}',
        f'PlayResY: {ASS_PLAY_Y}',
        'ScaledBorderAndShadow: yes',
        '',
        '[V4+ Styles]',
        'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, '
        'OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, '
        'ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, '
        'Alignment, MarginL, MarginR, MarginV, Encoding',
        f'Style: Default,{FONT_NAME},{font_size},{text_color},'
        f'{text_color},{outline_color},&H00000000,0,0,0,0,100,100,0,0,'
        f'1,1.5,1,{alignment},10,10,10,1',
        '',
        '[Events]',
        'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, '
        'Effect, Text',
    ])
    ass_path = os.path.splitext(srt_path)[0] + '.ass'
    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n' + '\n'.join(events) + '\n')
    return ass_path


def _has_position(style):
    """style 是否携带有效自定义位置。"""
    try:
        return (style.get('pos_x') is not None
                and style.get('pos_y') is not None
                and 0.0 < float(style['pos_x']) < 1.0
                and 0.0 < float(style['pos_y']) < 1.0)
    except (TypeError, ValueError, AttributeError):
        return False


def _build_extra_args(video_path, srt_path, out_path, style=None):
    """拼出 ffmpeg 命令行参数，全部走 list 传参，不经过 shell，杜绝注入。

    style 可包含：font_size, text_color, outline_color, alignment, margin_v,
    以及人工校对页设置的自定义位置 pos_x/pos_y（0~1，文本块中心比例）。
    颜色需为 '#RRGGBB' 形式，其余为数值，均由上游 app 层校验后传入。
    """
    style = style or {}

    if _has_position(style):
        # 自定义位置：srt -> ass（每条 \pos 定位），滤镜直接用 ass 自带样式
        ass_path = _write_ass(srt_path, style)
        ass_rel = os.path.basename(ass_path)
        vf = f'subtitles={ass_rel}'
    else:
        # 默认路径：srt + force_style（底部居中 + MarginV），与历史行为一致
        font_size = int(style.get('font_size', 32))
        text_color = _hex_to_ass(style.get('text_color', '#FFFFFF'))
        outline_color = _hex_to_ass(style.get('outline_color', '#000000'))
        alignment = int(style.get('alignment', 2))       # 2=底部居中
        margin_v = int(style.get('margin_v', 80))        # 底部间距（像素）
        w_margin = int(style.get('margin_l', 10))
        srt_rel = os.path.basename(srt_path)
        vf = (
            f"subtitles={srt_rel}:force_style='FontName={FONT_NAME},"
            f"FontSize={font_size},Alignment={alignment},MarginV={margin_v},"
            f"MarginL={w_margin},MarginR={w_margin},"
            f"PrimaryColour={text_color},OutlineColour={outline_color},"
            f"Outline=1.5,Shadow=1'"
        )
    return [
        '-y',
        '-i', video_path,
        '-vf', vf,
        '-map', '0:v:0',
        '-map', '0:a?',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '23',
        '-c:a', 'copy',
        '-movflags', '+faststart',
        out_path,
    ]


class RealizeAddSubtitles:
    def __init__(self, video_file, txt_file, out_path=None, style=None):
        self.src_video = os.path.abspath(video_file)
        self.srt = os.path.abspath(txt_file)
        self.out_path = out_path or self._default_out_path()
        self.style = style or {}

    def _default_out_path(self):
        base, ext = os.path.splitext(self.src_video)
        return f'{base}_srt{ext}'

    def burn(self, progress_cb=None):
        if not os.path.isfile(self.src_video):
            raise FileNotFoundError(f'源视频不存在：{self.src_video}')
        if not os.path.isfile(self.srt):
            raise FileNotFoundError(f'字幕文件不存在：{self.srt}')
        if os.path.getsize(self.srt) == 0:
            raise RuntimeError('字幕文件为空（未识别到语音），无内容可烧录')

        cmd = [FFMPEG, *_build_extra_args(self.src_video, self.srt, self.out_path, self.style)]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, encoding='utf-8', errors='replace',
            cwd=os.path.dirname(self.srt))

        total_sec = self._probe_duration(self.src_video) or 1.0
        last = [0.0]
        stderr_tail = []

        def _report(pct):
            if progress_cb:
                progress_cb(pct)

        for line in proc.stdout:
            stderr_tail.append(line.rstrip())
            if len(stderr_tail) > 12:
                stderr_tail.pop(0)
            m = re.search(r'time=(\d+):(\d+):(\d+)', line)
            if m:
                t = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
                pct = min(100.0, t / total_sec * 100.0)
                if pct - last[0] >= 1.0:
                    last[0] = pct
                    _report(pct)

        ret = proc.wait()
        if ret != 0 or not os.path.isfile(self.out_path):
            detail = ('\n' + '\n'.join(stderr_tail[-8:])) if stderr_tail else ''
            raise RuntimeError(
                f'ffmpeg 烧录字幕失败（退出码 {ret}）{detail}')

        _report(100)
        return self.out_path

    @staticmethod
    def _probe_duration(path):
        try:
            out = subprocess.run(
                [FFMPEG, '-i', path], capture_output=True, text=True).stderr
            m = re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)', out)
            if m:
                return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
        except Exception:  # noqa: BLE001 - 探测失败可回退
            pass
        return 0.0