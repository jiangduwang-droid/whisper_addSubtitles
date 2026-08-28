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


def _build_extra_args(video_path, srt_path, out_path, style=None):
    """拼出 ffmpeg 命令行参数，全部走 list 传参，不经过 shell，杜绝注入。

    style 可包含：font_size, text_color, outline_color, alignment, margin_v。
    颜色需为 '#RRGGBB' 形式，其余为整数，均由上游 app 层校验后传入。
    """
    style = style or {}
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