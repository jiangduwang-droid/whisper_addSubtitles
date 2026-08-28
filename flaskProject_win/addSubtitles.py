"""用 ffmpeg 把 srt 字幕烧录进视频。

为什么替换 moviepy：
- moviepy 的 TextClip 每条字幕都走 ImageMagick 渲染，再用 Python 逐帧拼合，
  最后 write_videofile 会整体重编码，大视频慢到不可用且吃掉大量内存。
- ffmpeg 的 subtitles 滤镜在 C 层一帧一步完成叠加，只重编码画面不重编码音频，
  速度快一个数量级，且兼容性更好。

为避免 srt 路径里出现冒号/反斜杠等被 ffmpeg 滤镜解析掉的问题，
本类约定：源视频、srt、输出文件必须位于同一任务目录，且文件名为
不含特殊字符的固定值（由 app.py 的任务沙箱保证）。
"""

import os
import re
import subprocess

FFMPEG = os.environ.get('FFMPEG_BIN', 'ffmpeg')
FONT_NAME = os.environ.get('SUBTITLE_FONT', 'SimHei')


def _build_extra_args(video_path, srt_path, out_path, w_margin=10, h_bottom=80):
    """拼出 ffmpeg 命令行参数，全部走 list 传参，不经过 shell，杜绝注入。"""
    srt_rel = os.path.basename(srt_path)  # 与视频同目录，直接引用相对名
    # subtitles 滤镜：画中叠加 + 底部字幕样式。引号用于闭合滤镜语法里的特殊字符。
    vf = (
        f"subtitles={srt_rel}:force_style='FontName={FONT_NAME},"
        f"FontSize=32,Alignment=2,MarginV={h_bottom},MarginL={w_margin},"
        f"MarginR={w_margin},PrimaryColour=&H00FFFFFF&,OutlineColour=&H00000000&,"
        f"Outline=1.5,Shadow=1'"
    )
    # 版权：只重编码有字幕叠加的视频，音频用 copy 直通，明显快于整体重编码
    return [
        '-y',
        '-i', video_path,
        '-vf', vf,
        '-map', '0:v:0',
        '-map', '0:a?',
        '-c:v', 'libx264',
        '-preset', 'veryfast',      # 速度优先，兼顾画质
        '-crf', '23',
        '-c:a', 'copy',
        '-movflags', '+faststart',
        out_path,
    ]


class RealizeAddSubtitles:
    def __init__(self, video_file, txt_file, out_path=None):
        self.src_video = os.path.abspath(video_file)
        self.srt = os.path.abspath(txt_file)
        self.out_path = out_path or self._default_out_path()

    def _default_out_path(self):
        base, ext = os.path.splitext(self.src_video)
        return f'{base}_srt{ext}'

    def burn(self, progress_cb=None):
        """执行烧录。返回最终视频路径。"""
        if not os.path.isfile(self.src_video):
            raise FileNotFoundError(f'源视频不存在：{self.src_video}')
        if not os.path.isfile(self.srt):
            raise FileNotFoundError(f'字幕文件不存在：{self.srt}')
        if os.path.getsize(self.srt) == 0:
            raise RuntimeError('字幕文件为空（未识别到语音），无内容可烧录')

        cmd = [FFMPEG, *_build_extra_args(self.src_video, self.srt, self.out_path)]
        # 长任务进度从 ffmpeg 的 `frame=...  time=` 行解析，按时长比例换算
        # cwd 设为字幕所在目录，使滤镜里的相对文件名能被正确解析
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
            # 保留最近若干行输出，失败时用于排障
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
        except Exception:  # noqa: BLE001 - 探测失败可回退，不影响主流程
            pass
        return 0.0