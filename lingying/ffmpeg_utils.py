from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class MediaInfo:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    video_codec: str = ""
    audio_codec: str = ""
    has_subtitles: bool = False
    subtitle_codec: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def ffmpeg_exe() -> str:
    env = os.environ.get("FFMPEG_BINARY")
    if env and Path(env).exists():
        return env
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if found:
            return found
        raise RuntimeError("找不到 ffmpeg。请安装 imageio-ffmpeg 或设置 FFMPEG_BINARY。")


def run_ffmpeg(args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    cmd = [ffmpeg_exe(), "-hide_banner", "-y"] + args
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-1500:]
        raise RuntimeError(f"ffmpeg 失败: {' '.join(args[:6])}\n{tail}")
    return proc


def _parse_rate(value: str) -> float:
    if not value:
        return 0.0
    if "/" in value:
        a, b = value.split("/", 1)
        try:
            return float(a) / float(b) if float(b) else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe(path: str | Path) -> MediaInfo:
    cmd = [ffmpeg_exe(), "-hide_banner", "-i", str(path)]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore"
    )
    text = (proc.stderr or "") + (proc.stdout or "")

    duration = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))

    width = height = 0
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", text)
    if m:
        width, height = int(m.group(1)), int(m.group(2))

    fps = 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", text)
    if m:
        fps = float(m.group(1))

    has_audio = "Audio:" in text
    vcodec = ""
    m = re.search(r"Video:\s*([a-zA-Z0-9_\-]+)", text)
    if m:
        vcodec = m.group(1)
    acodec = ""
    m = re.search(r"Audio:\s*([a-zA-Z0-9_\-]+)", text)
    if m:
        acodec = m.group(1)
    subtitle_codec = ""
    m = re.search(r"Subtitle:\s*([a-zA-Z0-9_\-]+)", text)
    if m:
        subtitle_codec = m.group(1)

    if duration <= 0:
        raise RuntimeError(f"无法读取视频时长: {path}")

    return MediaInfo(
        path=str(path),
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        has_audio=has_audio,
        video_codec=vcodec,
        audio_codec=acodec,
        has_subtitles=bool(subtitle_codec),
        subtitle_codec=subtitle_codec,
    )


def extract_audio(video: str | Path, wav: str | Path) -> None:
    run_ffmpeg(
        [
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav),
        ]
    )


def extract_embedded_subtitle(video: str | Path, subtitle: str | Path) -> None:
    """Extract the first embedded subtitle stream as SRT."""
    run_ffmpeg(
        [
            "-i",
            str(video),
            "-map",
            "0:s:0",
            "-c:s",
            "srt",
            str(subtitle),
        ]
    )


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    fps: float = 1.0,
    width: int = 480,
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pattern = out / "frame_%06d.jpg"
    run_ffmpeg(
        [
            "-i",
            str(video),
            "-vf",
            f"fps={fps},scale={width}:-2",
            "-q:v",
            "3",
            str(pattern),
        ]
    )
    return sorted(out.glob("frame_*.jpg"))


def render_clip(
    video: str | Path,
    start: float,
    end: float,
    out_path: str | Path,
    height: int | None = None,
) -> None:
    duration = max(0.4, float(end) - float(start))
    suffix = Path(out_path).suffix.lower()
    video_filters = []
    if height:
        height = max(144, int(height))
        video_filters.append(f"scale=-2:{height}")
    if suffix == ".avi":
        video_codec = ["-c:v", "mpeg4", "-q:v", "4"]
        audio_codec = ["-c:a", "libmp3lame", "-b:a", "128k"]
    else:
        video_codec = [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
        ]
        audio_codec = ["-c:a", "aac", "-b:a", "128k"]
    run_ffmpeg(
        [
            "-ss",
            f"{max(0.0, start):.3f}",
            "-i",
            str(video),
            "-t",
            f"{duration:.3f}",
            *(["-vf", ",".join(video_filters)] if video_filters else []),
            *video_codec,
            *audio_codec,
            "-avoid_negative_ts",
            "make_zero",
            str(out_path),
        ]
    )


def concat_clips(clips: list[str | Path], out_path: str | Path) -> None:
    if not clips:
        raise ValueError("没有可拼接的片段")
    list_file = Path(out_path).with_suffix(".txt")
    lines = []
    for clip in clips:
        safe = str(Path(clip).resolve()).replace("'", "'\\''")
        lines.append(f"file '{safe}'")
    list_file.write_text("\n".join(lines), encoding="utf-8")
    run_ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(out_path),
        ]
    )


def extract_frame_at(video: str | Path, timestamp: float, out_path: str | Path) -> None:
    run_ffmpeg(
        [
            "-ss",
            f"{max(0.0, timestamp):.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out_path),
        ]
    )


def write_json(path: str | Path, data: dict) -> None:
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
