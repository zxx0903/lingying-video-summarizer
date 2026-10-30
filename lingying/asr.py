from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Cue:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.01, self.end - self.start)

    def to_dict(self) -> dict:
        return asdict(self)


_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})"
)


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt_text(text: str) -> list[Cue]:
    cues: list[Cue] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip())
    for block in blocks:
        lines = [ln.strip("\ufeff").strip() for ln in block.split("\n") if ln.strip()]
        if len(lines) < 2:
            continue
        time_line = next((ln for ln in lines if "-->" in ln), None)
        if not time_line:
            continue
        m = _TIME_RE.search(time_line)
        if not m:
            continue
        start = _to_seconds(*m.groups()[:4])
        end = _to_seconds(*m.groups()[4:])
        idx = lines.index(time_line)
        content = " ".join(lines[idx + 1 :]).strip()
        content = re.sub(r"<[^>]+>", "", content)
        if content:
            cues.append(Cue(start=start, end=max(end, start + 0.2), text=content))
    cues.sort(key=lambda c: c.start)
    return cues


def parse_srt(path: str | Path) -> list[Cue]:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return parse_srt_text(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    return parse_srt_text(raw.decode("utf-8", errors="ignore"))


def find_sidecar_srt(video: str | Path) -> Path | None:
    video = Path(video)
    candidates = [
        video.with_suffix(".srt"),
        video.with_suffix(".zh.srt"),
        video.with_suffix(".chs.srt"),
        video.parent / (video.stem + ".srt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


_WHISPER_CACHE: dict[str, object] = {}


def _resolve_whisper_model(model_size: str) -> str:
    configured = os.environ.get("LINGYING_WHISPER_MODEL", "").strip()
    if configured:
        return configured
    pointer = _PROJECT_ROOT / "runtime" / "whisper_model_path.txt"
    if pointer.exists():
        configured = pointer.read_text(encoding="utf-8-sig").strip()
        if configured:
            return configured
    return model_size


def transcribe_with_whisper(
    audio_path: str | Path,
    model_size: str = "small",
    language: str | None = None,
) -> list[Cue]:
    """Transcribe with faster-whisper. Raises if the optional stack is missing."""
    from faster_whisper import WhisperModel

    model_source = _resolve_whisper_model(model_size)
    key = f"{model_source}:cpu"
    model = _WHISPER_CACHE.get(key)
    if model is None:
        model = WhisperModel(model_source, device="cpu", compute_type="int8")
        _WHISPER_CACHE[key] = model

    segments, _info = model.transcribe(
        str(audio_path),
        language=language or None,
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,
    )
    cues: list[Cue] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            cues.append(Cue(float(seg.start), float(seg.end), text))
    return cues


def format_timestamp(seconds: float, sep: str = ",") -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        s += 1
        ms -= 1000
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def cues_to_srt(cues: list[Cue]) -> str:
    lines: list[str] = []
    for i, cue in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(
            f"{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}"
        )
        lines.append(cue.text.strip())
        lines.append("")
    return "\n".join(lines)
