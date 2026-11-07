from __future__ import annotations

import html
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


_TIME_TOKEN = r"(?:(\d{1,2}):)?(\d{2}):(\d{2})[.,](\d{1,3})"
_TIME_RE = re.compile(
    _TIME_TOKEN + r"\s*-->\s*" + _TIME_TOKEN
)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _to_seconds(h: str | None, m: str, s: str, ms: str) -> float:
    return (
        int(h or 0) * 3600
        + int(m) * 60
        + int(s)
        + int(ms.ljust(3, "0")) / 1000.0
    )


def _clean_cue_text(text: str) -> str:
    text = html.unescape(text)
    text = _TAG_RE.sub("", text)
    return _SPACE_RE.sub(" ", text).strip()


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _overlap_length(previous: str, current: str) -> int:
    """Return the longest suffix/prefix overlap between two caption windows."""
    previous = previous.lower()
    current = current.lower()
    for length in range(min(len(previous), len(current)), 0, -1):
        if previous.endswith(current[:length]):
            return length
    return 0


def deduplicate_rolling_cues(cues: list[Cue]) -> list[Cue]:
    """Collapse YouTube-style rolling caption windows into incremental cues.

    Rolling captions repeat most of the previous line in the next cue. Removing
    the suffix/prefix overlap preserves the transcript order while preventing
    the same sentence from entering keyword extraction and summaries repeatedly.
    """
    output: list[Cue] = []
    previous_full = ""
    for cue in cues:
        text = _clean_cue_text(cue.text)
        if not text:
            continue
        if previous_full and text == previous_full:
            if output:
                output[-1].end = max(output[-1].end, cue.end)
            continue

        emitted_text = text
        if previous_full:
            overlap = _overlap_length(previous_full, text)
            compact_overlap = _compact_text(previous_full[-overlap:])
            shorter_cue = min(
                len(_compact_text(previous_full)),
                len(_compact_text(text)),
            )
            min_overlap = (
                3
                if re.search(r"[\u3400-\u9fff]", compact_overlap)
                else 7
            )
            # A one-word rolling cue such as "shown." often spans the whole
            # shorter caption. Accept that overlap even though it is short.
            if shorter_cue >= 4 and len(compact_overlap) >= shorter_cue * 0.85:
                min_overlap = 3
            if (
                overlap >= min_overlap
                and len(compact_overlap)
                >= shorter_cue * 0.35
            ):
                emitted_text = text[overlap:].lstrip(" .,;:!?，。；：！？")

        emitted_text = _clean_cue_text(emitted_text)
        previous_full = text
        if not emitted_text:
            if output:
                output[-1].end = max(output[-1].end, cue.end)
            continue
        output.append(
            Cue(
                start=cue.start,
                end=max(cue.end, cue.start + 0.2),
                text=emitted_text,
            )
        )
    return output


def parse_subtitle_text(text: str) -> list[Cue]:
    cues: list[Cue] = []
    normalized = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .lstrip("\ufeff")
        .strip()
    )
    blocks = re.split(r"\n\s*\n", normalized)
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
        content = _clean_cue_text(" ".join(lines[idx + 1 :]))
        if content:
            cues.append(Cue(start=start, end=max(end, start + 0.2), text=content))
    cues.sort(key=lambda c: c.start)
    return deduplicate_rolling_cues(cues)


# Backward-compatible name used by existing imports and tests.
parse_srt_text = parse_subtitle_text


def parse_subtitle(path: str | Path) -> list[Cue]:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return parse_subtitle_text(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    return parse_subtitle_text(raw.decode("utf-8", errors="ignore"))


# Kept for callers that predate WebVTT support.
parse_srt = parse_subtitle


def _language_priority(language: str) -> int:
    language = language.lower()
    if language in {"zh", "zho", "chi", "chs", "cht", "zh-cn", "zh-hans", "zh-hant"}:
        return 0
    if language in {"en", "eng"}:
        return 1
    return 2


def find_sidecar_subtitle(video: str | Path) -> Path | None:
    video = Path(video)
    matches: list[tuple[tuple[int, int, int, str], Path]] = []
    for path in video.parent.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".srt", ".vtt"}:
            continue
        base = path.name[: -len(path.suffix)]
        if base == video.stem:
            language = ""
        elif base.startswith(video.stem + "."):
            language = base[len(video.stem) + 1 :]
        else:
            continue
        matches.append(
            (
                (
                    _language_priority(language),
                    0 if path.suffix.lower() == ".srt" else 1,
                    0 if not language else 1,
                    path.name.lower(),
                ),
                path,
            )
        )
    return min(matches, key=lambda item: item[0])[1] if matches else None


# Backward-compatible alias; now also discovers language-suffixed SRT/VTT files.
find_sidecar_srt = find_sidecar_subtitle


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
