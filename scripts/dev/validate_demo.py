import json
import sys
from pathlib import Path

from lingying.ffmpeg_utils import probe

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "demo_final"
    data = json.loads((out / "result.json").read_text(encoding="utf-8"))
    errors = []

    if not data.get("summary", {}).get("chapters"):
        errors.append("no chapters")
    if not data.get("selected"):
        errors.append("no selected segments")
    if not (out / "result.json").exists():
        errors.append("missing result.json")
    if not (out / "summary.srt").exists():
        errors.append("missing summary.srt")
    if not (out / "timeline.png").exists():
        errors.append("missing timeline.png")

    for clip in data.get("clips", []):
        p = Path(clip["path"])
        if not p.is_absolute():
            p = out / p
        if not p.exists():
            errors.append(f"missing clip {p}")
        else:
            try:
                info = probe(p)
                if info.duration <= 1:
                    errors.append(f"clip too short {p}")
            except Exception as exc:
                errors.append(f"probe failed {p}: {exc}")

    preview = data.get("artifacts", {}).get("preview")
    if preview:
        p = Path(preview)
        if not p.is_absolute():
            p = out / p
        if not p.exists():
            errors.append("missing preview")
        else:
            info = probe(p)
            print(f"preview: {info.duration:.1f}s, {info.width}x{info.height}")

    m = data["metrics"]
    print(
        f"segments={m['segment_count']} selected={m['selected_count']} "
        f"summary={m['selected_duration']:.1f}s ratio={m['compression_ratio']:.1%} "
        f"processing={m['processing_seconds']:.1f}s"
    )
    print(f"chapters={len(data['summary']['chapters'])}")
    if errors:
        print("FAILED:")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
