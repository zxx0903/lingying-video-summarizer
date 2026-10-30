import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "outputs" / "demo_run" / "result.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    print(
        f"{'idx':>3} {'time':>11} {'score':>6} {'text':>6} {'audio':>6} "
        f"{'visual':>6} {'cue':>5} {'kw':>5} {'rms':>7} {'edge':>6} {'scene':>5} sel  keywords"
    )
    for s in data["segments"]:
        m = s["modality"]
        ts = s["text_stats"]
        au = s["audio_stats"]
        vi = s["visual_stats"]
        print(
            f"{s['index']:>3} {s['start']:>5.0f}-{s['end']:<5.0f} {s['score']:>6.3f} "
            f"{m['text']:>6.3f} {m['audio']:>6.3f} {m['visual']:>6.3f} "
            f"{ts['cue']:>5.2f} {ts['keyword_hit']:>5.2f} {au['rms']:>7.4f} "
            f"{vi['edge_density']:>6.4f} {s.get('scene_id', -1):>5} "
            f"{'Y' if s.get('selected') else '.':>3}  {','.join(s['keywords'][:3])}"
        )


if __name__ == "__main__":
    main()
