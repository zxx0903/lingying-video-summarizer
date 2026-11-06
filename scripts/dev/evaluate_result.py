from __future__ import annotations

import argparse
import json
from pathlib import Path

from lingying.evaluation import evaluate_result, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a LingYing result")
    parser.add_argument("result", help="Path to result.json")
    parser.add_argument("ground_truth", help="Important-segment annotation JSON")
    parser.add_argument("--out", help="Optional metrics JSON output")
    args = parser.parse_args()

    metrics = evaluate_result(load_json(args.result), load_json(args.ground_truth))
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
