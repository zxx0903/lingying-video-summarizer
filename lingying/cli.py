from __future__ import annotations

import argparse
import json
from pathlib import Path

from lingying.config import PipelineConfig, Weights
from lingying.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="灵影速览 · 教育视频智能摘要")
    parser.add_argument("--input", required=True, help="输入视频路径")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--srt", default=None, help="可选字幕文件")
    parser.add_argument("--target", type=float, default=None, help="目标摘要秒数")
    parser.add_argument("--ratio", type=float, default=0.25, help="目标摘要比例")
    parser.add_argument("--text", type=float, default=0.45, help="文本模态权重")
    parser.add_argument("--audio", type=float, default=0.30, help="音频模态权重")
    parser.add_argument("--visual", type=float, default=0.25, help="视觉模态权重")
    parser.add_argument("--no-asr", action="store_true", help="禁用 Whisper 转写")
    parser.add_argument("--no-ocr", action="store_true", help="禁用屏幕文字 OCR")
    args = parser.parse_args()

    config = PipelineConfig(
        target_seconds=args.target,
        target_ratio=args.ratio,
        weights=Weights(args.text, args.audio, args.visual),
        asr_enabled=not args.no_asr,
        ocr_enabled=not args.no_ocr,
    )

    def progress(stage: str, pct: float) -> None:
        print(f"[{pct * 100:5.1f}%] {stage}")

    result = run_pipeline(
        args.input, args.out, config=config, srt_path=args.srt, progress=progress
    )
    print("\n=== 摘要 ===")
    print(result["summary"]["overview"])
    print("\n=== 章节 ===")
    for ch in result["summary"]["chapters"]:
        print(f"- {ch['time']}  {ch['title']}")
    print("\n=== 指标 ===")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"\n结果目录: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
