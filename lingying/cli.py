from __future__ import annotations

import argparse
import json
from pathlib import Path

from lingying.config import PipelineConfig, Weights
from lingying.pipeline import run_pipeline
from lingying.templates import TEMPLATES_BY_ID


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
    parser.add_argument(
        "--template",
        default="education_general",
        choices=sorted(TEMPLATES_BY_ID),
        help="场景模板 ID",
    )
    parser.add_argument(
        "--keywords",
        default="",
        help="自定义主题关键词，使用逗号分隔",
    )
    parser.add_argument("--max-chapters", type=int, default=8, help="最大章节数")
    parser.add_argument(
        "--height",
        type=int,
        choices=(480, 720, 1080),
        default=None,
        help="输出视频高度；不填则保持原视频尺寸",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("mp4", "mov", "avi"),
        default="mp4",
        help="输出片段和预览格式",
    )
    parser.add_argument("--no-asr", action="store_true", help="禁用 Whisper 转写")
    parser.add_argument("--no-ocr", action="store_true", help="禁用屏幕文字 OCR")
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="启用视觉语言模型补充画面语义",
    )
    parser.add_argument(
        "--vlm-backend",
        choices=("qwen-vl", "openai-compatible"),
        default="qwen-vl",
        help="VLM 后端",
    )
    parser.add_argument(
        "--vlm-model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="VLM 模型名或本地目录",
    )
    args = parser.parse_args()

    config = PipelineConfig(
        target_seconds=args.target,
        target_ratio=args.ratio,
        weights=Weights(args.text, args.audio, args.visual),
        asr_enabled=not args.no_asr,
        ocr_enabled=not args.no_ocr,
        vlm_enabled=args.vlm,
        vlm_backend=args.vlm_backend,
        vlm_model=args.vlm_model,
        template_id=args.template,
        custom_keywords=[
            item.strip() for item in args.keywords.split(",") if item.strip()
        ],
        max_chapters=max(1, args.max_chapters),
        output_height=args.height,
        output_format=args.output_format,
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
