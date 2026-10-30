from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from lingying.asr import Cue, cues_to_srt  # noqa: E402
from lingying.ffmpeg_utils import concat_clips, probe, run_ffmpeg  # noqa: E402

INPUT_DIR = ROOT / "data" / "input"
BUILD_DIR = ROOT / "data" / "cache" / f"demo_build_{int(time.time())}"

FONT_PATHS = [
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]

SLIDES = [
    {
        "title": "什么是机器学习？",
        "bullets": [
            "从数据中自动学习规律",
            "用模型对新样本做预测",
            "重点：区别于手工编写规则",
        ],
        "narration": "同学们好，这节课我们学习机器学习的基本概念。机器学习不是把规则一条条写死，而是让程序从数据中自动发现规律，并用学到的模型对新样本做出预测。请大家重点理解数据和模型之间的关系。",
        "tag": "课程导入",
    },
    {
        "title": "监督学习与无监督学习",
        "bullets": [
            "监督学习：训练数据带标签",
            "无监督学习：没有标签，寻找结构",
            "分类、回归 与 聚类、降维",
        ],
        "narration": "按照训练数据有没有标签，我们可以把机器学习分为监督学习和无监督学习。监督学习使用带标签的数据，常见任务是分类和回归；无监督学习没有标签，目标是发现数据内部的结构，例如聚类和降维。这里要注意两类方法的区别。",
        "tag": "基本分类",
    },
    {
        "title": "训练集、验证集与测试集",
        "bullets": [
            "训练集：学习模型参数",
            "验证集：选择超参数",
            "测试集：只在最后评估一次",
        ],
        "narration": "接下来看数据划分。训练集用来学习模型参数，验证集用来选择超参数，测试集只在最后评估一次。很多同学会忽略这一点，导致测试集信息泄漏。记住，测试集不能参与调参。",
        "tag": "实验方法",
    },
    {
        "title": "损失函数与梯度下降",
        "bullets": [
            "损失函数衡量预测误差",
            "梯度下降沿负梯度更新参数",
            "学习率决定每一步的步长",
        ],
        "narration": "模型学习的过程，本质上是在最小化损失函数。损失函数衡量预测值和真实值之间的差距。梯度下降沿着负梯度方向更新参数，学习率决定每一步走多远。学习率太大容易震荡，太小则收敛很慢。",
        "tag": "核心原理",
    },
    {
        "title": "过拟合与正则化",
        "bullets": [
            "过拟合：训练集好、测试集差",
            "正则化限制模型复杂度",
            "常见手段：权重衰减、Dropout、早停",
        ],
        "narration": "如果模型在训练集上表现很好，但在测试集上明显变差，就说明可能过拟合了。常见的解决办法是正则化，例如权重衰减、Dropout 和早停。核心思想是限制模型复杂度，让模型学到更通用的规律。",
        "tag": "重点难点",
    },
    {
        "title": "本课总结",
        "bullets": [
            "三要素：数据、模型、目标",
            "重点：数据划分、损失函数、过拟合",
            "作业：完成一次简单分类实验",
        ],
        "narration": "最后总结一下。机器学习可以概括为数据、模型和目标三个要素。本节课的重点是数据划分、损失函数和过拟合。课后作业是选择一个简单数据集，完成一次分类实验，并记录训练集和测试集上的准确率。下节课再见。",
        "tag": "总结与作业",
    },
]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in FONT_PATHS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def gradient(w: int, h: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (w, h), top)
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(1, h - 1)
        color = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        draw.line([(0, y), (w, y)], fill=color)
    return img


def render_slide(index: int, slide: dict, out: Path) -> None:
    w, h = 1280, 720
    img = gradient(w, h, (13, 27, 54), (20, 52, 96))
    draw = ImageDraw.Draw(img)

    # Accent bar and decorative circles.
    draw.rectangle([0, 0, 12, h], fill=(56, 189, 248))
    draw.ellipse([w - 260, -120, w + 120, 220], fill=(29, 78, 216))
    draw.ellipse([w - 120, h - 160, w + 160, h + 140], fill=(14, 165, 233))

    draw.text((64, 48), "机器学习基础 · 第 %d 讲" % (index + 1), fill=(148, 197, 255), font=font(24))
    draw.text((64, 92), slide["tag"], fill=(34, 211, 238), font=font(22))
    draw.text((64, 150), slide["title"], fill=(255, 255, 255), font=font(58))
    draw.line([(64, 236), (560, 236)], fill=(56, 189, 248), width=4)

    y = 286
    for bullet in slide["bullets"]:
        draw.ellipse([72, y + 14, 86, y + 28], fill=(56, 189, 248))
        draw.text((106, y), bullet, fill=(226, 236, 255), font=font(32))
        y += 78

    draw.text((64, h - 62), "灵影速览 · 教育视频智能摘要 DEMO", fill=(120, 160, 210), font=font(20))
    draw.text((w - 110, h - 62), f"{index + 1} / {len(SLIDES)}", fill=(120, 160, 210), font=font(20))
    img.save(out)


async def _edge_tts(text: str, out: Path, voice: str) -> bool:
    try:
        import edge_tts

        communicate = edge_tts.Communicate(text, voice, rate="-4%")
        await communicate.save(str(out))
        return out.exists() and out.stat().st_size > 1000
    except Exception as exc:
        print(f"[warn] edge-tts failed: {exc}")
        return False


def make_voice(text: str, out: Path, voice: str) -> bool:
    try:
        return asyncio.run(_edge_tts(text, out, voice))
    except Exception as exc:
        print(f"[warn] TTS unavailable: {exc}")
        return False


def split_for_srt(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；])", text)
    return [p.strip() for p in parts if p.strip()]


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    voice = os.environ.get("DEMO_VOICE", "zh-CN-XiaoxiaoNeural")
    slide_videos: list[Path] = []
    all_cues: list[Cue] = []
    offset = 0.0
    meta = []

    for i, slide in enumerate(SLIDES):
        png = BUILD_DIR / f"slide_{i + 1}.png"
        render_slide(i, slide, png)
        mp3 = BUILD_DIR / f"voice_{i + 1}.mp3"
        has_voice = make_voice(slide["narration"], mp3, voice)

        if has_voice:
            info = probe(mp3)
            duration = max(6.0, info.duration + 0.6)
            audio_args = ["-i", str(mp3)]
        else:
            duration = max(12.0, len(slide["narration"]) / 5.2)
            audio_args = ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono"]

        mp4 = BUILD_DIR / f"slide_{i + 1}.mp4"
        run_ffmpeg(
            [
                "-loop",
                "1",
                "-i",
                str(png),
                *audio_args,
                "-t",
                f"{duration:.2f}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                str(mp4),
            ]
        )
        slide_videos.append(mp4)
        meta.append(
            {
                "index": i + 1,
                "title": slide["title"],
                "start": round(offset, 2),
                "end": round(offset + duration, 2),
                "duration": round(duration, 2),
                "has_voice": has_voice,
            }
        )

        # Split narration into proportional cues.
        sentences = split_for_srt(slide["narration"])
        weights = [max(1, len(s)) for s in sentences]
        total_w = sum(weights)
        cursor = offset + 0.2
        usable = max(1.0, duration - 0.4)
        for s, wt in zip(sentences, weights):
            seg_dur = usable * wt / total_w
            all_cues.append(Cue(cursor, cursor + max(0.8, seg_dur - 0.05), s))
            cursor += seg_dur
        offset += duration

    out_video = INPUT_DIR / "demo_lecture.mp4"
    concat_clips(slide_videos, out_video)
    srt = cues_to_srt(all_cues)
    (INPUT_DIR / "demo_lecture.srt").write_text(srt, encoding="utf-8")
    (INPUT_DIR / "demo_lecture.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    info = probe(out_video)
    print(f"[ok] {out_video}  duration={info.duration:.1f}s size={out_video.stat().st_size/1e6:.1f}MB")
    print(f"[ok] {INPUT_DIR / 'demo_lecture.srt'}")


if __name__ == "__main__":
    main()
