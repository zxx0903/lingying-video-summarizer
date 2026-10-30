#!/usr/bin/env python3
"""
Extract a reference audio clip from a video for speaker name mapping.

Usage:
    uv run python tools/extract_reference.py VIDEO START END OUTPUT

Arguments:
    VIDEO   Source video or audio file
    START   Start time — HH:MM:SS or plain seconds (e.g. 83)
    END     End time   — HH:MM:SS or plain seconds (e.g. 105)
    OUTPUT  Output WAV file (e.g. references/Host.wav)
            The filename stem becomes the speaker name.

Examples:
    uv run python tools/extract_reference.py livestream.mp4 00:01:23 00:01:45 references/Host.wav
    uv run python tools/extract_reference.py livestream.mp4 83 105 references/Guest_Alice.wav

Tips for a good reference clip:
    - 10–30 seconds of clean, single-speaker speech
    - Avoid moments with overlapping voices, background music, or crowd noise
    - Natural conversational speech works better than reading/scripted lines
    - Pick a stretch where only one person is talking throughout
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Extract a reference audio clip from a video for speaker name mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video",  help="Source video or audio file")
    parser.add_argument("start",  help="Start time (HH:MM:SS or seconds)")
    parser.add_argument("end",    help="End time   (HH:MM:SS or seconds)")
    parser.add_argument("output", help="Output WAV file (e.g. references/Host.wav)")
    args = parser.parse_args()

    video_path  = Path(args.video)
    output_path = Path(args.output)

    if not video_path.exists():
        print(f"❌ File not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if output_path.suffix.lower() != ".wav":
        print("⚠️  Output should be a .wav file for best compatibility", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer the bundled ffmpeg shipped with imageio-ffmpeg so the user
    # doesn't need a separate system installation.
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = "ffmpeg"

    cmd = [
        ffmpeg,
        "-y",                    # overwrite without prompting
        "-i", str(video_path),
        "-ss", args.start,
        "-to", args.end,
        "-vn",                   # strip video
        "-acodec", "pcm_s16le",  # uncompressed PCM — best for embedding models
        "-ar", "16000",          # 16 kHz — whisperx's native sample rate
        "-ac", "1",              # mono
        str(output_path),
    ]

    print(f"🎬 Source : {video_path}")
    print(f"⏱️  Range  : {args.start} → {args.end}")
    print(f"💾 Output : {output_path}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ ffmpeg failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    size_kb = output_path.stat().st_size // 1024
    speaker_name = output_path.stem
    references_dir = output_path.parent

    print(f"✅ Saved {size_kb} KB → {output_path}")
    print(f"\n💡 Speaker name will be: \"{speaker_name}\"")
    print(f"   Use with: --speaker-references {references_dir}/")


if __name__ == "__main__":
    main()