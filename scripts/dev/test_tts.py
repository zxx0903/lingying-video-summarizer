import asyncio
import sys
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parents[2]


async def main() -> None:
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "data" / "cache" / "tts_test.mp3"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    communicate = edge_tts.Communicate(
        "同学们好，这是一段测试语音。", "zh-CN-XiaoxiaoNeural", rate="-4%"
    )
    await communicate.save(str(out))
    print(out, out.stat().st_size)


if __name__ == "__main__":
    asyncio.run(main())
