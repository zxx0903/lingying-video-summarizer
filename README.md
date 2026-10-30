# 灵影速览 · 教育视频智能摘要（大创 Demo）

面向中文课程视频的**多模态摘要系统原型**：输入一段课程录像，自动完成
「语音/字幕 → 片段切分 → 文本 + 音频 + 视觉评分 → 章节与摘要 → 片段/关键帧/字幕导出」，
并提供可交互的 Web 界面。

当前版本的目标是**能演示、能讲清楚、能继续迭代**，不是最终产品。

---

## 1. 现在能做什么

- 输入：本地视频、上传视频、内置演示课程（自动带字幕）。
- 自动切分课程片段，生成**章节标题、时间戳、章节摘要**。
- 三模态打分：
  - 文本：关键词 TF-IDF、教学提示词（重点/注意/总结/作业等）、文本信息量；
- 无字幕时可识别 PPT/实验指导书等屏幕文字，并将 OCR 文本用于切片与摘要；
  - 音频：能量、有效语音比例、能量波动、语速；
  - 视觉：边缘密度、画面变化、色彩丰富度。
- **动态门控融合**：某个模态缺失（无音轨/无字幕）时自动降权。
- **场景/大纲对齐**：优先保证每个章节/幻灯片都有代表片段，避免摘要全挤在视频前几分钟。
- 输出：摘要预览 MP4、分段 MP4、关键帧拼图、时间轴 PNG、摘要 SRT、`result.json`。
- Web 交互：保留/删除片段后重新渲染、关键词检索、结果下载。
- 纯 CPU 可跑，不依赖 GPU 或云端 API；可选接入 Whisper / 大模型。

## 2. 快速开始

### 方式 A：VS Code（推荐）

1. 用 VS Code 打开本项目文件夹。
2. 首次使用先运行任务 **`0. Setup Python environment`**（需要联网安装依赖）。
3. 运行任务 **`1. Generate demo lecture video`**（生成演示课程，约 20 秒）。
4. 运行任务 **`2. Run Streamlit demo`**，浏览器打开 `http://127.0.0.1:8501`。
5. 左侧保持「使用演示课程」，点击「开始生成摘要」。

`.vscode/settings.json` 默认使用项目内的 `.venv`；如果本机已有便携运行环境，
启动脚本也会优先复用它。需要 Python 3.10 或 3.11。

### 方式 B：PowerShell

```powershell
# 首次安装核心依赖（需要联网）
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1

# 需要 Whisper 语音转写和 OCR 时
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Optional

# 生成演示课程（只需一次）
powershell -ExecutionPolicy Bypass -File .\scripts\make_demo.ps1

# 启动 Web Demo
powershell -ExecutionPolicy Bypass -File .\scripts\run_demo.ps1

# 或直接跑命令行
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m lingying.cli `
  --input .\data\input\demo_lecture.mp4 `
  --out .\outputs\cli_demo
```

桌面版默认把 Whisper 模型放在 `runtime/whisper_model_path.txt` 指向的本地目录。
也可以设置环境变量 `LINGYING_WHISPER_MODEL` 指向模型目录；未配置时，
`faster-whisper` 会按模型名称自行下载。语音转写会自动检测视频语言。

### 方式 C：VS Code 调试

- `F5` → 选择 **Streamlit Demo** 或 **CLI Demo**。
- 任务面板 → **3. Smoke test (CLI)**，约 5 秒跑完一遍完整流水线。

## 3. 演示数据与实测指标

内置演示课程 `data/input/demo_lecture.mp4`：

- 6 页幻灯片、中文讲解、约 **2 分 08 秒**；
- 配套字幕 `demo_lecture.srt` 与章节元数据 `demo_lecture.json`；
- 已在 `outputs/demo_final/` 生成一套完整结果。

当前实测（CPU，1 FPS 抽帧）：

| 指标 | 数值 |
|---|---|
| 原视频 | 127.6 s |
| 摘要时长 | 52.2 s（约 41%） |
| 章节数 | 6（每页幻灯片 1 个） |
| 处理耗时 | 约 4.6 s |
| 实时倍率 | 约 28× 实时 |
| 依赖 | 纯 CPU，无 GPU / 无云 API |

> 这些数字只代表当前 Demo 在 2 分钟合成课程上的表现，不能直接外推到 1 小时真实课程；
> 长视频的耗时、覆盖率和画质需要重新评测。

## 4. 目录结构

```text
项目根目录
├─ README.md                     ← 本文件
├─ .vscode\                      ← VS Code 任务与调试配置
├─ lingying\                     ← 我们自己的系统代码
│  ├─ app.py                     Streamlit Web 界面
│  ├─ cli.py                     命令行入口
│  ├─ pipeline.py                端到端流水线
│  ├─ ingest/features/fusion...  各功能模块
│  └─ ...
├─ scripts\                      ← 一键脚本与开发工具
│  ├─ make_demo.py               生成演示课程（幻灯片 + 中文 TTS + SRT）
│  ├─ make_demo.ps1
│  ├─ run_demo.ps1
│  ├─ smoke_test.ps1
│  └─ dev\
│     ├─ inspect_result.py       查看片段得分表
│     ├─ test_app.py             Streamlit 端到端自测
│     └─ test_tts.py             Edge-TTS 连通性测试
├─ data\
│  ├─ input\                     输入视频 / 字幕 / 大纲
│  └─ cache\                     中间缓存（抽帧、TTS、构建产物）
├─ outputs\                      每次运行的结果目录
│  └─ demo_final\                已生成的一套完整 Demo 结果
├─ third_party\                  ← 开源参考项目（保留原始 LICENSE）
│  ├─ openclip-main\             MIT
│  ├─ EDUVSUM-master\            MIT
│  ├─ MMSum_model-main\          Apache-2.0
│  └─ _archives\                 源码 zip 与重复克隆备份
├─ docs\                         选型、设计、答辩、计划
└─ runtime\                      便携 Python 3.11 + 已安装依赖
```

## 5. 处理流程

```text
视频
 ├─ FFmpeg 读取元信息 / 抽音轨
 ├─ FFmpeg 1 FPS 抽帧（长视频自动降采样）
 ├─ 字幕 SRT，或 faster-whisper 转写
 ├─ 片段切分（按字幕停顿 + 最大时长）
 ├─ 场景检测 / 课程大纲对齐
 ├─ 多模态特征
 │   ├─ 文本：TF-IDF 关键词、教学提示词、信息量、语速
 │   ├─ 音频：RMS 能量、有效语音比例、能量波动
 │   └─ 视觉：边缘密度、帧间变化、色彩丰富度
 ├─ 动态门控融合评分
 ├─ 场景覆盖选择 + 相似度去重 + 目标时长裁剪
 ├─ 章节 / 摘要生成（模板，可选 LLM 润色）
 └─ FFmpeg 渲染片段、关键帧、时间轴、SRT、JSON
```

更详细的公式和取舍见 [docs/02_系统设计.md](docs/02_系统设计.md)。

## 6. 与开源项目的关系

本 Demo **不是**把某个仓库改名后直接运行：

- `third_party/openclip-main`（MIT）：工程流水线与中文 ASR/剪辑思路参考；
- `third_party/EDUVSUM-master`（MIT）：教育视频多模态重要片段识别的算法参考；
- `third_party/MMSum_model-main`（Apache-2.0）：多模态摘要任务定义与评测参考；
- **核心实现**（`lingying/`）是为本项目新写的：动态门控融合、场景覆盖选择、
  大纲对齐、章节生成、时间轴/关键帧渲染、交互式 Web 界面、演示数据生成。

各项目的许可证与使用边界见 [third_party/NOTICE.md](third_party/NOTICE.md)。

## 7. 当前限制

- 这是**原型**：没有用户系统、任务队列、数据库和并发控制。
- 视觉特征目前是轻量手工特征（边缘/变化/色彩），还不是 CLIP 语义向量；
  但接口已经按“视觉模态分数”预留，后续可直接替换。
- 默认优先使用同名字幕；无字幕时尝试 Whisper 语音转写，失败后使用
  `rapidocr-onnxruntime` 识别 PPT/屏幕文字。
- OCR 只读取画面文字，不能理解纯人物、动作或复杂的视觉语义；这类内容仍需
  接入视觉语言模型。
- 章节标题优先使用同目录的 `*.json` 大纲；没有大纲时使用启发式标题，质量会下降。
- 片段按目标时长做统一裁剪，可能截断句子；后续应对齐字幕 cue 或做语义完整性约束。
- 只在合成课程上验证过，尚未在真实课堂、慕课、会议录像上做系统评测。

## 8. 推荐阅读顺序

1. [docs/01_开源选型与改造说明.md](docs/01_开源选型与改造说明.md)
2. [docs/02_系统设计.md](docs/02_系统设计.md)
3. [docs/03_答辩与演示脚本.md](docs/03_答辩与演示脚本.md)
4. [docs/04_后续开发计划.md](docs/04_后续开发计划.md)
5. [docs/05_项目一页纸.md](docs/05_项目一页纸.md)
6. [docs/06_相关开源项目清单.md](docs/06_相关开源项目清单.md)
