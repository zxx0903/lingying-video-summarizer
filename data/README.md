# 输入数据说明

## 视频

把视频放在 `data/input/` 下，支持 `mp4 / mov / mkv / avi / webm`。

## 字幕（可选）

与视频同名、同目录的 `.srt` 会被自动读取，例如：

```text
data/input/lecture_01.mp4
data/input/lecture_01.srt
```

有字幕时不会调用 Whisper，处理更快、更稳定。演示视频自带
`demo_lecture.srt`。

## 课程大纲（可选，但强烈推荐）

与视频同名的 `.json` 或 `.outline.json`，用于提供章节标题与时间范围：

```json
[
  {"title": "课程导入", "start": 0, "end": 120},
  {"title": "监督学习", "start": 120, "end": 360}
]
```

系统会优先按大纲划分章节，并在每个章节内选代表片段；没有大纲时则使用
画面场景检测 + 启发式标题。

## 输出

每次运行会新建 `outputs/<视频名>_<时间戳>/`，包含：

```text
result.json        完整结果（片段、得分、章节、指标）
preview.mp4        摘要预览（按选中片段拼接）
clips/             每个片段单独 MP4
keyframes/         关键帧 JPG
keyframes.jpg      关键帧拼图
timeline.png       多模态重要度时间轴
summary.srt        与摘要视频对齐的字幕
audio/             抽取的音轨
frames/            1 FPS 抽帧结果
```
