# 第三方项目与许可证

本目录保留三个开源参考项目的原始代码与 LICENSE，便于溯源和答辩时说明。
`lingying/` 下的系统代码为本项目新写，没有直接复制这三个项目的源码。

| 目录 | 项目 | 许可证 | 在本项目中的用途 |
|---|---|---|---|
| `openclip-main/` | https://github.com/linzzzzzz/openclip | MIT | 工程流水线、中文 ASR 路由、片段编辑交互的参考 |
| `EDUVSUM-master/` | https://github.com/VideoAnalysis/EDUVSUM | MIT | 教育视频的音视频+文本特征与重要片段识别思路 |
| `MMSum_model-main/` | https://github.com/Jielin-Qiu/MMSum_model | Apache-2.0 | 多模态摘要任务定义、数据集与评测参考 |

## 使用边界

- MIT / Apache-2.0 允许修改与商用，但需要保留版权与许可证声明。
- 参考项目的数据集、预训练权重可能有独立条款，需单独确认。
- 本项目当前没有把上述项目的模型权重打包进 Demo。

## 其他依赖

- FFmpeg：通过 `imageio-ffmpeg` 提供的二进制调用，许可证以其发布包为准。
- Streamlit：Apache-2.0。
- OpenCV：Apache-2.0。
- jieba：MIT。
- edge-tts：仅用于生成演示课程的语音，运行时不需要联网。
