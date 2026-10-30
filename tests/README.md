# 测试

当前的可执行自测在 `scripts/` 下：

- `scripts/smoke_test.ps1`：命令行端到端冒烟测试；
- `scripts/dev/test_app.py`：用 Streamlit AppTest 跑一遍 Web 界面；
- `scripts/dev/inspect_result.py`：查看每个片段的得分与模态贡献。

后续接入真实数据后，建议在这里补充：

- 片段切分单元测试；
- 融合评分边界测试（无音轨、无字幕、极短视频）；
- 选择算法的时间覆盖率测试；
- 真实课程的回归集。
