# 测试

当前的可执行自测在 `scripts/` 下：

- `scripts/smoke_test.ps1`：命令行端到端冒烟测试；
- `scripts/dev/test_app.py`：用 Streamlit AppTest 跑一遍 Web 界面；
- `scripts/dev/inspect_result.py`：查看每个片段的得分与模态贡献。
- `scripts/dev/evaluate_result.py`：根据人工重要片段标注计算覆盖率和召回指标。

当前已经补充：

- 目标时长容差与时间分层选择测试；
- 章节数量上限测试；
- 模板先验和用户模板保存测试；
- DOCX 导出结构测试；
- 时间覆盖率、召回率和冗余指标测试。

真实课程的正式回归集仍需要使用 EDUVSUM 官方标注或独立中文课程标注。
