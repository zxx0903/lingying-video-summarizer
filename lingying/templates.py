from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SummaryTemplate:
    id: str
    name: str
    scene: str
    keywords: tuple[str, ...]


# Keyword templates are deliberately transparent. A template only adds a small
# ranking prior; it never invents content or bypasses the evidence in the video.
SUMMARY_TEMPLATES: tuple[SummaryTemplate, ...] = (
    SummaryTemplate(
        "education_general",
        "教育 · 通用课程",
        "education",
        ("定义", "原理", "概念", "重点", "总结", "注意", "方法", "结论"),
    ),
    SummaryTemplate(
        "education_knowledge",
        "教育 · 知识点",
        "education",
        ("定义", "概念", "原理", "公式", "本质", "核心"),
    ),
    SummaryTemplate(
        "education_steps",
        "教育 · 实验步骤",
        "education",
        ("步骤", "操作", "实验", "接线", "方法", "流程", "注意事项"),
    ),
    SummaryTemplate(
        "education_exam",
        "教育 · 考点复习",
        "education",
        ("重点", "难点", "考点", "考试", "例题", "记住", "复习"),
    ),
    SummaryTemplate(
        "education_homework",
        "教育 · 作业要求",
        "education",
        ("作业", "要求", "提交", "截止", "报告", "练习"),
    ),
    SummaryTemplate(
        "education_math",
        "教育 · 数理推导",
        "education",
        ("定理", "公式", "证明", "推导", "计算", "求解", "结论"),
    ),
    SummaryTemplate(
        "education_programming",
        "教育 · 编程实现",
        "education",
        ("代码", "函数", "变量", "算法", "输入", "输出", "运行", "调试"),
    ),
    SummaryTemplate(
        "security_intrusion",
        "安防 · 异常闯入",
        "security",
        ("闯入", "翻越", "进入", "入侵", "异常", "报警", "边界"),
    ),
    SummaryTemplate(
        "security_fire",
        "安防 · 烟火告警",
        "security",
        ("火灾", "烟雾", "明火", "燃烧", "消防", "报警"),
    ),
    SummaryTemplate(
        "security_fight",
        "安防 · 人员冲突",
        "security",
        ("冲突", "打斗", "追赶", "倒地", "聚众", "危险"),
    ),
    SummaryTemplate(
        "security_equipment",
        "安防 · 设备故障",
        "security",
        ("故障", "停机", "泄漏", "冒烟", "损坏", "异常"),
    ),
    SummaryTemplate(
        "security_vehicle",
        "安防 · 车辆事件",
        "security",
        ("车辆", "碰撞", "逆行", "停车", "车牌", "事故"),
    ),
    SummaryTemplate(
        "security_access",
        "安防 · 人员出入",
        "security",
        ("出入", "门禁", "识别", "陌生人", "尾随", "授权"),
    ),
    SummaryTemplate(
        "security_patrol",
        "安防 · 巡检记录",
        "security",
        ("巡检", "隐患", "整改", "检查", "设备", "记录"),
    ),
    SummaryTemplate(
        "social_goal",
        "社交媒体 · 赛事进球",
        "social",
        ("进球", "射门", "助攻", "得分", "绝杀", "goal"),
    ),
    SummaryTemplate(
        "social_funny",
        "社交媒体 · 内容笑点",
        "social",
        ("好笑", "笑点", "搞笑", "梗", "反转", "哈哈"),
    ),
    SummaryTemplate(
        "social_opinion",
        "社交媒体 · 核心观点",
        "social",
        ("认为", "观点", "结论", "关键是", "我认为", "意味着"),
    ),
    SummaryTemplate(
        "social_highlight",
        "社交媒体 · 高光时刻",
        "social",
        ("精彩", "高光", "第一次", "突破", "记录", "获胜"),
    ),
    SummaryTemplate(
        "social_product",
        "社交媒体 · 产品亮点",
        "social",
        ("产品", "功能", "优势", "特点", "体验", "解决"),
    ),
    SummaryTemplate(
        "meeting_decisions",
        "会议 · 决策结论",
        "meeting",
        ("决定", "结论", "同意", "方案", "结论是", "下一步"),
    ),
    SummaryTemplate(
        "meeting_tasks",
        "会议 · 行动项",
        "meeting",
        ("负责", "完成", "截止", "行动项", "跟进", "安排"),
    ),
    SummaryTemplate(
        "meeting_risks",
        "会议 · 风险问题",
        "meeting",
        ("风险", "问题", "阻塞", "依赖", "延期", "影响"),
    ),
)

TEMPLATES_BY_ID = {item.id: item for item in SUMMARY_TEMPLATES}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_TEMPLATES_PATH = PROJECT_ROOT / "data" / "templates" / "user_templates.json"


def get_template(template_id: str | None) -> SummaryTemplate:
    return TEMPLATES_BY_ID.get(template_id or "", SUMMARY_TEMPLATES[0])


def load_custom_templates(path: str | Path | None = None) -> list[SummaryTemplate]:
    path = Path(path) if path else CUSTOM_TEMPLATES_PATH
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    result = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        template_id = str(item.get("id", "")).strip()
        keywords = item.get("keywords")
        if not template_id or not isinstance(keywords, list):
            continue
        result.append(
            SummaryTemplate(
                template_id,
                str(item.get("name") or template_id),
                str(item.get("scene") or "custom"),
                tuple(str(word) for word in keywords if str(word).strip()),
            )
        )
    return result


def save_custom_template(
    name: str,
    keywords: list[str],
    scene: str = "custom",
    path: str | Path | None = None,
) -> SummaryTemplate:
    name = name.strip()
    cleaned = []
    for word in keywords:
        word = str(word).strip()
        if word and word not in cleaned:
            cleaned.append(word)
    if not name or not cleaned:
        raise ValueError("模板名称和关键词不能为空")
    path = Path(path) if path else CUSTOM_TEMPLATES_PATH
    existing = load_custom_templates(path)
    slug = re.sub(r"[^0-9A-Za-z_-]+", "_", name).strip("_")
    if not slug:
        slug = f"template_{int(time.time())}"
    template_id = f"user_{slug}"
    template = SummaryTemplate(template_id, name, scene, tuple(cleaned))
    existing = [item for item in existing if item.id != template_id]
    existing.append(template)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "id": item.id,
                    "name": item.name,
                    "scene": item.scene,
                    "keywords": list(item.keywords),
                }
                for item in existing
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return template


def resolve_template_keywords(
    template_id: str | None,
    custom_keywords: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    template = TEMPLATES_BY_ID.get(template_id or "")
    if template is None:
        template = next(
            (item for item in load_custom_templates() if item.id == template_id),
            SUMMARY_TEMPLATES[0],
        )
    result: list[str] = []
    for word in (*template.keywords, *(custom_keywords or [])):
        word = str(word).strip()
        if word and word.lower() not in {item.lower() for item in result}:
            result.append(word)
    return result
