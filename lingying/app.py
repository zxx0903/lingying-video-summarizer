from __future__ import annotations

import html
import json
import time
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from lingying.config import PipelineConfig, Weights
from lingying.feedback import record_selection_feedback
from lingying.pipeline import run_pipeline
from lingying.rerender import rerender_selection
from lingying.templates import (
    SUMMARY_TEMPLATES,
    load_custom_templates,
    save_custom_template,
)
from lingying.vlm import vlm_availability

ROOT = Path(__file__).resolve().parent.parent
DATA_INPUT = ROOT / "data" / "input"
OUTPUTS = ROOT / "outputs"
DEMO_VIDEO = DATA_INPUT / "demo_lecture.mp4"
DEMO_SRT = DATA_INPUT / "demo_lecture.srt"

st.set_page_config(
    page_title="灵影速览 · 教育视频智能摘要",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .hero {
        background: linear-gradient(120deg, #12224a 0%, #1d4ed8 55%, #0ea5e9 100%);
        color: #fff; border-radius: 18px; padding: 22px 28px; margin-bottom: 18px;
    }
    .hero h1 {margin: 0 0 6px 0; font-size: 30px;}
    .hero p {margin: 0; opacity: .92; font-size: 15px;}
    .card {
        background: #f7f9fd; border: 1px solid #e3e9f4; border-radius: 14px;
        padding: 14px 16px; margin-bottom: 10px;
    }
    .tag {display:inline-block; background:#e8f0ff; color:#1d4ed8; border-radius: 999px;
          padding: 2px 10px; margin-right: 6px; font-size: 12px;}
    .muted {color:#6b7280; font-size: 13px;}
    div[data-testid="stMetricValue"] {font-size: 24px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _fmt(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _zip_clips(clips: list[dict], out_zip: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for c in clips:
            p = Path(c["path"])
            if not p.is_absolute():
                p = out_dir / p
            if p.exists():
                zf.write(p, arcname=p.name)
    return out_zip


st.markdown(
    """
    <div class="hero">
      <h1>灵影速览 · 教育视频智能摘要</h1>
      <p>多模态片段评分 · 课程章节生成 · 关键帧与片段导出 · 可交互调整</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("① 选择视频")
    source = st.radio(
        "输入方式",
        ["使用演示课程", "上传视频", "填写本地路径"],
        label_visibility="collapsed",
    )
    video_path: Path | None = None
    srt_path: Path | None = None

    if source == "使用演示课程":
        if DEMO_VIDEO.exists():
            video_path = DEMO_VIDEO
            if DEMO_SRT.exists():
                srt_path = DEMO_SRT
            st.success(f"演示视频已就绪：{DEMO_VIDEO.name}")
        else:
            st.warning("演示视频尚未生成，请先运行 scripts/make_demo.py")
    elif source == "上传视频":
        up = st.file_uploader(
            "支持 mp4 / mov / mkv / avi", type=["mp4", "mov", "mkv", "avi", "webm"]
        )
        if up is not None:
            DATA_INPUT.mkdir(parents=True, exist_ok=True)
            safe_name = Path(up.name).name
            target = DATA_INPUT / safe_name
            target.write_bytes(up.getbuffer())
            video_path = target
            side = target.with_suffix(".srt")
            if side.exists():
                srt_path = side
            st.success(f"已保存：{safe_name}")
    else:
        raw = st.text_input("视频绝对路径", value=str(DEMO_VIDEO))
        if raw and Path(raw).exists():
            video_path = Path(raw)
            side = video_path.with_suffix(".srt")
            if side.exists():
                srt_path = side
            st.success("路径有效")
        elif raw:
            st.error("路径不存在")

    st.header("② 摘要设置")
    auto_ratio = st.checkbox("按比例自动决定时长（25%）", value=True)
    if auto_ratio:
        target_seconds = None
        ratio = st.slider("摘要比例", 0.10, 0.60, 0.25, 0.05)
    else:
        ratio = 0.25
        target_seconds = float(
            st.slider("目标摘要时长（分钟）", 1, 10, 3, 1) * 60
        )

    st.caption("③ 模态权重（会自动按可用性重新归一化）")
    w_text = st.slider("文本 / 语音转写", 0.0, 1.0, 0.45, 0.05)
    w_audio = st.slider("音频能量 / 语速", 0.0, 1.0, 0.30, 0.05)
    w_visual = st.slider("视觉变化 / 画面信息量", 0.0, 1.0, 0.25, 0.05)

    use_asr = st.checkbox(
        "没有字幕时使用 Whisper 转写（需安装 faster-whisper）",
        value=True,
        help="同名字幕优先。未识别到字幕时尝试语音转写；失败后自动识别幻灯片文字。",
    )
    use_ocr = st.checkbox(
        "字幕/语音都不可用时识别屏幕文字（OCR）",
        value=True,
        help="适合 PPT、板书和实验演示视频。需要 rapidocr-onnxruntime。",
    )
    vlm_probe = PipelineConfig()
    vlm_ready, vlm_detail = vlm_availability(vlm_probe)
    use_vlm = st.checkbox(
        "使用视觉语言模型理解画面（需本地 VLM）",
        value=False,
        disabled=not vlm_ready,
        help=(
            "为场景关键帧生成画面描述，适用于无字幕、无清晰文字的视频。"
            f" 当前状态：{vlm_detail}"
        ),
    )
    vlm_max_frames = st.slider(
        "VLM 场景帧数",
        16,
        160,
        48,
        8,
        disabled=not vlm_ready or not use_vlm,
        help="越多覆盖越完整，但推理时间线性增加。",
    )
    template_names = {
        item.id: item.name
        for item in (*SUMMARY_TEMPLATES, *load_custom_templates())
    }
    template_id = st.selectbox(
        "场景模板",
        options=list(template_names),
        format_func=lambda item: template_names[item],
    )
    custom_keywords_text = st.text_input(
        "自定义主题关键词",
        placeholder="例如：梯度下降, 过拟合, 实验步骤",
        help="多个关键词用逗号分隔，只作为可解释的小幅排序先验。",
    )
    with st.expander("保存为自定义模板"):
        template_name = st.text_input("模板名称", key="new_template_name")
        if st.button("保存模板", key="save_template_button"):
            try:
                parsed_keywords = [
                    item.strip()
                    for item in custom_keywords_text.replace("，", ",").split(",")
                    if item.strip()
                ]
                save_custom_template(template_name, parsed_keywords)
                st.success("模板已保存")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    max_chapters = st.slider("最大章节数", 3, 12, 8, 1)
    output_profile = st.selectbox(
        "导出规格",
        ["保持原尺寸 · MP4", "720P · MP4", "1080P · MP4", "720P · MOV", "720P · AVI"],
        index=0,
    )

    run_clicked = st.button(
        "开始生成摘要", type="primary", width="stretch", disabled=video_path is None
    )

if run_clicked and video_path is not None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUTS / f"{video_path.stem}_{stamp}"
    config = PipelineConfig(
        target_seconds=target_seconds,
        target_ratio=ratio,
        weights=Weights(w_text, w_audio, w_visual),
        asr_enabled=use_asr,
        ocr_enabled=use_ocr,
        vlm_enabled=use_vlm,
        vlm_max_frames=vlm_max_frames,
        template_id=template_id,
        custom_keywords=[
            item.strip()
            for item in custom_keywords_text.replace("，", ",").split(",")
            if item.strip()
        ],
        max_chapters=max_chapters,
        output_height=(
            720
            if output_profile.startswith(("720P", "1080P")) and "1080P" not in output_profile
            else 1080
            if output_profile.startswith("1080P")
            else None
        ),
        output_format=(
            "mov"
            if "MOV" in output_profile
            else "avi"
            if "AVI" in output_profile
            else "mp4"
        ),
    )
    bar = st.progress(0.0, text="准备中…")
    status = st.empty()

    def progress(stage: str, pct: float) -> None:
        bar.progress(min(1.0, max(0.0, pct)), text=stage)
        status.caption(f"当前步骤：{stage}")

    try:
        result = run_pipeline(
            video_path, out_dir, config=config, srt_path=srt_path, progress=progress
        )
        st.session_state["result"] = result
        st.session_state["out_dir"] = str(out_dir)
        st.session_state["chosen_ids"] = set(result["selected_ids"])
        st.session_state["adjusted_preview"] = None
        st.session_state["adjusted_summary"] = None
        st.session_state.pop("search_results", None)
        bar.empty()
        status.empty()
        st.rerun()
    except Exception as exc:
        st.error(f"处理失败：{exc}")
        st.exception(exc)

result = st.session_state.get("result")

if not result:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="card"><b>多模态评分</b><br><span class="muted">文本关键词、教学提示词、'
            "音频能量与语速、画面变化与信息密度共同决定片段重要度。</span></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="card"><b>课程结构化</b><br><span class="muted">自动生成章节、时间戳、'
            "关键词和关键帧，便于复习与检索。</span></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="card"><b>交互式调整</b><br><span class="muted">保留或删除片段、'
            "关键词检索、重新渲染与导出。</span></div>",
            unsafe_allow_html=True,
        )
    st.info("左侧选择演示课程或上传视频，然后点击「开始生成摘要」。")
    st.stop()

out_dir = Path(st.session_state["out_dir"])
chosen: set[str] = st.session_state.get("chosen_ids") or set(result["selected_ids"])
summary = st.session_state.get("adjusted_summary") or result["summary"]
content_source = {
    "srt": "同名字幕",
    "whisper": "Whisper 语音转写",
    "ocr": "屏幕文字 OCR",
}.get(result.get("asr_mode"), "未识别到文本")
if result.get("asr_mode") == "srt":
    content_source = {
        "sidecar": "外挂字幕",
        "embedded": "内嵌字幕",
        "explicit": "指定字幕",
    }.get(result.get("subtitle_source"), "字幕")
content_mode = result.get("content_mode", result.get("asr_mode", "none"))
source_labels = {
    "srt": content_source,
    "whisper": "Whisper 语音转写",
    "ocr": "屏幕文字 OCR",
    "vlm": "视觉语言模型",
    "srt+vlm": content_source + " + 视觉语言模型",
    "whisper+vlm": "Whisper 语音转写 + 视觉语言模型",
    "ocr+vlm": "屏幕文字 OCR + 视觉语言模型",
}
if content_mode == "none":
    st.error(
        "没有获得字幕、语音转写、屏幕文字或画面语义，当前只能按音频和画面变化"
        "生成片段，无法可靠概括内容。请启用 VLM、安装语音/OCR依赖，或提供同名字幕。"
    )
else:
    st.caption(
        "内容来源："
        + source_labels.get(content_mode, content_source)
    )

m = result["metrics"]
cols = st.columns(5)
cols[0].metric("原视频", _fmt(m["duration"]))
cols[1].metric("摘要时长", _fmt(m["selected_duration"]))
cols[2].metric("压缩比", f"{m['compression_ratio']:.0%}")
cols[3].metric("片段数", f"{m['selected_count']}")
cols[4].metric("处理耗时", f"{m['processing_seconds']:.1f}s")

st.markdown(
    f'<div class="card"><b>课程摘要</b><br>{html.escape(summary["overview"])}<br>'
    + " ".join(
        f'<span class="tag">{html.escape(str(k))}</span>'
        for k in summary.get("keywords", [])[:8]
    )
    + "</div>",
    unsafe_allow_html=True,
)

video_for_play = (
    st.session_state.get("adjusted_preview")
    or result.get("artifacts", {}).get("preview")
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📚 章节", "✂️ 片段与调整", "🖼️ 关键帧", "📈 时间轴", "⬇️ 导出 / 数据"]
)

with tab1:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("课程章节")
        for ch in summary.get("chapters", []):
            with st.expander(f"{ch['index']}. {ch['title']}　·　{ch['time']}", expanded=ch["index"] <= 2):
                st.write(ch["summary"])
                st.caption(
                    "关键词：" + "、".join(ch.get("keywords", [])[:6])
                    + f"　|　平均重要度 {ch.get('score', 0):.3f}"
                )
                cid = ch.get("segment_ids", [])
                clip = next(
                    (
                        s.get("clip")
                        for s in result.get("selected", [])
                        if s["id"] in cid and s.get("clip")
                    ),
                    None,
                )
                if clip:
                    st.video(str(out_dir / clip))
                evidence = []
                for seg in result.get("selected", []):
                    if seg["id"] in cid and seg.get("visual_description"):
                        evidence.append(seg["visual_description"].replace("\n", "；"))
                if evidence:
                    st.caption("视觉证据：" + "；".join(evidence[:3]))
    with right:
        st.subheader("原始视频")
        st.video(str(result["input"]["path"]))
        if video_for_play:
            st.subheader("摘要预览")
            p = Path(video_for_play)
            if not p.is_absolute():
                p = out_dir / p
            st.video(str(p))

with tab2:
    st.subheader("片段重要度与手动调整")
    st.caption("勾选保留的片段，点击「应用调整」后会重新渲染摘要视频并更新章节。")
    seg_rows = []
    for seg in result["segments"]:
        seg_rows.append(
            {
                "id": seg["id"],
                "序号": seg["index"],
                "开始": _fmt(seg["start"]),
                "结束": _fmt(seg["end"]),
                "时长": round(seg["end"] - seg["start"], 1),
                "重要度": round(seg.get("score", 0.0), 3),
                "文本": (seg.get("semantic_text") or seg.get("text", "") or "（无文本证据）")[:46],
                "画面理解": (seg.get("visual_description", "") or "")[:46],
                "关键词": "、".join(seg.get("keywords", [])[:3]),
                "保留": seg["id"] in chosen,
            }
        )
    df = pd.DataFrame(seg_rows)
    edited = st.data_editor(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "id": st.column_config.Column("ID", width="small"),
            "重要度": st.column_config.ProgressColumn(
                "重要度", min_value=0.0, max_value=1.0, format="%.3f"
            ),
        },
        disabled=["序号", "开始", "结束", "时长", "重要度", "文本", "画面理解", "关键词"],
        key="seg_editor",
    )
    keep_ids = set(edited.loc[edited["保留"], "id"].tolist())
    if st.button("应用调整并重新渲染", type="primary"):
        if not keep_ids:
            st.warning("至少保留一个片段。")
        else:
            with st.spinner("正在重新渲染选中片段…"):
                preview, new_summary, clips = rerender_selection(result, out_dir, keep_ids)
            record_selection_feedback(result, out_dir, keep_ids)
            st.session_state["chosen_ids"] = keep_ids
            st.session_state["adjusted_preview"] = str(preview)
            st.session_state["adjusted_summary"] = new_summary
            st.success(f"已按 {len(keep_ids)} 个片段重新渲染。")
            st.rerun()

    st.divider()
    st.subheader("关键词检索")
    q = st.text_input("输入关键词，例如：梯度下降、过拟合、训练集", key="kw_search")
    if q:
        hits = [
            s
            for s in result["segments"]
            if q.lower()
            in (
                s.get("semantic_text")
                or s.get("text", "")
                or ""
            ).lower()
            or any(q in k for k in s.get("keywords", []))
        ]
        if not hits:
            st.info("没有匹配片段。")
        for s in hits[:8]:
            st.markdown(
                f"**{_fmt(s['start'])} - {_fmt(s['end'])}**　重要度 {s.get('score', 0):.3f}  \n"
                f"{(s.get('text') or '（无字幕）')[:120]}"
            )
        if hits and st.button("生成主题子摘要", key="query_summary_button"):
            target_seconds = float(result["metrics"].get("target_seconds") or 120.0)
            query_ids: set[str] = set()
            query_duration = 0.0
            for seg in sorted(hits, key=lambda item: item.get("score", 0.0), reverse=True):
                duration = float(seg["end"]) - float(seg["start"])
                if query_ids and query_duration + duration > target_seconds * 1.05:
                    continue
                query_ids.add(seg["id"])
                query_duration += duration
                if query_duration >= target_seconds * 0.9:
                    break
            with st.spinner("正在按主题检索结果重新生成子摘要…"):
                preview, query_summary, _clips = rerender_selection(
                    result, out_dir, query_ids
                )
            st.session_state["chosen_ids"] = query_ids
            st.session_state["adjusted_preview"] = str(preview)
            st.session_state["adjusted_summary"] = query_summary
            st.success(f"已生成主题子摘要：{len(query_ids)} 个片段。")
            st.rerun()

with tab3:
    st.subheader("关键帧")
    sheet = result.get("artifacts", {}).get("contact_sheet")
    if sheet:
        st.image(str(out_dir / sheet), width="stretch")
    kfs = result.get("keyframes", [])
    if kfs:
        cols = st.columns(3)
        for i, kf in enumerate(kfs):
            with cols[i % 3]:
                p = out_dir / kf["path"]
                if p.exists():
                    st.image(str(p), caption=kf["label"], width="stretch")

with tab4:
    st.subheader("多模态重要度时间轴")
    tl = result.get("artifacts", {}).get("timeline")
    if tl:
        st.image(str(out_dir / tl), width="stretch")
    st.caption("柱高表示融合后的重要度，橙色为入选片段。")

with tab5:
    st.subheader("导出")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "下载结果 JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="lingying_result.json",
            mime="application/json",
            width="stretch",
        )
    with c2:
        srt_rel = result.get("artifacts", {}).get("srt")
        if srt_rel and (out_dir / srt_rel).exists():
            st.download_button(
                "下载摘要字幕 SRT",
                data=(out_dir / srt_rel).read_bytes(),
                file_name="summary.srt",
                mime="text/plain",
                width="stretch",
            )
    with c3:
        if result.get("clips"):
            zpath = out_dir / "clips.zip"
            if not zpath.exists():
                _zip_clips(result["clips"], zpath, out_dir)
            st.download_button(
                "下载片段压缩包",
                data=zpath.read_bytes(),
                file_name="lingying_clips.zip",
                mime="application/zip",
                width="stretch",
            )
    c4, c5, c6 = st.columns(3)
    with c4:
        text_rel = result.get("artifacts", {}).get("text")
        if text_rel and (out_dir / text_rel).exists():
            st.download_button(
                "下载文本摘要 TXT",
                data=(out_dir / text_rel).read_bytes(),
                file_name="summary.txt",
                mime="text/plain",
                width="stretch",
            )
    with c5:
        docx_rel = result.get("artifacts", {}).get("docx")
        if docx_rel and (out_dir / docx_rel).exists():
            st.download_button(
                "下载结构化摘要 Word",
                data=(out_dir / docx_rel).read_bytes(),
                file_name="summary.docx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                width="stretch",
            )
    with c6:
        pdf_rel = result.get("artifacts", {}).get("keyframes_pdf")
        if pdf_rel and (out_dir / pdf_rel).exists():
            st.download_button(
                "下载关键帧 PDF",
                data=(out_dir / pdf_rel).read_bytes(),
                file_name="keyframes.pdf",
                mime="application/pdf",
                width="stretch",
            )
    st.divider()
    st.subheader("运行信息")
    st.json(
        {
            "asr_mode": result.get("asr_mode"),
            "subtitle_source": result.get("subtitle_source"),
            "vlm_mode": result.get("vlm_mode"),
            "content_mode": result.get("content_mode"),
            "metrics": result.get("metrics"),
            "warnings": result.get("warnings"),
            "output_dir": str(out_dir),
        }
    )
    if result.get("warnings"):
        for w in result["warnings"]:
            st.warning(w)
