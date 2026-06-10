from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.generate_report import build_report
from scripts.ocr_utils import QUALITY_LABELS_CN, quality_tags_to_cn


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
REPORT_PATH = ROOT / "docs" / "analysis_report.md"


st.set_page_config(
    page_title="工程场景 OCR 评测与错误归因",
    page_icon=":mag:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d7dde2;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: #475569 !important;
        opacity: 1 !important;
        font-weight: 600;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricValue"] div,
    [data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: #0f172a !important;
        opacity: 1 !important;
    }
    div[data-testid="stDataFrame"] {border: 1px solid #d7dde2; border-radius: 8px;}
    .small-note {color: #5c6670; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    eval_path = DATA_DIR / "ocr_evaluation.csv"
    queue_path = DATA_DIR / "review_queue.csv"
    meta_path = DATA_DIR / "ocr_engine_meta.json"
    if not eval_path.exists() or not queue_path.exists():
        return pd.DataFrame(), pd.DataFrame(), {}

    df = pd.read_csv(eval_path, encoding="utf-8-sig").fillna("")
    queue = pd.read_csv(queue_path, encoding="utf-8-sig").fillna("")
    df["is_correct"] = df["is_correct"].astype(str).str.lower().isin(["true", "1", "yes"])
    queue["image_is_correct"] = queue["image_is_correct"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)
    df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce").fillna(0)
    queue["avg_confidence"] = pd.to_numeric(queue["avg_confidence"], errors="coerce").fillna(0)
    queue["max_review_score"] = pd.to_numeric(queue["max_review_score"], errors="coerce").fillna(0)

    meta = {}
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return df, queue, meta


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def explode_quality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["quality_tag"] = out["quality_tags"].replace("", "clear").str.split("|")
    out = out.explode("quality_tag")
    out["quality_label"] = out["quality_tag"].map(lambda tag: quality_tags_to_cn(tag if tag != "clear" else ""))
    return out


def status_color(priority: str) -> str:
    return {"高": "🔴 高", "中": "🟠 中", "低": "🟢 低"}.get(priority, priority)


def show_empty_state() -> None:
    st.title("工程场景 OCR 识别结果评测与错误归因分析系统")
    st.warning("还没有生成默认数据。先运行下面 4 条命令生成图片、OCR 结果、评测表和报告。")
    st.code(
        """python -m pip install -r requirements.txt
python scripts/generate_dataset.py --n 500 --seed 20260609
python scripts/run_ocr.py --engine auto
python scripts/evaluate_ocr.py
python scripts/generate_report.py
streamlit run app.py""",
        language="powershell",
    )
    st.info("当前设计支持 PaddleOCR；如果 PaddleOCR 不可用，会自动切换到模拟 OCR，网页和报告仍可完整展示。")


df, queue, meta = load_data()
if df.empty or queue.empty:
    show_empty_state()
    st.stop()

st.title("工程场景 OCR 识别结果评测与错误归因分析系统")

engine_used = meta.get("engine_used", "unknown")
fallback_reason = meta.get("fallback_reason", "")
caption = f"OCR 引擎：{engine_used}"
if fallback_reason:
    caption += f"；{fallback_reason}"
st.caption(caption)

total_images = df["image_id"].nunique()
total_fields = len(df)
field_accuracy = float(df["is_correct"].mean()) if total_fields else 0.0
image_accuracy = float(queue["image_is_correct"].mean()) if len(queue) else 0.0
avg_confidence = float(df["confidence"].mean()) if total_fields else 0.0
high_risk_images = int((queue["review_priority"] == "高").sum())

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("分析图片数", f"{total_images}")
m2.metric("字段级准确率", pct(field_accuracy))
m3.metric("图片级准确率", pct(image_accuracy))
m4.metric("平均置信度", f"{avg_confidence:.3f}")
m5.metric("高风险样本", f"{high_risk_images}")

tab_overview, tab_review, tab_detail, tab_report = st.tabs(["总体分析", "复核清单", "单图查看", "自动报告"])

with tab_overview:
    left, right = st.columns([1, 1])
    wrong = df.loc[~df["is_correct"]].copy()
    error_counts = wrong["error_type"].value_counts().reset_index()
    error_counts.columns = ["错误类型", "数量"]
    with left:
        st.subheader("错误类型分布")
        if error_counts.empty:
            st.success("当前没有错误样本。")
        else:
            fig = px.pie(error_counts, names="错误类型", values="数量", hole=0.42, color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("字段准确率")
        field_acc = df.groupby("field_label", as_index=False)["is_correct"].mean().sort_values("is_correct")
        field_acc["准确率"] = field_acc["is_correct"]
        fig = px.bar(field_acc, x="field_label", y="准确率", color="准确率", color_continuous_scale="Tealrose", range_y=[0, 1])
        fig.update_layout(xaxis_title="", yaxis_title="", coloraxis_showscale=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("设备类别错误率")
        device_rate = df.groupby("device_type", as_index=False)["is_correct"].mean()
        device_rate["错误率"] = 1 - device_rate["is_correct"]
        device_rate = device_rate.rename(columns={"device_type": "设备类别"})
        fig = px.bar(device_rate.sort_values("错误率", ascending=False), x="设备类别", y="错误率", color="设备类别", color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(xaxis_title="", yaxis_title="", showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("图片质量与识别准确率")
        qdf = explode_quality(df)
        quality_acc = qdf.groupby("quality_label", as_index=False)["is_correct"].mean().sort_values("is_correct")
        quality_acc["准确率"] = quality_acc["is_correct"]
        fig = px.bar(quality_acc, x="quality_label", y="准确率", color="准确率", color_continuous_scale="RdYlGn", range_y=[0, 1])
        fig.update_layout(xaxis_title="", yaxis_title="", coloraxis_showscale=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("主要错误原因")
    reason_counts = wrong["error_reason"].str.split("；").explode().value_counts().head(12).reset_index()
    reason_counts.columns = ["错误原因", "次数"]
    st.dataframe(reason_counts, use_container_width=True, hide_index=True)

with tab_review:
    st.subheader("待人工复核图片清单")
    review_df = queue.copy()
    review_df["质量问题"] = review_df["quality_tags"].map(quality_tags_to_cn)
    review_df["复核优先级"] = review_df["review_priority"].map(status_color)

    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
    priority_options = ["全部"] + sorted(review_df["review_priority"].unique().tolist())
    device_options = ["全部"] + sorted(review_df["device_type"].unique().tolist())
    quality_options = ["全部"] + list(QUALITY_LABELS_CN.values())
    selected_priority = f1.selectbox("复核优先级", priority_options)
    selected_device = f2.selectbox("设备类别", device_options)
    selected_quality = f3.selectbox("质量问题", quality_options)
    max_conf = f4.slider("最高平均置信度", 0.0, 1.0, 1.0, 0.01)

    filtered = review_df.copy()
    if selected_priority != "全部":
        filtered = filtered.loc[filtered["review_priority"] == selected_priority]
    if selected_device != "全部":
        filtered = filtered.loc[filtered["device_type"] == selected_device]
    if selected_quality != "全部":
        filtered = filtered.loc[filtered["质量问题"].str.contains(selected_quality, regex=False)]
    filtered = filtered.loc[filtered["avg_confidence"] <= max_conf]
    filtered = filtered.sort_values(["max_review_score", "wrong_field_count"], ascending=False)

    st.dataframe(
        filtered[
            [
                "image_id",
                "device_type",
                "质量问题",
                "wrong_field_count",
                "avg_confidence",
                "复核优先级",
                "image_path",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

with tab_detail:
    st.subheader("单张图片识别结果查看")
    ids = queue.sort_values(["max_review_score", "wrong_field_count"], ascending=False)["image_id"].tolist()
    selected_id = st.selectbox("选择图片", ids)
    image_info = queue.loc[queue["image_id"] == selected_id].iloc[0]
    image_fields = df.loc[df["image_id"] == selected_id].copy()
    image_fields["是否正确"] = image_fields["is_correct"].map({True: "正确", False: "错误"})

    left, right = st.columns([1, 1.15])
    with left:
        img_path = ROOT / str(image_info["image_path"])
        st.markdown(f"**{selected_id}**")
        st.markdown(f"<span class='small-note'>设备类别：{image_info['device_type']}；质量问题：{quality_tags_to_cn(image_info['quality_tags'])}</span>", unsafe_allow_html=True)
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.warning(f"图片文件不存在：{img_path}")
    with right:
        st.dataframe(
            image_fields[
                [
                    "field_label",
                    "manual_value",
                    "ocr_value",
                    "confidence",
                    "是否正确",
                    "error_type",
                    "error_reason",
                    "review_priority",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

with tab_report:
    st.subheader("自动生成错误分析报告")
    report_text = build_report(df, queue)
    if REPORT_PATH.exists():
        saved_report = REPORT_PATH.read_text(encoding="utf-8")
        if saved_report.strip() and "本文件由 `python scripts/generate_report.py` 自动生成" not in saved_report:
            report_text = saved_report

    st.download_button(
        "下载 Markdown 报告",
        data=report_text.encode("utf-8"),
        file_name="engineering_ocr_analysis_report.md",
        mime="text/markdown",
    )
    st.markdown(report_text)
