from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from scripts.ocr_utils import quality_tags_to_cn
except ImportError:
    from ocr_utils import quality_tags_to_cn


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Markdown OCR analysis report.")
    return parser.parse_args()


def top_lines(series: pd.Series, limit: int = 5) -> list[str]:
    total = max(int(series.sum()), 1)
    lines = []
    for name, count in series.head(limit).items():
        lines.append(f"- {name}：{int(count)} 次，占 {count / total * 100:.1f}%")
    return lines


def build_report(df: pd.DataFrame, queue: pd.DataFrame | None = None) -> str:
    total_images = df["image_id"].nunique()
    total_fields = len(df)
    field_accuracy = float(df["is_correct"].mean()) if total_fields else 0.0
    image_accuracy = float(df.groupby("image_id")["is_correct"].all().mean()) if total_images else 0.0
    avg_confidence = float(df["confidence"].mean()) if total_fields else 0.0

    wrong = df.loc[~df["is_correct"]].copy()
    error_type_counts = wrong["error_type"].value_counts()
    field_accuracy_table = df.groupby("field_label")["is_correct"].mean().sort_values()
    device_error_table = (1 - df.groupby("device_type")["is_correct"].mean()).sort_values(ascending=False)

    exploded = df.assign(quality_tag=df["quality_tags"].replace("", "clear").str.split("|")).explode("quality_tag")
    quality_accuracy = exploded.groupby("quality_tag")["is_correct"].mean().sort_values()
    quality_error = (1 - quality_accuracy).sort_values(ascending=False)
    quality_line = "、".join(f"{quality_tags_to_cn(tag)} 错误率 {pct(rate)}" for tag, rate in quality_error.head(4).items())

    high_risk = 0
    if queue is not None and not queue.empty:
        high_risk = int((queue["review_priority"] == "高").sum())
    else:
        high_risk = int((df["review_priority"] == "高").sum())

    lines = [
        "# 工程场景 OCR 识别结果评测与错误归因分析报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. 核心结论",
        "",
        f"本次共分析 {total_images} 张工程场景图片，覆盖 {total_fields} 个字段级识别结果。",
        f"OCR 整体字段级准确率为 {pct(field_accuracy)}，图片级完全正确率为 {pct(image_accuracy)}，平均置信度为 {avg_confidence:.3f}。",
        f"待人工重点复核样本共 {high_risk} 条，主要集中在低亮度、遮挡、模糊和字体过小等质量问题图片。",
        "",
        "## 2. 错误类型分布",
        "",
    ]
    lines.extend(top_lines(error_type_counts))
    lines.extend(
        [
            "",
            "## 3. 字段准确率",
            "",
        ]
    )
    for field, acc in field_accuracy_table.items():
        lines.append(f"- {field}：{pct(float(acc))}")

    lines.extend(["", "## 4. 设备类别错误率", ""])
    for device, rate in device_error_table.items():
        lines.append(f"- {device}：{pct(float(rate))}")

    lines.extend(["", "## 5. 图片质量影响", ""])
    lines.append(f"不同质量标签下的主要风险为：{quality_line}。")
    lines.append("这说明 OCR 错误并非均匀分布，而是明显受拍摄质量、文本清晰度和背景干扰影响。")

    lines.extend(["", "## 6. 复核与优化建议", ""])
    lines.extend(
        [
            "- 对桥吊编号、龙门吊编号、堆场口、作业区域和日期字段增加格式规则校验。",
            "- 对低亮度、遮挡、模糊、字体过小样本优先进入人工复核队列。",
            "- 采集侧建议控制拍摄角度，补充照明，避免设备结构遮挡编号区域。",
            "- 模型侧建议补充相似字符样本，例如 0/O、1/I、3/8，以及中文状态字段的混淆样本。",
            "- 业务侧可把高风险字段作为安全资料整理的检查项，降低编号录入错误带来的后续追溯成本。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parse_args()
    eval_path = DATA_DIR / "ocr_evaluation.csv"
    if not eval_path.exists():
        raise FileNotFoundError("data/ocr_evaluation.csv not found. Run scripts/evaluate_ocr.py first.")
    df = pd.read_csv(eval_path, encoding="utf-8-sig").fillna("")
    queue_path = DATA_DIR / "review_queue.csv"
    queue = pd.read_csv(queue_path, encoding="utf-8-sig").fillna("") if queue_path.exists() else None
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(df, queue)
    (DOCS_DIR / "analysis_report.md").write_text(report, encoding="utf-8")
    print("Report saved to docs/analysis_report.md.")


if __name__ == "__main__":
    main()
