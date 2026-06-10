from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

try:
    from scripts.ocr_utils import CRITICAL_FIELDS, classify_error, build_error_reason, is_pattern_valid
except ImportError:
    from ocr_utils import CRITICAL_FIELDS, classify_error, build_error_reason, is_pattern_valid


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CORE_COLUMNS = [
    "image_id",
    "image_path",
    "device_type",
    "field_name",
    "field_label",
    "manual_value",
    "ocr_value",
    "quality_tags",
    "confidence",
    "is_correct",
    "error_type",
    "error_reason",
    "review_priority",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR results against manual labels.")
    return parser.parse_args()


def priority_score(row: pd.Series, wrong_fields_by_image: dict[str, int]) -> int:
    score = 0
    if not bool(row["is_correct"]):
        score += 35
    if row["field_name"] in CRITICAL_FIELDS and not bool(row["is_correct"]):
        score += 22
    if float(row["confidence"]) < 0.65:
        score += 18
    if wrong_fields_by_image.get(row["image_id"], 0) >= 2:
        score += 15
    if not is_pattern_valid(str(row["field_name"]), str(row["ocr_value"] or "")):
        score += 10
    if "occlusion" in str(row["quality_tags"]) or "low_light" in str(row["quality_tags"]):
        score += 8
    return min(score, 100)


def priority_label(score: int) -> str:
    if score >= 70:
        return "高"
    if score >= 35:
        return "中"
    return "低"


def main() -> None:
    parse_args()
    labels_path = DATA_DIR / "manual_labels.csv"
    ocr_path = DATA_DIR / "ocr_results.csv"
    if not labels_path.exists():
        raise FileNotFoundError("data/manual_labels.csv not found. Run scripts/generate_dataset.py first.")
    if not ocr_path.exists():
        raise FileNotFoundError("data/ocr_results.csv not found. Run scripts/run_ocr.py first.")

    labels = pd.read_csv(labels_path, encoding="utf-8-sig").fillna("")
    ocr = pd.read_csv(ocr_path, encoding="utf-8-sig").fillna("")
    merged = labels.merge(ocr, on=["image_id", "field_name", "field_label"], how="left")
    merged["ocr_value"] = merged["ocr_value"].fillna("")
    merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce").fillna(0.0)

    merged["error_type"] = merged.apply(
        lambda row: classify_error(row["manual_value"], row["ocr_value"], row["field_name"], row.get("error_type_hint", "")),
        axis=1,
    )
    merged["is_correct"] = merged["error_type"].eq("完全正确")
    merged["error_reason"] = merged.apply(
        lambda row: build_error_reason(row["error_type"], row["quality_tags"], row["field_name"], row["ocr_value"]),
        axis=1,
    )

    wrong_counts = merged.loc[~merged["is_correct"]].groupby("image_id").size().to_dict()
    merged["review_score"] = merged.apply(lambda row: priority_score(row, wrong_counts), axis=1)
    merged["review_priority"] = merged["review_score"].map(priority_label)

    output = merged[CORE_COLUMNS + ["review_score", "ocr_engine"]].copy()
    output.to_csv(DATA_DIR / "ocr_evaluation.csv", index=False, encoding="utf-8-sig")

    image_review = (
        output.groupby(["image_id", "image_path", "device_type", "quality_tags"], as_index=False)
        .agg(
            field_count=("field_name", "count"),
            wrong_field_count=("is_correct", lambda s: int((~s).sum())),
            avg_confidence=("confidence", "mean"),
            max_review_score=("review_score", "max"),
        )
    )
    image_review["image_is_correct"] = image_review["wrong_field_count"].eq(0)
    image_review["review_priority"] = image_review["max_review_score"].map(priority_label)
    image_review.to_csv(DATA_DIR / "review_queue.csv", index=False, encoding="utf-8-sig")

    print(f"Evaluation saved to data/ocr_evaluation.csv with {len(output)} field records.")
    print(f"Review queue saved to data/review_queue.csv with {len(image_review)} image records.")


if __name__ == "__main__":
    main()
