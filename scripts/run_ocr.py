from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

import pandas as pd

try:
    from scripts.ocr_utils import FIELD_LABELS, FIELDS
except ImportError:
    from ocr_utils import FIELD_LABELS, FIELDS


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaddleOCR or deterministic simulated OCR on generated images.")
    parser.add_argument("--engine", choices=["auto", "paddle", "simulate"], default="auto")
    parser.add_argument("--seed", type=int, default=20260609)
    return parser.parse_args()


def try_create_paddleocr():
    from paddleocr import PaddleOCR

    try:
        return PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    except TypeError:
        return PaddleOCR(use_angle_cls=True, lang="ch")


def flatten_paddle_result(result) -> list[tuple[str, float]]:
    lines: list[tuple[str, float]] = []
    if not result:
        return lines
    for page in result:
        if not page:
            continue
        for item in page:
            try:
                text = str(item[1][0])
                score = float(item[1][1])
            except (TypeError, IndexError, ValueError):
                continue
            lines.append((text, score))
    return lines


def parse_field_values(lines: list[tuple[str, float]]) -> dict[str, tuple[str, float]]:
    joined = "\n".join(text for text, _ in lines)
    avg_score = sum(score for _, score in lines) / len(lines) if lines else 0.0
    patterns = {
        "qc_id": r"(?:桥吊编号|桥吊|QC)[：:\s]*([QO]C[-－]?\d{1,2})",
        "rtg_id": r"(?:龙门吊编号|龙门吊|RTG)[：:\s]*([RPF]TG[-－]?\d{1,2})",
        "yard_gate": r"(?:堆场口|YARD)[：:\s]*(YARD[-－]?[A-Z]\d)",
        "work_area": r"(?:作业区域|区域)[：:\s]*([A-D][-－]?\d{1,2})",
        "equipment_status": r"(?:设备状态|状态)[：:\s]*(正常|待检|维修中|停用|止常|维修信|停川)",
        "date": r"(?:日期)?[：:\s]*(20\d{2}[-/]\d{2}[-/]\d{2})",
    }
    parsed: dict[str, tuple[str, float]] = {}
    for field_name, pattern in patterns.items():
        match = re.search(pattern, joined, flags=re.IGNORECASE)
        value = match.group(1) if match else ""
        parsed[field_name] = (value, avg_score if value else 0.0)
    return parsed


def run_paddle(labels: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ocr = try_create_paddleocr()
    rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    for image_id, group in labels.groupby("image_id", sort=True):
        image_path = ROOT / str(group.iloc[0]["image_path"])
        result = ocr.ocr(str(image_path), cls=True)
        lines = flatten_paddle_result(result)
        parsed = parse_field_values(lines)

        for line_no, (text, score) in enumerate(lines, start=1):
            raw_rows.append({"image_id": image_id, "line_no": line_no, "text": text, "confidence": round(score, 4)})

        for _, label in group.iterrows():
            field_name = label["field_name"]
            value, score = parsed.get(field_name, ("", 0.0))
            rows.append(
                {
                    "image_id": image_id,
                    "field_name": field_name,
                    "field_label": label["field_label"],
                    "ocr_value": value,
                    "confidence": round(float(score), 4),
                    "ocr_engine": "paddleocr",
                    "error_type_hint": "",
                }
            )
    return rows, raw_rows


def error_probability(field_name: str, quality_tags: str) -> float:
    tags = [tag for tag in str(quality_tags).split("|") if tag]
    probability = 0.018
    tag_weights = {
        "blur": 0.060,
        "tilt": 0.055,
        "occlusion": 0.095,
        "low_light": 0.082,
        "noise": 0.058,
        "compression": 0.050,
        "small_font": 0.088,
        "complex_bg": 0.064,
    }
    field_weights = {
        "qc_id": 0.024,
        "rtg_id": 0.027,
        "yard_gate": 0.030,
        "work_area": 0.022,
        "equipment_status": 0.014,
        "date": 0.025,
    }
    probability += sum(tag_weights.get(tag, 0.0) for tag in tags)
    probability += field_weights.get(field_name, 0.0)
    return min(probability, 0.86)


def replace_one(value: str, pairs: list[tuple[str, str]], rng: random.Random) -> str:
    candidates = [(src, dst) for src, dst in pairs if src in value]
    if not candidates:
        return value
    src, dst = rng.choice(candidates)
    return value.replace(src, dst, 1)


def mutate_value(value: str, field_name: str, quality_tags: str, rng: random.Random) -> tuple[str, str, float]:
    tags = [tag for tag in str(quality_tags).split("|") if tag]
    choices = ["数字识别错误", "字母识别错误", "漏识别", "字段缺失", "顺序错误"]
    if field_name == "equipment_status":
        choices = ["中文识别错误", "漏识别", "字段缺失"]
    if "occlusion" in tags:
        choices.extend(["漏识别", "字段缺失"])
    if "complex_bg" in tags or "noise" in tags:
        choices.append("误识别")

    error_type = rng.choice(choices)
    if error_type in {"漏识别", "字段缺失"}:
        return "", error_type, rng.uniform(0.08, 0.42)
    if error_type == "误识别":
        return rng.choice(["NOISE-88", "QC-B3", "YARD-47", "RTG-O2"]), error_type, rng.uniform(0.32, 0.62)
    if error_type == "数字识别错误":
        mutated = replace_one(value, [("0", "O"), ("O", "0"), ("1", "I"), ("3", "8"), ("8", "3"), ("5", "S")], rng)
        if mutated == value:
            mutated = value[:-1] + rng.choice("0386")
        return mutated, error_type, rng.uniform(0.46, 0.78)
    if error_type == "字母识别错误":
        mutated = replace_one(value, [("O", "0"), ("I", "1"), ("A", "4"), ("B", "8"), ("G", "6"), ("Q", "O")], rng)
        if mutated == value:
            mutated = "0" + value[1:]
        return mutated, error_type, rng.uniform(0.48, 0.80)
    if error_type == "中文识别错误":
        mapping = {"正常": "止常", "待检": "侍检", "维修中": "维修信", "停用": "停川"}
        return mapping.get(value, "未知"), error_type, rng.uniform(0.45, 0.76)
    if error_type == "顺序错误":
        if field_name == "date":
            parts = value.split("-")
            return "-".join([parts[1], parts[2], parts[0]]), error_type, rng.uniform(0.50, 0.82)
        if "-" in value:
            left, right = value.split("-", 1)
            return f"{right}-{left}", error_type, rng.uniform(0.50, 0.82)
    return value, "完全正确", rng.uniform(0.88, 0.99)


def run_simulation(labels: pd.DataFrame, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []

    for _, label in labels.iterrows():
        field_name = str(label["field_name"])
        manual_value = str(label["manual_value"])
        quality_tags = str(label["quality_tags"] or "")
        prob = error_probability(field_name, quality_tags)
        if rng.random() < prob:
            ocr_value, hint, confidence = mutate_value(manual_value, field_name, quality_tags, rng)
        else:
            ocr_value, hint, confidence = manual_value, "完全正确", rng.uniform(0.86, 0.99)

        rows.append(
            {
                "image_id": label["image_id"],
                "field_name": field_name,
                "field_label": FIELD_LABELS[field_name],
                "ocr_value": ocr_value,
                "confidence": round(confidence, 4),
                "ocr_engine": "simulated",
                "error_type_hint": hint,
            }
        )

    for image_id, group in labels.groupby("image_id", sort=True):
        text = " | ".join(f"{row.field_label}:{row.manual_value}" for row in group.itertuples())
        raw_rows.append({"image_id": image_id, "line_no": 1, "text": text, "confidence": 0.92})
    return rows, raw_rows


def main() -> None:
    args = parse_args()
    manual_path = DATA_DIR / "manual_labels.csv"
    if not manual_path.exists():
        raise FileNotFoundError("data/manual_labels.csv not found. Run scripts/generate_dataset.py first.")

    labels = pd.read_csv(manual_path, encoding="utf-8-sig").fillna("")
    engine_used = args.engine
    fallback_reason = ""

    if args.engine in {"auto", "paddle"}:
        try:
            result_rows, raw_rows = run_paddle(labels)
            engine_used = "paddleocr"
        except Exception as exc:
            if args.engine == "paddle":
                raise
            fallback_reason = f"PaddleOCR unavailable, switched to simulated OCR: {exc}"
            result_rows, raw_rows = run_simulation(labels, args.seed)
            engine_used = "simulated"
    else:
        result_rows, raw_rows = run_simulation(labels, args.seed)
        engine_used = "simulated"

    with (DATA_DIR / "ocr_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)

    with (DATA_DIR / "ocr_raw_lines.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    meta = {"engine_requested": args.engine, "engine_used": engine_used, "fallback_reason": fallback_reason}
    (DATA_DIR / "ocr_engine_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OCR finished with engine={engine_used}. Results saved to data/ocr_results.csv.")
    if fallback_reason:
        print(fallback_reason)


if __name__ == "__main__":
    main()
