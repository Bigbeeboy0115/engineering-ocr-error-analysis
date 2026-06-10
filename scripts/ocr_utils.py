from __future__ import annotations

import re
from dataclasses import dataclass


FIELDS = [
    ("qc_id", "桥吊编号", r"QC-\d{2}"),
    ("rtg_id", "龙门吊编号", r"RTG-\d{2}"),
    ("yard_gate", "堆场口", r"YARD-[A-Z]\d"),
    ("work_area", "作业区域", r"[A-D]-\d{2}"),
    ("equipment_status", "设备状态", r"(正常|待检|维修中|停用)"),
    ("date", "日期", r"20\d{2}-\d{2}-\d{2}"),
]

FIELD_LABELS = {name: label for name, label, _ in FIELDS}
FIELD_PATTERNS = {name: pattern for name, _, pattern in FIELDS}
CRITICAL_FIELDS = {"qc_id", "rtg_id", "yard_gate", "work_area", "date"}

ERROR_TYPES = [
    "完全正确",
    "漏识别",
    "误识别",
    "数字识别错误",
    "字母识别错误",
    "中文识别错误",
    "字段缺失",
    "顺序错误",
]

QUALITY_REASON_MAP = {
    "blur": "图像模糊导致字符边缘不清晰",
    "tilt": "拍摄角度倾斜导致文本基线偏移",
    "occlusion": "遮挡覆盖了部分关键字段",
    "low_light": "低亮度降低了文字与背景对比度",
    "noise": "噪声干扰了字符轮廓",
    "compression": "压缩失真造成细节丢失",
    "small_font": "字体过小导致编号细节难以分辨",
    "complex_bg": "复杂背景干扰文本检测",
}

QUALITY_LABELS_CN = {
    "blur": "模糊",
    "tilt": "倾斜",
    "occlusion": "遮挡",
    "low_light": "低亮度",
    "noise": "噪声",
    "compression": "压缩失真",
    "small_font": "字体过小",
    "complex_bg": "背景复杂",
}


@dataclass(frozen=True)
class FieldRecord:
    image_id: str
    image_path: str
    device_type: str
    field_name: str
    field_label: str
    manual_value: str
    quality_tags: str


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().upper()
    replacements = {
        "：": ":",
        "－": "-",
        "—": "-",
        "–": "-",
        " ": "",
        "\t": "",
        "\n": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def quality_tags_to_cn(tags: object) -> str:
    raw = "" if tags is None else str(tags)
    values = [tag for tag in raw.split("|") if tag]
    return "、".join(QUALITY_LABELS_CN.get(tag, tag) for tag in values) if values else "清晰"


def is_pattern_valid(field_name: str, value: str) -> bool:
    pattern = FIELD_PATTERNS.get(field_name)
    if not pattern:
        return True
    return bool(re.fullmatch(pattern, normalize_text(value)))


def classify_error(manual_value: object, ocr_value: object, field_name: str, hint: object = "") -> str:
    hint_text = str(hint or "").strip()
    if hint_text in ERROR_TYPES:
        return hint_text

    manual = normalize_text(manual_value)
    ocr = normalize_text(ocr_value)

    if manual == ocr:
        return "完全正确"
    if not ocr:
        return "漏识别"
    if not manual and ocr:
        return "误识别"
    if sorted(manual) == sorted(ocr) and manual != ocr:
        return "顺序错误"

    if field_name in {"equipment_status"}:
        return "中文识别错误"

    digit_pairs = {("0", "O"), ("O", "0"), ("1", "I"), ("I", "1"), ("3", "8"), ("8", "3"), ("5", "S"), ("S", "5")}
    if len(manual) == len(ocr):
        diffs = [(a, b) for a, b in zip(manual, ocr) if a != b]
        if diffs and all((a, b) in digit_pairs for a, b in diffs):
            if any(a.isdigit() or b.isdigit() for a, b in diffs):
                return "数字识别错误"
            return "字母识别错误"
        if any(a.isdigit() != b.isdigit() for a, b in diffs):
            return "数字识别错误"
        if any(a.isalpha() or b.isalpha() for a, b in diffs):
            return "字母识别错误"

    if field_name in {"qc_id", "rtg_id", "yard_gate", "work_area", "date"}:
        if re.search(r"\d", manual + ocr):
            return "数字识别错误"
        return "字母识别错误"

    return "误识别"


def build_error_reason(error_type: str, quality_tags: object, field_name: str, ocr_value: object) -> str:
    if error_type == "完全正确":
        return "识别正确"

    tags = [tag for tag in str(quality_tags or "").split("|") if tag]
    reasons = [QUALITY_REASON_MAP.get(tag, tag) for tag in tags]

    if field_name in CRITICAL_FIELDS and not is_pattern_valid(field_name, str(ocr_value or "")):
        reasons.append("关键编号字段未通过格式规则校验")

    type_reason = {
        "漏识别": "文本检测阶段可能未定位到该字段",
        "误识别": "背景或噪声被误判为文字",
        "数字识别错误": "数字与相似字符混淆，例如 0/O、3/8、1/I",
        "字母识别错误": "字母与相似字符混淆，例如 O/0、I/1",
        "中文识别错误": "中文字段语义相近或笔画被干扰",
        "字段缺失": "字段标签或字段值未被完整识别",
        "顺序错误": "字符内容接近但读取顺序异常",
    }.get(error_type)
    if type_reason:
        reasons.append(type_reason)

    return "；".join(dict.fromkeys(reasons)) if reasons else "需人工复核确认"
