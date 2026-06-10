from __future__ import annotations

import argparse
import csv
import random
import shutil
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

try:
    from scripts.ocr_utils import FIELDS
except ImportError:
    from ocr_utils import FIELDS


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"

DEVICE_TYPES = ["桥吊", "龙门吊", "堆场口", "混合作业牌"]
STATUSES = ["正常", "待检", "维修中", "停用"]
QUALITY_TAGS = ["blur", "tilt", "occlusion", "low_light", "noise", "compression", "small_font", "complex_bg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic engineering OCR images and manual labels.")
    parser.add_argument("--n", type=int, default=500, help="Number of images to generate.")
    parser.add_argument("--seed", type=int, default=20260609, help="Random seed.")
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def pick_quality_tags(rng: random.Random) -> list[str]:
    count = rng.choices([0, 1, 2, 3], weights=[18, 42, 30, 10], k=1)[0]
    if count == 0:
        return []
    return rng.sample(QUALITY_TAGS, k=count)


def make_values(rng: random.Random, idx: int) -> dict[str, str]:
    base_day = date(2026, 6, 1)
    return {
        "qc_id": f"QC-{rng.randint(1, 36):02d}",
        "rtg_id": f"RTG-{rng.randint(1, 28):02d}",
        "yard_gate": f"YARD-{rng.choice('ABCDEFGH')}{rng.randint(1, 9)}",
        "work_area": f"{rng.choice('ABCD')}-{rng.randint(1, 30):02d}",
        "equipment_status": rng.choice(STATUSES),
        "date": str(base_day + timedelta(days=(idx % 30))),
    }


def draw_complex_background(draw: ImageDraw.ImageDraw, width: int, height: int, rng: random.Random) -> None:
    for _ in range(26):
        color = tuple(rng.randint(120, 225) for _ in range(3))
        x1 = rng.randint(0, width)
        y1 = rng.randint(0, height)
        x2 = min(width, x1 + rng.randint(20, 130))
        y2 = min(height, y1 + rng.randint(8, 60))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=rng.randint(1, 3))
    for _ in range(16):
        color = tuple(rng.randint(130, 230) for _ in range(3))
        draw.line([(rng.randint(0, width), rng.randint(0, height)), (rng.randint(0, width), rng.randint(0, height))], fill=color, width=1)


def add_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.array(image).astype(np.int16)
    noise = np.random.default_rng(rng.randint(0, 10_000_000)).normal(0, 18, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_quality(image: Image.Image, tags: list[str], rng: random.Random) -> Image.Image:
    if "low_light" in tags:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.42, 0.68))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 0.95))
    if "noise" in tags:
        image = add_noise(image, rng)
    if "blur" in tags:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.2, 2.3)))
    if "tilt" in tags:
        image = image.rotate(rng.uniform(-7.5, 7.5), expand=True, fillcolor=(236, 238, 235))
    if "compression" in tags:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=rng.randint(24, 42))
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    return image


def generate_image(values: dict[str, str], quality_tags: list[str], out_path: Path, rng: random.Random) -> None:
    width, height = 760, 460
    bg = (232, 238, 235) if "low_light" not in quality_tags else (170, 176, 171)
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)

    if "complex_bg" in quality_tags:
        draw_complex_background(draw, width, height, rng)
    else:
        for y in range(0, height, 34):
            draw.line([(0, y), (width, y)], fill=(218, 226, 221), width=1)

    title_font = load_font(34 if "small_font" not in quality_tags else 24)
    field_font = load_font(30 if "small_font" not in quality_tags else 20)
    small_font = load_font(18)

    draw.rectangle([28, 24, width - 28, height - 24], outline=(54, 90, 86), width=3)
    draw.text((48, 36), "工程场景 OCR 采集牌", font=title_font, fill=(19, 66, 62))
    draw.text((width - 190, 44), "模拟数据", font=small_font, fill=(96, 103, 98))

    y = 104
    for field_name, label, _ in FIELDS:
        text = f"{label}：{values[field_name]}"
        draw.text((64, y), text, font=field_font, fill=(25, 31, 35))
        y += 50 if "small_font" not in quality_tags else 42

    draw.rectangle([505, 100, 690, 385], outline=(94, 126, 120), width=2)
    draw.line([535, 360, 650, 130], fill=(94, 126, 120), width=4)
    draw.line([535, 130, 650, 360], fill=(94, 126, 120), width=4)
    draw.rectangle([555, 305, 626, 352], outline=(94, 126, 120), width=3)

    if "occlusion" in quality_tags:
        for _ in range(rng.randint(1, 2)):
            x = rng.randint(45, 420)
            y = rng.randint(120, 330)
            draw.rectangle([x, y, x + rng.randint(100, 210), y + rng.randint(24, 46)], fill=(83, 91, 92))

    image = apply_quality(image, quality_tags, rng)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=88)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if IMAGE_DIR.exists():
        shutil.rmtree(IMAGE_DIR)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    label_rows: list[dict[str, str]] = []

    for idx in range(1, args.n + 1):
        image_id = f"ENG-{idx:04d}"
        device_type = rng.choice(DEVICE_TYPES)
        values = make_values(rng, idx)
        quality_tags = pick_quality_tags(rng)
        rel_path = Path("data") / "images" / f"{image_id}.jpg"
        image_path = ROOT / rel_path

        generate_image(values, quality_tags, image_path, rng)
        quality_text = "|".join(quality_tags)
        difficulty = min(100, 18 + len(quality_tags) * 22 + (12 if "occlusion" in quality_tags else 0))

        manifest_rows.append(
            {
                "image_id": image_id,
                "image_path": str(rel_path).replace("\\", "/"),
                "device_type": device_type,
                "quality_tags": quality_text,
                "difficulty_score": difficulty,
                "source": "synthetic",
            }
        )
        for field_name, label, _ in FIELDS:
            label_rows.append(
                {
                    "image_id": image_id,
                    "image_path": str(rel_path).replace("\\", "/"),
                    "device_type": device_type,
                    "field_name": field_name,
                    "field_label": label,
                    "manual_value": values[field_name],
                    "quality_tags": quality_text,
                }
            )

    for filename, rows in [("image_manifest.csv", manifest_rows), ("manual_labels.csv", label_rows)]:
        with (DATA_DIR / filename).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Generated {args.n} images and {len(label_rows)} manual field labels in {DATA_DIR}.")


if __name__ == "__main__":
    main()
