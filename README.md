# Engineering OCR Error Analysis

工程场景 OCR 识别结果评测与错误归因分析系统。项目模拟桥吊、龙门吊、堆场口等工程现场图片识别流程，生成带人工标注的图片数据集，执行 OCR 或可复现的模拟 OCR，计算字段级准确率，识别错误类型，并生成可交互 Streamlit 看板和 Markdown 分析报告。

## Project Scope

- Generate 500 synthetic engineering-scene images with manual labels.
- Add controlled image-quality perturbations: blur, tilt, occlusion, low brightness, noise, compression artifacts, small font, and complex background.
- Run OCR with `--engine auto`: PaddleOCR is used when available; otherwise the pipeline falls back to a deterministic simulated OCR engine.
- Evaluate OCR output at field level and image level.
- Attribute errors to quality tags, field formats, device categories, and confidence levels.
- Produce a review queue for high-risk samples.

## Repository Structure

```text
app.py
data/
  images/
  image_manifest.csv
  manual_labels.csv
  ocr_results.csv
  ocr_evaluation.csv
  review_queue.csv
docs/
  analysis_report.md
scripts/
  generate_dataset.py
  run_ocr.py
  evaluate_ocr.py
  generate_report.py
  ocr_utils.py
requirements.txt
start_app.bat
```

## Data Fields

The generated images include six core fields:

- `桥吊编号`, for example `QC-03`
- `龙门吊编号`, for example `RTG-12`
- `堆场口`, for example `YARD-A7`
- `作业区域`, for example `B-15`
- `设备状态`, for example `正常`
- `日期`, for example `2026-06-09`

The evaluation table includes:

`image_id`, `image_path`, `device_type`, `field_name`, `manual_value`, `ocr_value`, `quality_tags`, `confidence`, `is_correct`, `error_type`, `error_reason`, `review_priority`

## Error Taxonomy

- 漏识别
- 误识别
- 数字识别错误
- 字母识别错误
- 中文识别错误
- 字段缺失
- 顺序错误
- 完全正确

## Run Locally

```powershell
python -m pip install -r requirements.txt
python scripts/generate_dataset.py --n 500 --seed 20260609
python scripts/run_ocr.py --engine auto
python scripts/evaluate_ocr.py
python scripts/generate_report.py
streamlit run app.py
```

On Windows, `start_app.bat` can be used to launch the Streamlit app.

## Dashboard

The Streamlit app includes:

- Overall field-level accuracy and image-level accuracy
- Average confidence and high-risk sample count
- Error type distribution
- Device category error-rate comparison
- Field accuracy comparison
- Image quality versus OCR accuracy
- Review queue filtering by error type, quality tag, device type, and confidence
- Single-image inspection with manual labels, OCR output, and attribution
- Auto-generated Markdown report download

## Data Statement

This project uses synthetic images and simulated engineering data. It does not contain company internal data and does not represent any real port, construction site, or enterprise system.
