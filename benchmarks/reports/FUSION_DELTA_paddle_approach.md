**Predictor: `paddle`** (`benchmarks/paddle_predictor.py`, PaddleOCR (see `weights_sha256`/notes below for the exact model files) — a real OCR baseline, not the final production pipeline. Ground truth is synthetic-generated (see below), so this is a real, non-circular measurement of this specific OCR engine.)

Ground truth below is synthetic-generated (`label_source: synthetic_truth`), not human- or OCR-labeled; the real-footage (indian_road) figure is a stability diagnostic only, reported separately in `STABILITY.md` — it is NOT an accuracy number and must never be presented as one.

# FUSION_DELTA

Before: `e2e_fusion_off_paddle_approach_001.json` (manifest `b5df9dfd90f4f2eaf7426e0015a1afe115272639c74df63c06b0998a83116fcb`, commit `ad52135024d0c675ca36245d44ec68898850e810`)
After: `e2e_fusion_on_paddle_approach_001.json` (manifest `b5df9dfd90f4f2eaf7426e0015a1afe115272639c74df63c06b0998a83116fcb`, commit `ad52135024d0c675ca36245d44ec68898850e810`)

| Bucket | n | plate height (px) | fusion OFF | fusion ON | delta |
|---|---|---|---|---|---|
| >100 | 987 | 27px | 0.17 (163/987) | 0.21 (210/987) | +0.05 |
| 80-100 | 484 | 20px | 0.07 (32/484) | 0.21 (101/484) | +0.14 |
| 60-80 | 503 | 16px (below floor) | 0.01 (5/503) | 0.19 (94/503) | +0.18 |
| 40-60 | 500 | 11px (below floor) | 0.00 (0/500) | 0.19 (94/500) | +0.19 |
| 30-40 | 274 | 8px (below floor) | 0.00 (0/274) | 0.19 (51/274) | +0.19 |
| <30 | 363 | 5px (below floor) | 0.00 (0/363) | 0.20 (74/363) | +0.20 |
| ALL | 3111 | -- | 0.06 (200/3111) | 0.20 (624/3111) | +0.14 |

Fabrication count -- OFF: 0, ON: 152 (never folded into the rate above).

**Notes:**
- (before) weights_sha256 is a combined hash of 30 model files (PP-OCRv4 (paddleocr 3.7.0, paddlepaddle 3.3.1, mkldnn disabled)).
- (after) weights_sha256 is a combined hash of 30 model files (PP-OCRv4 (paddleocr 3.7.0, paddlepaddle 3.3.1, mkldnn disabled)).
