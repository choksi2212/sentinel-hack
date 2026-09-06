**Predictor: `paddle`** (`benchmarks/paddle_predictor.py`, PaddleOCR (see `weights_sha256`/notes below for the exact model files) — a real OCR baseline, not the final production pipeline. Ground truth is synthetic-generated (see below), so this is a real, non-circular measurement of this specific OCR engine.)

Ground truth below is synthetic-generated (`label_source: synthetic_truth`), not human- or OCR-labeled; the real-footage (indian_road) figure is a stability diagnostic only, reported separately in `STABILITY.md` — it is NOT an accuracy number and must never be presented as one.

# FUSION_DELTA

Before: `e2e_fusion_off_paddle_fixed_distance_001.json` (manifest `b5df9dfd90f4f2eaf7426e0015a1afe115272639c74df63c06b0998a83116fcb`, commit `ad52135024d0c675ca36245d44ec68898850e810`)
After: `e2e_fusion_on_paddle_fixed_distance_001.json` (manifest `b5df9dfd90f4f2eaf7426e0015a1afe115272639c74df63c06b0998a83116fcb`, commit `ad52135024d0c675ca36245d44ec68898850e810`)

| Bucket | n | plate height (px) | fusion OFF | fusion ON | delta |
|---|---|---|---|---|---|
| >100 | 494 | 31px | 0.19 (95/494) | 0.43 (213/494) | +0.24 |
| 80-100 | 506 | 20px | 0.02 (12/506) | 0.16 (80/506) | +0.13 |
| 60-80 | 519 | 15px (below floor) | 0.00 (2/519) | 0.00 (0/519) | -0.00 |
| 40-60 | 503 | 12px (below floor) | 0.00 (0/503) | 0.00 (0/503) | +0.00 |
| 30-40 | 493 | 8px (below floor) | 0.00 (0/493) | 0.00 (0/493) | +0.00 |
| <30 | 506 | 5px (below floor) | 0.00 (0/506) | 0.00 (0/506) | +0.00 |
| ALL | 3021 | -- | 0.04 (109/3021) | 0.10 (293/3021) | +0.06 |

Fabrication count -- OFF: 0, ON: 113 (never folded into the rate above).

**60-80, 40-60, 30-40, <30** average under ~20px of plate height — below ~20px of plate height this engine detects no text regardless of fusion. An empirical floor measured on this corpus, not a scorer defect.

**Notes:**
- (before) weights_sha256 is a combined hash of 30 model files (PP-OCRv4 (paddleocr 3.7.0, paddlepaddle 3.3.1, mkldnn disabled)).
- (after) weights_sha256 is a combined hash of 30 model files (PP-OCRv4 (paddleocr 3.7.0, paddlepaddle 3.3.1, mkldnn disabled)).
