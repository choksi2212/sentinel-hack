**Predictor: `paddle`** (`benchmarks/stub_predictor.py`, canned/illustrative — not a real model. Replace before citing this table as a measurement.)

Ground truth below is synthetic-generated (`label_source: synthetic_truth`), not human- or OCR-labeled; the real-footage (indian_road) figure is a stability diagnostic only, reported separately in `STABILITY.md` — it is NOT an accuracy number and must never be presented as one.

# FUSION_DELTA

Before: `e2e_fusion_off_paddle_001.json` (manifest `abd2637c4ae316fa1ac79a022f67ce116b9acf97268a360e6af95f934428153a`, commit `4b2930f09cf5709eb80fd8c704a40a8559429620`)
After: `e2e_fusion_on_paddle_001.json` (manifest `abd2637c4ae316fa1ac79a022f67ce116b9acf97268a360e6af95f934428153a`, commit `4b2930f09cf5709eb80fd8c704a40a8559429620`)

| Bucket | n | fusion OFF | fusion ON | delta |
|---|---|---|---|---|
| >100 | 984 | 0.17 (172/984) | 0.23 (231/984) | +0.06 |
| 80-100 | 481 | 0.07 (36/481) | 0.22 (108/481) | +0.15 |
| 60-80 | 504 | 0.01 (7/504) | 0.21 (105/504) | +0.19 |
| 40-60 | 499 | 0.00 (0/499) | 0.21 (103/499) | +0.21 |
| 30-40 | 273 | 0.00 (0/273) | 0.21 (57/273) | +0.21 |
| <30 | 361 | 0.00 (0/361) | 0.22 (79/361) | +0.22 |
| ALL | 3102 | 0.07 (215/3102) | 0.22 (683/3102) | +0.15 |

Fabrication count -- OFF: 0, ON: 164 (never folded into the rate above).

**Notes:**
- (before) 4816 unverified_real rows present (real footage, no known plate text) -- excluded from every number above; see STABILITY.md for that data.
- (before) weights_sha256 is a combined hash of 30 model files (PP-OCRv4 (paddleocr 3.7.0, paddlepaddle 3.3.1, mkldnn disabled)).
- (after) 4816 unverified_real rows present (real footage, no known plate text) -- excluded from every number above; see STABILITY.md for that data.
- (after) weights_sha256 is a combined hash of 30 model files (PP-OCRv4 (paddleocr 3.7.0, paddlepaddle 3.3.1, mkldnn disabled)).
