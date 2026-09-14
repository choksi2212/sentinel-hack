**Predictor: `stub`** (`benchmarks/stub_predictor.py`, canned/illustrative — not a real model. Replace before citing this table as a measurement.)

Ground truth below is synthetic-generated (`label_source: synthetic_truth`), not human- or OCR-labeled; the real-footage (indian_road) figure is a stability diagnostic only, reported separately in `STABILITY.md` — it is NOT an accuracy number and must never be presented as one.

# FUSION_DELTA

Before: `e2e_fusion_off_005.json` (manifest `abd2637c4ae316fa1ac79a022f67ce116b9acf97268a360e6af95f934428153a`, commit `b45dc3f985c255263f129bcb8a47e8c66dc6d14b`)
After: `e2e_fusion_on_005.json` (manifest `abd2637c4ae316fa1ac79a022f67ce116b9acf97268a360e6af95f934428153a`, commit `b45dc3f985c255263f129bcb8a47e8c66dc6d14b`)

| Bucket | n | fusion OFF | fusion ON | delta |
|---|---|---|---|---|
| >100 | 984 | 0.96 (942/984) | 0.98 (964/984) | +0.02 |
| 80-100 | 481 | 0.90 (432/481) | 0.96 (461/481) | +0.06 |
| 60-80 | 504 | 0.75 (380/504) | 0.87 (439/504) | +0.12 |
| 40-60 | 499 | 0.48 (238/499) | 0.72 (360/499) | +0.24 |
| 30-40 | 273 | 0.29 (79/273) | 0.45 (122/273) | +0.16 |
| <30 | 361 | 0.12 (45/361) | 0.17 (63/361) | +0.05 |
| ALL | 3102 | 0.68 (2116/3102) | 0.78 (2409/3102) | +0.09 |

Fabrication count -- OFF: 321, ON: 326 (never folded into the rate above).

**Notes:**
- (before) 4816 unverified_real rows present (real footage, no known plate text) -- excluded from every number above; see STABILITY.md for that data.
- (before) weights_sha256: n/a -- stub predictor has no weights file to hash.
- (after) 4816 unverified_real rows present (real footage, no known plate text) -- excluded from every number above; see STABILITY.md for that data.
- (after) weights_sha256: n/a -- stub predictor has no weights file to hash.
