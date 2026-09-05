**Predictor: `stub`** (`benchmarks/stub_predictor.py`, canned/illustrative — not a real model. Replace before citing this table as a measurement.)

Ground truth below is synthetic-generated (`label_source: synthetic_truth`), not human- or OCR-labeled; the real-footage (indian_road) figure is a stability diagnostic only, reported separately in `STABILITY.md` — it is NOT an accuracy number and must never be presented as one.

# FUSION_DELTA

Before: `e2e_fusion_off_004.json` (manifest `abd2637c4ae316fa1ac79a022f67ce116b9acf97268a360e6af95f934428153a`, commit `d57843c454005c3714d72992b3b2cea3eb178172`)
After: `e2e_fusion_on_004.json` (manifest `abd2637c4ae316fa1ac79a022f67ce116b9acf97268a360e6af95f934428153a`, commit `d57843c454005c3714d72992b3b2cea3eb178172`)

| Bucket | n | fusion OFF | fusion ON | delta |
|---|---|---|---|---|
| >100 | 984 | 1.00 (984/984) | 1.00 (984/984) | +0.00 |
| 80-100 | 481 | 1.00 (481/481) | 1.00 (481/481) | +0.00 |
| 60-80 | 504 | 0.00 (0/504) | 1.00 (504/504) | +1.00 |
| 40-60 | 499 | 0.00 (0/499) | 1.00 (499/499) | +1.00 |
| 30-40 | 273 | 0.00 (0/273) | 1.00 (273/273) | +1.00 |
| <30 | 361 | 0.00 (0/361) | 1.00 (361/361) | +1.00 |
| ALL | 3102 | 0.47 (1465/3102) | 1.00 (3102/3102) | +0.53 |

Fabrication count -- OFF: 169, ON: 335 (never folded into the rate above).

**Notes:**
- (before) 4816 unverified_real rows present (real footage, no known plate text) -- excluded from every number above; see STABILITY.md for that data.
- (before) weights_sha256: n/a -- stub predictor has no weights file to hash.
- (after) 4816 unverified_real rows present (real footage, no known plate text) -- excluded from every number above; see STABILITY.md for that data.
- (after) weights_sha256: n/a -- stub predictor has no weights file to hash.
