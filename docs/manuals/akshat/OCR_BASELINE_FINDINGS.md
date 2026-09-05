# OCR_BASELINE_FINDINGS — PaddleOCR (PP-OCRv4-mobile) on the synthetic corpus

For Manas — model selection input. From the first real-predictor run
(approach tracks, 3,437 frames, before the plate_bbox/width_bucket fix
documented in RUNLOG — the **condition breakdown below does not depend on
that bug**: it groups by `slice`/degradation, not by the mislabelled width
bucket, so it stands on its own).

## Condition breakdown at the largest nominal size (`>100` bucket)

| Condition | Detection rate | n |
|---|---|---|
| easy (no added degradation) | 64% | 192/298 |
| glare | 49% | 81/165 |
| perspective | 52% | 96/183 |
| night | 26% | 58/222 |
| **motion_blur** | **0.5%** | **1/215** |

## The headline finding

**Motion blur defeats PP-OCRv4-mobile almost completely, independent of
plate size.** Even at this run's largest nominal width, a motion-blurred
plate is read correctly essentially never (1/215). This is not a
resolution problem — it doesn't improve by making the plate bigger. It is
a property of this specific (lightweight, "mobile") model against this
specific degradation.

Practical implication for model selection: if the real deployment expects
motion blur (moving vehicles, rolling shutter, low frame rate), a heavier
recognition model, motion-deblurring as a pre-processing stage, or a
detector explicitly trained/fine-tuned on blurred plates is likely required
— PP-OCRv4-mobile's out-of-the-box blur robustness is not sufficient on its
own, at any distance.

Night is the second-largest gap (26%), consistent with a general-purpose
OCR model not being tuned for low-light/IR-style capture; glare and
perspective land in a similar, moderate 49-52% band. `easy` (undegraded)
tops out at 64% even at this run's largest available size — see
`docs/manuals/akshat/RUNLOG.md` for the separate finding that the
synthetic renderer's `plate_bbox` was not tightly cropped to the plate
(canvas size used instead), which suppresses effective resolution across
every condition including `easy`; a corrected run is in progress and will
supersede the exact percentages here without changing this ranking's shape
(motion_blur << night < glare ~ perspective < easy).

## Status

This is a first-pass baseline against one lightweight OCR engine (PaddleOCR
PP-OCRv4-mobile, CPU), not a final model comparison. Numbers here will be
superseded once the plate_bbox fix lands and the corpus is re-run — treat
the **relative ranking** of conditions as the durable finding, the exact
percentages as provisional.
