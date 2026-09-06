# FINDINGS — TRINETRA-HARD, PaddleOCR baseline

Submission-facing summary. Predictor: PaddleOCR PP-OCRv4-mobile (CPU), via
`benchmarks/paddle_predictor.py`. Ground truth: synthetic-generated
(`label_source: synthetic_truth`) — the true plate string is the generator's
own filename, never an OCR guess and never human-labeled, so scoring this
engine against it is a real, non-circular measurement. Full detail and raw
JSON: `benchmarks/reports/FUSION_DELTA_paddle_fixed_distance.md`,
`FUSION_DELTA_paddle_approach.md`.

## Finding A — Fusion doubles the correct-plate rate where plates are readable

At `>100px` plate height, fusion raises the correct-plate rate from **0.19
(95/494) to 0.43 (213/494)** — more than double, on fixed-distance tracks
(width held constant within the track; see Methodology). `80-100px` shows
the same direction, smaller magnitude: **0.02 (12/506) to 0.16 (80/506)**.

This is the headline result: temporal consensus across frames of the same
`TrackKey` recovers plates that a single frame does not, but only where a
frame in the track is individually readable to begin with.

## Finding B — ~20px of plate height is a hard floor for this engine

Below ~20px of plate height, fusion changes essentially nothing:

| width bucket | mean plate height (px) | fusion OFF | fusion ON |
|---|---|---|---|
| 60-80px | 15px | 0.00 (2/519) | 0.00 (0/519) |
| 40-60px | 12px | 0.00 (0/503) | 0.00 (0/503) |
| 30-40px | 8px | 0.00 (0/493) | 0.00 (0/493) |
| <30px | 5px | 0.00 (0/506) | 0.00 (0/506) |

`60-80px` is not perfectly zero — 2 of 519 frames read correctly without
fusion. Small enough to be noise at this n, but real: worth stating plainly
rather than rounding it out of existence. Below ~20px, this is an empirical
floor measured on this corpus, not a scorer defect and not a fusion failure
— there is essentially no readable frame anywhere in these tracks for
consensus to recover, by construction (fixed-distance tracks hold width,
and therefore plate height, constant across every frame). **Practical
implication: camera placement/zoom must keep the plate above ~20px of
rendered height for this engine family to have any chance at all.** A
different (heavier, or purpose-trained) OCR model may push this floor lower;
this number is specific to PP-OCRv4-mobile, not a law of physics.

## Finding C — Fusion introduces fabrication

Fusion's consensus rule (highest-confidence whole reading across a track)
can propagate a confident plate string onto a frame that was individually
unreadable. Fabrication count, fixed-distance tracks: **0 (fusion OFF) to
113 (fusion ON)**, out of 3,021 scored frames. This is never folded into the
accuracy rate above — it is reported as its own count, per CLAUDE.md's
measurement contract — but it is a real cost of temporal consensus that a
production fusion design has to budget for: a track-level "best reading"
policy has no per-frame eligibility gate, so it will confidently label
frames that should stay unlabeled.

The same mechanism has a quieter, opposite-direction cost: at `60-80px`,
fusion turned **2 correct readings into 0** (0.00 (2/519) OFF -> 0.00
(0/519) ON — see Finding B). Small enough to be noise at this n, but it is
the same track-level consensus rule doing the same thing it does everywhere
else: overwriting a per-frame answer with the track's single "best"
reading. When that reading is wrong, a track-level policy can discard a
correct single-frame answer just as readily as it can rescue an incorrect
one. State plainly as the other side of the fusion tradeoff, not just the
fabrication cost above.

## Methodology note — why per-bucket numbers require fixed-distance tracks

Two track types were built and scored separately:

- **Approach tracks** (plate width sweeps within the track, modeling a
  vehicle crossing the camera's field of view): fusion ON lands at ~0.19-0.21
  in **every** width bucket, including `<30px`. This is not six independent
  results — it is one number six times. Because every approach track visits
  every width bucket, the track's single most-confident reading (usually
  from its widest, easiest frame) gets credited to that track's narrowest
  frames too under a whole-track consensus rule. The flat rate is a corpus
  artifact of that design, not a per-bucket measurement.
- **Fixed-distance tracks** (plate width held constant within the track,
  degradation varied per frame instead) remove that artifact: a `<30px`
  track has no wide, easy frame anywhere in it to borrow a reading from. The
  fixed-distance numbers above are the honest per-bucket answer; the
  approach-track numbers are kept only as a labeled demonstration of the
  artifact, never cited as a width-bucket result.

Both track types are scored from the same corpus, same predictor, same
commit — see the individual reports for manifest hashes and full JSON.
