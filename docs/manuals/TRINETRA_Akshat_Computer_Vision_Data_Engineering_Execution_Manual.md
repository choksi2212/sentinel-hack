# TRINETRA — EXECUTION MANUAL
## Akshat — Computer Vision Data & Benchmarking

**Version 2.0 · 2026-09-01 · 6 build days to the qualification gate**

> **Precedence.** [`docs/TRINETRA_Canonical_Contracts.md`](docs/TRINETRA_Canonical_Contracts.md) is normative. Blocks marked `COPIED FROM CANONICAL — DO NOT EDIT HERE` are verbatim.

---

## 1. Your job in one paragraph

You decide what is true. Manas builds the pipeline; you are the reason anyone believes a number it produces. You own the datasets, their licences, the frozen test set nobody is allowed to tune on, the synthetic corpus that covers what real data doesn't, and the benchmark reports that decide whether we spend money on an A100. Your hardest constraint is not technical — it is discipline. Every shortcut available to you produces a better-looking number and a less true one.

**You own:** `datasets/` · `datasets/LICENSES.md` · `benchmarks/` · `scripts/synth/` · TRINETRA-HARD · every accuracy claim in the submission.

**You do not own:** the pipeline, the schema, the UI. You measure them.

**Your machine:** RTX 4060 8 GB. The tightest GPU on the team, which makes you the natural owner of the memory-guard checks — if it runs for you, it runs for everyone.

---

## 2. Day plan — anchored to the real calendar

Today is **1 September 2026**. Submission is **7 September**. Hackathon is **10–11 September**. Six build days.

| Date | Day | You must finish | Proof |
|---|---|---|---|
| **Sep 1** | D1 | Licence register started; thundarstrom + justjuu downloaded; `datasets/` layout; **first offline assets committed** | `LICENSES.md` has a row per asset; Manas has a clip to test with by evening |
| **Sep 2** | D2 | Manifest hashing, split-leakage check, ground-truth format | `check_split_leakage.py` green; SHA-256 manifest committed |
| **Sep 3** | D3 | **TRINETRA-HARD v1 frozen** (~1,000 obs) + baseline benchmark | Baseline report with width buckets; taxonomy input to Manas |
| **Sep 4** | D4 | Synthetic corpus generator + 4 difficulty tiers | 10k+ synthetic plates; per-tier accuracy measured |
| **Sep 5** | D5 | Fusion before/after benchmark; regression set; eval one-command | The consensus delta number |
| **Sep 6** | D6 | Final benchmark freeze; A100 gate rows; report pack | All 11 gate rows answered; reports in `benchmarks/` |
| Sep 7 | — | **SUBMIT** | — |
| Sep 8–9 | D7–D8 | Post-training eval if A100 was gated GO; TRINETRA-HARD rebalance (once) | Keep-or-discard decision made on E2E value |

**D1 matters more than it looks.** Manas cannot develop against a live-only stream. Until you commit usable offline clips, the whole team is blocked on network weather.

---

## 3. Licensing — do this before you download anything

**"Free to download" ≠ "safe to reuse."** This is a government-facing submission. A judge who asks "what licence is that dataset?" and gets a shrug has learned something about the whole project.

### 3.1 The register is mandatory

`datasets/LICENSES.md` — one row per asset, **nine fields**, no exceptions:

| Field | Example |
|---|---|
Asset name | `thundarstrom-indian-plates`
Source URL | `https://…`
Licence | `CC BY 4.0`
Commercial use | Yes
Redistribution | Yes with attribution
Modification | Yes
Attribution text | *exact string to reproduce*
Our use | Plate detector training
Date recorded | 2026-09-01

Write the row **when you download**, not later. "Later" means the night before submission, reconstructing provenance from browser history.

### 3.2 The Ultralytics rule

**Ultralytics YOLO is AGPL-3.0.** AGPL's network-use clause means deploying a service built on it can obligate source disclosure. For a police platform, that is a non-starter.

- Ultralytics **may** appear in a benchmark comparison table, clearly labelled, as a reference point.
- Ultralytics **may not** appear in the shipped pipeline.

Corollary that has already caused one error in these documents: **RF-DETR is Roboflow's, Apache-2.0.** It is not an Ultralytics model. If you see it attributed to Ultralytics, correct it — misattribution re-imports exactly the licence exposure the stack was chosen to avoid.

### 3.3 Register — locked

| Dataset | Size | Licence | Use | Status |
|---|---|---|---|---|
| **thundarstrom** Indian plates | 3,742 imgs | CC BY 4.0 | Plate detection training | **USE — D1** |
| **justjuu** plate detection | 8,823 (6,176/1,765/882) | CC BY 4.0 | Plate detection training | **USE — D1** |
| `justjuu/rtdetr-v2-license-plate-detection` | weights | Apache-2.0 | Plate detector baseline | **USE — D1** |
| **FANVID** | 1,463 clips @180×320, 49 plates, 31,096 boxes | CC BY 4.0 | **Evaluation only** | USE — eval |
| **CCPD** | 300k+ | MIT | **Localization pretraining only** | USE — restricted |
| Own synthetic corpus | generated | Ours | OCR training | **BUILD — D4** |
| **TRINETRA-HARD** | ~1,000 obs | Ours | Frozen eval | **BUILD — D3** |
| RF-DETR Nano/Small/Medium/Large | weights | Apache-2.0 | Vehicle detection | USE |
| RF-DETR Plus/XL/2XL | weights | different terms | — | **EXCLUDED** |
| ByteTrack | code | MIT | Tracking | USE |
| PaddleOCR | code+weights | Apache-2.0 | OCR | USE |
| RTMDet / YOLOX | weights | Apache-2.0 | Detector fallbacks | USE if needed |
| Ultralytics YOLO | weights | **AGPL-3.0** | Benchmark comparison only | **NEVER SHIPPED** |
| DataCluster | — | **CC BY-NC-ND** | — | **EXCLUDED** (non-commercial, no derivatives) |
| UFPR-ALPR | — | restrictive | — | **EXCLUDED** |
| Gamester03 | — | unverified | — | **PENDING** — do not use until verified |
| thirdeyelabs | 210 GB | check | Sample 10–30k if needed | OPTIONAL |
| Open Images V7 | large | CC BY 4.0 | Vehicle diversity | OPTIONAL |
| BDD100K | large | check terms | Night/weather | OPTIONAL |
| VeRi / CityFlow / VehicleID | — | varies | Re-ID | **DEFERRED** |

Two restrictions with teeth:

**CCPD is localization-only.** 300k images is tempting for OCR, but they are **Chinese** plates. Training Indian OCR on Chinese plate grammar teaches the model the wrong character distribution, the wrong province-code structure, and the wrong aspect ratios. It will produce plausible-looking Chinese-shaped output for Indian plates, and the errors will be systematic rather than random — which is much harder to notice.

**FANVID is evaluation-only.** 180×320 clips with 49 distinct plates. Train on it and you memorize 49 plates; the benchmark then reports how well you recall them. That is not a measurement of anything.

---

## 4. Dataset roles — the do-not-mix table

| Dataset | Allowed | Forbidden | Why the misuse is fatal |
|---|---|---|---|
| thundarstrom | Plate detection train/val | Appearing in TRINETRA-HARD | Test contamination — accuracy becomes recall of training data |
| justjuu | Plate detection train/val | Appearing in TRINETRA-HARD | Same |
| CCPD | Localization pretraining | Indian OCR grammar | Systematic wrong-alphabet errors that look like random noise |
| FANVID | Evaluation | Any training | 49 plates memorized; benchmark measures nothing |
| Synthetic | OCR + hard-case training | Being the **only** eval | Synthetic-to-real gap hidden until demo day |
| TRINETRA-HARD | Final frozen eval | **Any** training, tuning, or threshold selection | The one honest number is destroyed, and you cannot tell |

That last row is the one that will actually tempt you, at 2 am on D6, when a threshold tweak turns 71% into 78%. Once you tune on the test set, the number is no longer a measurement — and there is no way to detect from the outside that it happened. You lose the only instrument you have.

### 4.1 Enforce it in code, not memory

```bash
python scripts/check_split_leakage.py
```

Must check:
- Perceptual hash overlap between train/val/test
- Exact file-hash overlap
- Same plate string appearing across splits
- TRINETRA-HARD source groups disjoint from every training source

Run it in CI. A green check is worth more than an intention, because on D6 nobody will remember which folder they copied a file from.

---

## 5. TRINETRA-HARD — build it on D3, freeze it, don't touch it

Real-world CCTV is nothing like a benchmark dataset. Plates are small, motion-blurred, lit by sodium vapour, angled 40° off axis, and occasionally behind a bus.

**~1,000 observations:**

| Slice | Count |
|---|---|
Easy / clean | 200
Motion blur | 200
Night / low light | 200
Glare / headlight wash | 150
Extreme perspective | 150
Tiny plates (<30 px) | 100

Rules:
- Source groups **disjoint** from all training data
- Ground truth verified by a human, character by character
- Rebalanced **exactly once**, before the final freeze, never in response to a bad result
- Never trained on, never tuned on, never used to pick a threshold

The point of over-weighting the hard slices is that the easy ones are already solved. A benchmark that is 80% easy images reports a number dominated by cases nobody was worried about.

### 5.1 Width buckets — report every metric this way

`COPIED FROM CANONICAL — DO NOT EDIT HERE` (Contracts §7)

```
>100 px · 80–100 px · 60–80 px · 40–60 px · 30–40 px · <30 px
```

A single average is not an acceptable deliverable. "92% accuracy" usually decomposes into 98% above 80 px and 51% below 40 px. The second number determines whether this works on real Gujarat infrastructure, where most plates are small. Publish the breakdown yourself — a judge who extracts it from you has found a weakness; a judge who reads it in your report has found rigour.

---

## 6. The primary metric

`COPIED FROM CANONICAL — DO NOT EDIT HERE` (Contracts §7)

```
E2E correct-plate event rate = correct final plate events / eligible vehicle events
```

- **Eligible** = plate human-readable in ≥1 sampled frame of ground truth
- **Correct** = fused `normalized` string exactly equals ground truth

Everything else — mAP, CER, FPS, p95 latency, VRAM — is a **diagnostic**. Diagnostics explain the primary number. They never substitute for it.

This is not pedantry. A plate detector at 0.97 mAP is fully compatible with a system that reads almost no plates, because detection is one of nine stages and stages multiply. Reporting mAP as the headline is the standard way ALPR projects overstate themselves, and it is transparent to anyone who has built one.

### 6.1 Report shape

```json
{
  "run_id": "bench-2026-09-03-baseline",
  "dataset": {"name": "TRINETRA-HARD", "version": "v1", "manifest_sha256": "…"},
  "models": {
    "detector": "rfdetr-small", "detector_weights_sha256": "…",
    "plate_detector": "rtdetrv2-justjuu", "ocr": "paddleocr-3.0",
    "tracker": "bytetrack", "pipeline_version": "0.1.0"
  },
  "hardware": {"gpu": "RTX 4060 8GB", "driver": "…"},
  "primary": {"e2e_correct_plate_event_rate": 0.71, "eligible_events": 842, "correct_events": 598},
  "by_width_bucket": {
    ">100": 0.94, "80-100": 0.89, "60-80": 0.78,
    "40-60": 0.62, "30-40": 0.41, "<30": 0.08
  },
  "diagnostics": {"plate_map50": 0.91, "ocr_cer": 0.11, "fps": 38.2, "p95_latency_ms": 142, "peak_vram_mb": 4820},
  "failure_buckets": {"plate_too_small": 118, "ocr_wrong": 71, "plate_miss": 40, "vehicle_miss": 15}
}
```

Every field is required. A report without `manifest_sha256` and the model hashes is not reproducible, which means it is not evidence.

### 6.2 Warm-up

Discard the **first 2** predictions from every timing measurement. CUDA autotuning and lazy allocation make them 5–20× slower than steady state. Including them makes every latency number in the report fiction — and low by a factor that varies run to run, so it isn't even consistently wrong.

### 6.3 Ground truth format — define this on D2

You cannot score the primary metric without a ground-truth format that keys to the same identity the pipeline uses. One JSONL file per clip, one line per ground-truth vehicle pass:

```json
{"clip": "ahmedabad_ringroad_01.mp4", "gt_id": 12,
 "plate": "GJ01AB1234",
 "eligible": true,
 "first_pts_ms": 341800, "last_pts_ms": 344900,
 "plate_width_px_max": 62,
 "width_bucket": "60-80",
 "slice": "motion_blur",
 "notes": "partial glare on last frame"}
```

| Field | Rule |
|---|---|
`gt_id` | Yours, stable per clip. Not a `track_id` — the pipeline's track IDs are its output, not your input. |
`plate` | Normalized form, `^[A-Z0-9]+$`, human-verified character by character |
`eligible` | `true` if readable by a human in ≥1 sampled frame. **This is the denominator.** |
`first_pts_ms` / `last_pts_ms` | Source-timeline window, from PTS — never wallclock |
`plate_width_px_max` | Largest observed plate width; determines `width_bucket` |
`slice` | Which TRINETRA-HARD slice: `easy`/`motion_blur`/`night`/`glare`/`perspective`/`tiny` |

`eligible: false` rows still belong in the file. A plate no human can read is not a model failure, and excluding those rows entirely means you can never report how many there were — which is itself a finding about camera placement.

### 6.4 How the eval harness reads events

The harness consumes the same `EventEnvelope` v1.1 Mihir ingests (Contracts §3). The three fields you actually need:

```json
{"camera_id": "cam04",
 "stream_session_id": "3a7f1e02-…",
 "track_id": 42,
 "source_pts_ms": 343100,
 "plate": {"normalized": "GJ01AB1234", "plate_width_px": 62,
           "evidence_count": 3, "match_state": "probable"}}
```

Matching rule — **key on `TrackKey = (camera_id, stream_session_id, track_id)`**, then align to ground truth by PTS window overlap:

```python
def match(event, gt_rows):
    return [g for g in gt_rows
            if g["first_pts_ms"] - 500 <= event["source_pts_ms"] <= g["last_pts_ms"] + 500]
```

Two rules that decide whether your numbers mean anything:

**Align on `source_pts_ms`, never on `observed_at`.** Wallclock drifts during replay and is meaningless under `--mode fast`. PTS is the source timeline and is the only clock your ground truth shares with the pipeline.

**Never key on `(camera_id, track_id)` alone.** If you drop the session, two vehicles from either side of a reconnect collapse into one, and your scorer will report a miss and a false positive for what was actually correct behaviour — sending Manas to debug a model that is fine.

Scoring, with `plate: null` handled explicitly:

| Event | Ground truth | Counts as |
|---|---|---|
`normalized` == `gt.plate` | `eligible: true` | **correct** |
`normalized` != `gt.plate` | `eligible: true` | incorrect → bucket `ocr_wrong` |
`plate: null` | `eligible: true` | incorrect → bucket by cause |
`plate: null` | `eligible: false` | **not counted** — correct abstention |
`normalized` non-null | `eligible: false` | **fabrication — report separately and loudly** |
no event at all | `eligible: true` | incorrect → `vehicle_miss` or `dropped_frame` |

That fifth row deserves its own line in every report. A plate produced where no human could read one is not a scoring detail; it is the system asserting something it cannot know, and it is the single most damaging failure this project can have. Count it, name it, and tell Manas the same day.

---

## 7. Benchmark suite — seven tasks

| # | Task | Output |
|---|---|---|
| 1 | Vehicle detection | mAP@50, mAP@50-95 per class, incl. auto-rickshaw |
| 2 | Plate detection | mAP@50, small-object mAP, recall by width bucket |
| 3 | OCR isolated | CER, exact-match rate, per width bucket |
| 4 | **E2E correct-plate rate** | **the primary metric** |
| 5 | Throughput | FPS at 1 / 4 / 8 / 16 simulated cameras |
| 6 | Latency | p50 / p95 / p99 frame→event, warm-up discarded |
| 7 | Resource | peak VRAM, RSS, at each camera count |

Task 5 matters more than it looks. Single-camera FPS is not the operating point — a 30-camera grid is throughput-bound, and a model that is fast alone and slow at 8 concurrent streams is the wrong model. Measure the shape of the curve, not one point on it.

### 7.1 One command, unattended

```bash
python -m benchmarks.run --suite all --dataset trinetra-hard --out benchmarks/reports/
```

This is gate row 11 for the A100 rental. If evaluation needs a human babysitting it, you cannot evaluate a trained model at 3 am while the pod meter runs.

### 7.2 Regression set

A small fast subset (~100 obs) that runs in under two minutes on every commit. Its job is catching the change that improves the headline number while breaking something else — the classic being a preprocessing tweak that helps big plates and destroys small ones.

---

## 8. Synthetic corpus — D4

Real Indian CCTV plate data at 30–60 px with verified ground truth is scarce. Synthetic data is how you cover that gap, and its great advantage is that ground truth is **free and exact** — you know the string because you drew it.

### 8.1 Generation chain

```
plate string (Indian grammar)
  → render on correct plate background, correct font, correct spacing
  → perspective warp (yaw/pitch/roll)
  → motion blur (directional, magnitude by tier)
  → downscale to target width bucket
  → JPEG compression artefacts
  → lighting: day / dusk / sodium vapour / headlight glare
  → composite onto real road background
```

Order matters. Downscale **before** compression, because that is the physical order in a real camera. Reverse them and you produce artefacts no real camera makes, and the model learns to depend on them.

### 8.2 Difficulty tiers

| Tier | Plate width | Purpose |
|---|---|---|
Easy | >100 px | Sanity check
Medium | 60–100 px | Realistic good case
Hard | 30–60 px | **The operating point** — spend your generation budget here
Extreme | <30 px | Failure boundary characterisation

Most real detections land in Hard. Generate accordingly — a corpus that is mostly Easy trains for a case that was never the problem.

### 8.3 Grammar

```
^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$
```

Weight Gujarat state codes (`GJ`) heavily, include neighbours (`MH`, `RJ`, `MP`, `DL`), include BH-series. Include the confusable characters deliberately — `0/O`, `1/I/L`, `8/B`, `5/S`, `2/Z`, `6/G` — because those are where the errors are, and a corpus that avoids them trains a model that has never seen the hard part.

### 8.4 Mixture hypotheses — these are hypotheses, test them

**Plate detector:** 35% Indian real · 25% general real · 20% hard synthetic · 20% augmentation
**OCR:** 50% synthetic Indian · 30% real Indian · 20% low-res temporal sequences

Labelled hypotheses, not settled facts. If a 50/50 split beats 35/25/20/20 on TRINETRA-HARD, the data wins. Record what you tried, including what failed — an ablation table is worth more to a judge than a single ratio presented as received wisdom.

---

## 9. The fusion measurement — your best deliverable

On D5, run the E2E benchmark twice:

1. **Single-frame OCR** — best available frame per track
2. **Temporal consensus** — the fusion in Contracts §4.3

Report the delta by width bucket.

This is the most valuable number you will produce, because the gain comes from **algorithm design, not compute**. It costs no GPU, no rental, and no training. If single-frame is 58% and consensus is 71% at 40–60 px, that thirteen-point gain is a defensible engineering claim about a real design decision — which is a stronger story than a fine-tune that bought two points.

Make sure Manas has this number for the architecture walkthrough.

---

## 10. Failure taxonomy — feed this to Manas on D3

You produce the data; Manas classifies. Every miss lands in exactly one bucket:

`vehicle_miss` · `plate_miss` · `plate_too_small` · `ocr_wrong` · `ocr_partial` · `fusion_wrong` · `track_broken` · `track_merged` · `duplicate` · `dropped_frame`

The output that matters is **which single bucket dominates**. That answer decides whether we rent an A100 and what we would train on it:

| Dominant bucket | What it implies |
|---|---|
`plate_too_small` | **No software fix.** Report width buckets, recommend camera placement. Do not train. |
`ocr_wrong` | Synthetic OCR corpus, possibly a fine-tune |
`plate_miss` | Plate detector training on small objects |
`vehicle_miss` | Vehicle detector training on Indian roads |
`fusion_wrong` | Fusion weighting — free to fix |
`track_merged` | Session handling bug, not a model problem |

Two co-dominant buckets means the analysis isn't finished. Renting a GPU at that point is a guess with a credit card attached.

---

## 11. The A100 gate — you own 5 of 11 rows

| # | Check | Owner |
|---|---|---|
| 1 | Offline pipeline runs end to end | Manas |
| 2 | **Baseline benchmark recorded** | **You** |
| 3 | **TRINETRA-HARD frozen** | **You** |
| 4 | Failure taxonomy written | Manas |
| 5 | Bottleneck is one component | Manas |
| 6 | Local smoke train completes | Manas |
| 7 | **Dataset manifest frozen (SHA-256)** | **You** |
| 8 | **Split leakage check green** | **You** |
| 9 | **Licences recorded for every training asset** | **You** |
| 10 | `config/training.yaml` committed | Manas |
| 11 | **Eval runs unattended, one command** | **You** |

Any red row → **no rental**. You are the person who says no. That is the job.

Cost if we go: $1.64/hr running, $0.083/hr stopped. ~10 h ≈ $16.40, ~25 h ≈ $41. The only paid resource in the entire project.

### 11.1 The ratchet, and the clause that governs it

```
BASELINE → TARGET FAILURE ANALYSIS → LOCAL SMOKE TRAIN → FROZEN DATA/CONFIG
        → A100 TRAIN → FROZEN HARD TEST → KEEP ONLY IF END-TO-END VALUE IMPROVES
```

**+3 mAP and +0 E2E correct-plate rate → the model is discarded.** Whatever it cost, whatever the effort. You enforce this, because you hold the frozen test set and you are the only one positioned to say the trained model isn't better.

### 11.2 Manifest freezing

```bash
python scripts/freeze_manifest.py --dataset trinetra-hard --out datasets/manifests/
```

SHA-256 per file plus a manifest hash. Every benchmark report cites the manifest hash. Without it, "we got 71%" is unfalsifiable — nobody, including you next week, can tell which files that was on.

---

## 12. First offline assets — D1, before anything else

Commit or document these on day one:

| Asset | Purpose |
|---|---|
2–3 clips, 30–120 s, Indian traffic, visible plates | Primary dev loop for Manas |
1 night clip | Low-light path |
1 clip with a hard scene cut | Discontinuity detection |
100-frame JPEG sequence | `FrameSequenceSource` |
Ground-truth plate list for each clip | E2E scoring |

Manas cannot iterate on a live-only, non-seekable stream. This is his primary development path for six days, not a fallback. Get it to him by D1 evening.

Large binaries: keep them out of git or use Git LFS. A 2 GB repo is a repo nobody clones on demo morning.

---

## 13. Anti-patterns

| Do not | Consequence |
|---|---|
| Train on TRINETRA-HARD | The one honest number is destroyed, undetectably |
| Tune a threshold on TRINETRA-HARD | Same thing, wearing a disguise |
| Train Indian OCR on CCPD | Systematic wrong-alphabet errors that look like noise |
| Train on FANVID | You memorize 49 plates and call it accuracy |
| Report a single average accuracy | Hides the small-plate collapse a judge will find |
| Report mAP as the headline | Overstates by ignoring eight other stages |
| Include warm-up in timings | Every latency number is fiction |
| Benchmark at 1 camera only | Wrong operating point for a 30-camera grid |
| Download first, check the licence later | You reconstruct provenance the night before submission |
| Ship Ultralytics YOLO | AGPL-3.0 exposure on a government submission |
| Attribute RF-DETR to Ultralytics | Re-imports the exact licence risk we avoided |
| Use DataCluster | CC BY-NC-ND: non-commercial, no derivatives |
| Use Gamester03 before verifying | Unverified terms |
| Compare runs on different manifests | The comparison means nothing |
| Rebalance TRINETRA-HARD after a bad result | That is tuning on the test set |
| Commit 210 GB | Nobody clones the repo |

---

## 14. Definition of done

- [ ] `datasets/LICENSES.md` complete — 9 fields, one row per asset, zero unverified assets in use
- [ ] thundarstrom + justjuu downloaded, attributed, split
- [ ] CCPD used for localization pretraining only, documented
- [ ] FANVID used for evaluation only, documented
- [ ] First offline assets delivered to Manas on **D1**
- [ ] `check_split_leakage.py` green and in CI
- [ ] Ground-truth JSONL format defined on D2, with `eligible` and `width_bucket` on every row
- [ ] Eval harness keys on `TrackKey` and aligns on `source_pts_ms`, never `observed_at`
- [ ] Fabrication count (plate emitted where `eligible: false`) reported separately in every run
- [ ] SHA-256 manifests frozen; every report cites a manifest hash
- [ ] **TRINETRA-HARD v1 frozen** — ~1,000 obs, 6 slices, sources disjoint, human-verified
- [ ] Baseline benchmark with width buckets and full provenance
- [ ] All 7 benchmark tasks runnable by one command, unattended
- [ ] Regression set under 2 minutes, in CI
- [ ] Synthetic generator with 4 tiers, 10k+ plates, Hard tier weighted
- [ ] Mixture hypotheses tested and the ablation recorded
- [ ] **Fusion before/after delta measured by width bucket**
- [ ] Failure taxonomy data delivered; dominant bucket identified with Manas
- [ ] Your 5 A100 gate rows answered honestly
- [ ] Keep-or-discard decision made on E2E value, not mAP
- [ ] TRINETRA-HARD rebalanced at most once, before final freeze

---

## 15. Your daily loop

```bash
python scripts/check_split_leakage.py
python -m benchmarks.run --suite regression --dataset trinetra-hard
python -m benchmarks.run --suite all --dataset trinetra-hard --out benchmarks/reports/
git diff --stat datasets/LICENSES.md
```

**Final principle:** every number in the submission is yours. The team's credibility is one unverifiable claim away from being gone — and the numbers that are hardest to produce honestly are the ones a judge asks about first. When someone wants a better number than the data supports, the answer is no.
