# SPEC — TRINETRA-HARD

The frozen evaluation set. Owner: Akshat. Consumed by: Manas (model selection),
and by every accuracy claim in the submission.

---

## 1. Purpose

One question: **on hard real-world conditions, how often does the system emit
the correct plate string for a vehicle that actually had a readable plate?**

TRINETRA-HARD answers "how good is it". The regression set (§7) answers "did I
just break something". They are different sets with different jobs.

## 2. Size and composition

~300 observations. Reduced from the original 1,000 because 300 human-verified
observations are worth more than 1,000 machine-labeled ones, and the labeling
budget is one morning.

| Slice | Count | Definition |
|---|---|---|
| `easy` | 60 | Daylight, frontal-ish, plate width > 80 px, no motion blur |
| `motion_blur` | 60 | Visible directional blur on the plate region |
| `night` | 60 | Low light or IR; headlight bloom permitted |
| `glare` | 45 | Direct sun, headlight wash, or specular reflection on the plate |
| `perspective` | 45 | Plate yaw or pitch beyond ~25°, or heavy skew |
| `tiny` | 30 | Plate width < 40 px regardless of other conditions |
| **Total** | **300** | |

Slices are assigned by dominant condition. An observation that is both night
and blurred goes in whichever condition a human judges dominant — recorded in
`slice_reason` so the call is auditable.

**Counts are targets, not guarantees.** If a slice cannot be filled from
available data, report the actual count and say so. A `glare` slice with 19
observations is a real result reported honestly; padding it to 45 with
near-duplicates is not.

## 3. Source discipline

- Source groups **disjoint** from all training and validation data.
- `indian_road` (thirdeyelabs) is the only source with temporal continuity and
  is therefore the **only** source usable for the fusion before/after
  measurement. Everything else is single-frame.
- **Split by clip ID, never by frame.** A clip contributing frames to training
  contributes nothing to TRINETRA-HARD, and vice versa.
- FANVID is **evaluation-only** by license. It may appear here; it may never
  appear in training.

## 4. Row schema

One JSONL row per observation, at `datasets/trinetra-hard/index.jsonl`:

```json
{
  "obs_id": "th_0001",
  "slice": "night",
  "slice_reason": "IR illumination, plate legible, no blur",
  "source_dataset": "indian_road",
  "clip_id": "clip_0042",
  "frame_path": "datasets/raw/indian_road/.../frame_00871.jpg",
  "source_pts_ms": 29033,
  "camera_id": "th_cam_01",
  "stream_session_id": "th_sess_01",
  "track_id": 17,
  "plate_bbox": [412, 288, 47, 19],
  "plate_width_px": 47,
  "width_bucket": "40-60",
  "eligible": true,
  "plate_text": "GJ01AB1234",
  "label_source": "human",
  "label_confidence": "certain"
}
```

Field notes:

- **`eligible`** — `true` only if a human can read the plate from the frame.
  An ineligible observation is still kept: it is how fabrication is measured.
  A system that emits a confident string for an unreadable plate is worse than
  one that emits nothing, and that only shows up if ineligible rows are present.
- **`label_source`** — `human` | `ocr_candidate`. Only `human` rows count
  toward the headline. `ocr_candidate` rows are unverified and must be excluded
  from any reported number until promoted.
- **`label_confidence`** — `certain` | `probable`. `probable` rows are reported
  separately and never folded into the headline.
- **`width_bucket`** — derived from `plate_width_px`, never hand-entered.
- **`TrackKey`** is `(camera_id, stream_session_id, track_id)`. For single-frame
  sources, synthesise a stable session and track id per image so the key shape
  is uniform across the set.

## 5. Freezing

```
datasets/trinetra-hard/
├── index.jsonl
├── MANIFEST.sha256      # hash of every referenced frame + hash of index.jsonl
└── FREEZE.md            # date, commit, counts per slice, known gaps
```

Once frozen, `index.jsonl` does not change. Every benchmark report cites
`dataset_manifest_sha256`. A report that cannot name the manifest it ran
against is not a result.

Rebalance: **at most once**, before final freeze, documented in `FREEZE.md`.

## 6. What makes this set honest

The failure mode is not technical. Every shortcut available here produces a
better-looking number and a less true one:

- filling a thin slice with near-duplicates
- dropping observations the model happens to fail
- labeling from OCR output and calling it ground truth
- reporting an average that hides the sub-40 px collapse
- silently excluding ineligible rows so fabrication never appears

A judge will probe the sub-40 px number first, because that is where CCTV
actually lives. Report it.

## 7. Regression set — separate, small, fast

`datasets/regression/`, ~40 observations across the same six slices. Runs in
under two minutes. Purpose is change detection, not measurement. May overlap
TRINETRA-HARD; may be tuned against, because nothing is ever reported from it.
