# MONDAY — swap the stub for the real pipeline

Read this instead of the code. Two lines change in `benchmarks/run.py`. That's it.

## The two lines

`benchmarks/run.py` line 20:
```python
from benchmarks import stub_predictor
```
becomes
```python
from ai import real_predictor  # or wherever your module actually lives
```

`benchmarks/run.py` line 123:
```python
predictions = {r["obs_id"]: stub_predictor.predict(r, fusion_enabled) for r in rows}
```
becomes
```python
predictions = {r["obs_id"]: real_predictor.predict(r, fusion_enabled) for r in rows}
```

Nothing else in `run.py`, `scorer.py`, or `delta.py` needs to change.

## The exact function signature `run.py` calls

```python
def predict(row: dict, fusion_enabled: bool) -> str | None:
    ...
```

- **Called once per row** in `datasets/trinetra-hard/index.jsonl` — every row, not just
  eligible ones. Your function decides what to do with ineligible/synthetic/real rows;
  the scorer sorts out what counts afterward.
- **`row`** is one full JSONL row (see `datasets/trinetra-hard/schema.json`). The fields
  your model actually needs: `frame_path` (where the image/crop lives — for
  `synthetic_plates` rows this has a `#frameN` suffix, for `indian_road` a
  `#clip_frame.jpg` suffix inside a tar shard — you may need to add a loader for that,
  it doesn't exist yet), `plate_bbox` (`[x, y, w, h]` in pixels), `camera_id` /
  `stream_session_id` / `track_id` (the `TrackKey` — use this to group frames for
  fusion, never `obs_id` order), `source_pts_ms` (align on this, never `observed_at`
  — there is no `observed_at` field, which is deliberate).
- **`fusion_enabled`**: `True`/`False`. If your model has one fusion path, branch on
  this. If fusion is a separate stage entirely, this flag should just turn it on/off.
- **Return value**: a plain string (the predicted plate text) or `None` (no
  prediction / model declined). That's the whole contract. Don't return a dict, don't
  return a confidence score here — `scorer.py` only ever calls `.get(obs_id)` and
  compares the string. If you need to log confidence/latency/etc., do it inside your
  own module and put summary numbers in the report's `diagnostics` block (see below).

## What the scorer does with your string (you don't need to replicate this)

`normalize()` uppercases and strips spaces/hyphens before comparing — don't
pre-normalize in your predictor, it's harmless if you do but redundant.
`eligible: false` rows: any non-`None` string you return counts as a fabrication,
regardless of what the string says. Return `None` for those unless you're
deliberately testing the fabrication path.

## Checklist if predictions don't align to ground-truth rows

Run this before trusting any number:

```bash
py -3.11 -c "
import json
from ai import real_predictor
rows = [json.loads(l) for l in open('datasets/trinetra-hard/index.jsonl')]
preds = {r['obs_id']: real_predictor.predict(r, True) for r in rows}
print('rows:', len(rows), 'predictions:', len(preds))
print('non-null predictions:', sum(1 for v in preds.values() if v is not None))
print('sample:', list(preds.items())[:3])
"
```

1. **`len(preds) != len(rows)`** — your dict comprehension or `obs_id` key is wrong.
   It shouldn't be possible (one entry per row, always), so if this fails the bug is
   upstream of `predict()`, not in it.
2. **`non-null predictions == 0`** — your predictor is silently failing (wrong path
   resolution for `frame_path`, model not loaded, exception swallowed somewhere).
   `run.py` will NOT crash on this — that's by design (a missing prediction is a
   miss) — which means a completely broken predictor produces a quiet, plausible-
   looking `rate: 0.0` report instead of an error. **A rate of exactly 0.0 or 1.0
   across every bucket is a red flag, not a result.** Check `non-null predictions`
   before you believe either.
3. **Rate looks too good, too fast** — check you're not accidentally reading
   `plate_text` off the `row` itself inside `predict()` (easy copy-paste mistake from
   the stub, which does exactly that on purpose). Grep your predictor file for
   `row["plate_text"]` or `row.get("plate_text")` — it should not appear anywhere.
4. **`frame_path` resolution** — `synthetic_plates` rows point at a real PNG on disk
   plus a `#frameN` suffix describing which synthetic degradation step to apply (see
   `scripts/synth/build_sequences.py` `apply_condition()` / `degradation_params` on
   the row if you want to reproduce the exact frame instead of re-deriving it).
   `indian_road` rows point inside a `.tar` shard (`path.tar#clipid_0042.jpg`) — you
   need a small loader (`tarfile.open(...).extractfile(name)`), there isn't one yet
   because indian_road rows are diagnostic-only and never scored. If you extend
   real-model inference to indian_road, write that loader — don't reuse
   `stub_predictor`'s (nonexistent) one.
5. **`by_plate_width` buckets all-null** — means `n: 0` for every bucket, which means
   `rows` came back empty, which means `--dataset` doesn't match a directory under
   `datasets/` or `index.jsonl` doesn't exist there. Check the path, not the model.
6. **Sanity check against the fixtures, not just real data**: `py -3.11
   benchmarks/scorer.py` still has to print "all six fixtures pass" after any change
   near the scoring path. If it doesn't, stop — nothing downstream is trustworthy
   until it does (this is the harness's own gate, not a suggestion).

## Then run it for real

```bash
py -3.11 -m benchmarks.run --suite e2e --dataset trinetra-hard --fusion off --out benchmarks/reports/
py -3.11 -m benchmarks.run --suite e2e --dataset trinetra-hard --fusion on  --out benchmarks/reports/
py -3.11 -m benchmarks.delta --before benchmarks/reports/e2e_fusion_off_XXX.json \
    --after benchmarks/reports/e2e_fusion_on_XXX.json --out benchmarks/reports/FUSION_DELTA.md
```

Or just `bash scripts/run_all.sh` once `--predictor` supports your module (currently
hardcoded to `stub` in `run.py`'s `--predictor` choices — add your name there, or
just remove the `choices=` restriction if you don't need to switch back and forth).

The `FUSION_DELTA.md`/`STABILITY.md` first line names the predictor used
(`report["predictor"]`) — once it says your model's name instead of `stub`, the
table is a real result. Until then, it's a rehearsal.
