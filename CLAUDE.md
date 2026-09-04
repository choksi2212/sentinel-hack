# TRINETRA — Akshat's lane (CV data & benchmarking)

Read this file completely before any action. It is the operating contract.

**Repo:** `github.com/choksi2212/sentinel-hack`
**Branch:** `data/trinetra-hard` (branched from `origin/manas`)
**Submission:** Monday 7 September 2026
**Machine:** Windows 10, RTX 4060 8 GB, Git Bash, `py -3.11`

---

## 1. What this lane owns

| Path | Status |
|---|---|
| `datasets/` | owned — manifests, licenses, TRINETRA-HARD index |
| `benchmarks/` | owned — harness, scorer, reports |
| `scripts/synth/` | owned — synthetic corpus tooling |
| `docs/manuals/akshat/` | owned |

## 2. What this lane must never touch

`ai/` · `config/` · `tests/` · `backend/` · `frontend/` · `.github/`

These have other owners (Manas, Mihir, Parth). Reading them is fine and often
necessary — `ai/contracts/` defines the schemas this lane measures against.
Writing to them is not. If a task appears to require editing one of these,
**stop and report** rather than editing.

## 3. Git rules — absolute

Permitted: `status`, `diff`, `log`, `add`, `commit`, `push` (own branch only),
`fetch`, `branch`.

**Forbidden without exception:**
- `push --force` or `--force-with-lease`, on any branch
- any push to `main` or `manas`
- `merge`, `rebase`, or `cherry-pick` involving `main` or `manas`
- `reset --hard`, `clean -fdx`, `checkout .` on a dirty tree
- `rm` / `rm -r` on anything under `datasets/raw/`

Commit format (repo convention):

```
<type>(<scope>): <imperative summary under 72 chars>
```

Types: `feat` `fix` `refactor` `test` `docs` `chore` `perf` `ci`
Scopes for this lane: `data` `bench` `synth` `docs`

Commit after every completed phase. Push after every commit. Small commits —
if the session dies, everything up to the last push survives.

## 4. Data layout — do not reorganise

```
sentinel-hack/datasets/
├── raw/            ← JUNCTION to A:\projects\...\datasets\raw  (5.6 GB, read-only)
├── manifests/      ← SHA-256 manifests, tracked in git
├── trinetra-hard/  ← the frozen benchmark index, tracked
├── synthetic/      ← synthetic corpus index, tracked
└── LICENSES.md     ← one row per asset, 9 fields, mandatory
```

`datasets/raw/` is **read-only**. Never move, rename, delete, convert in place,
or write into it. All derived artifacts go elsewhere. It is gitignored and must
stay that way — it is 5.6 GB and will not survive a commit.

## 5. Measurement facts — non-negotiable

These are contract-level. Do not reinterpret them.

- **Primary metric:** E2E correct-plate event rate. Everything else
  (mAP, CER, FPS, latency, VRAM) is a diagnostic used to *explain* the primary
  number, never reported as the headline.
- **Event identity:** `TrackKey = (camera_id, stream_session_id, track_id)`.
  Alignment is on `source_pts_ms`. **Never** `observed_at`.
- **Every rate carries its raw sample count.** `0.72` alone is not a result;
  `0.72 (43/60)` is.
- **No accuracy number is ever a single average.** Always broken out by plate
  width bucket: `>100`, `80-100`, `60-80`, `40-60`, `30-40`, `<30` px.
- **Fabrication is counted separately.** A plate string emitted where
  `eligible: false` is a fabrication, reported as its own count in every run,
  never folded into the error rate.
- **`indian_road` splits by CLIP ID, never by frame.** Same-clip frames are
  near-duplicates; perceptual hashing will not catch them. A frame-level split
  produces silent leakage and an inflated number.
- **TRINETRA-HARD is never trained on and never tuned on.** Rebalanced at most
  once, before final freeze.
- **No asset is used without a `LICENSES.md` row.** No row → the asset is
  excluded, not "used pending verification". This is a government submission.

## 6. Headline deliverable

**Fusion before/after delta, broken out by width bucket, with raw counts.**

That single table is what the submission's accuracy claim rests on. Everything
else in this lane exists to make it credible.

## 7. Reporting

Append a dated entry to `docs/manuals/akshat/RUNLOG.md` after every phase:
what was attempted, what landed, what failed, what is now blocked. The operator
is reading this from a phone — lead with status, keep it under ten lines.

## 8. When to stop

Stop and report rather than proceeding if:

- a task requires writing outside the owned paths
- a task requires a forbidden git operation
- `datasets/raw/` is missing or the junction is broken
- a dataset has no verifiable license
- clip identity cannot be recovered from `indian_road`
- any step would delete or overwrite unrecoverable data

A blocked task reported clearly is a good outcome. A blocked task worked around
by guessing is not.
