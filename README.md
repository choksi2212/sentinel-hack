# TRINETRA

Federated multi-camera vehicle intelligence for the Gujarat Police *Sentinel*
grid. TRINETRA turns up to thirty live CCTV streams into a queryable record of
which vehicle was seen where and when — one structured event per sighting,
delivered to a backend that stitches sightings into journeys and serves them to
an operations dashboard.

Built for a six-day hackathon by a four-person team; see
[docs/REPOSITORY.md](docs/REPOSITORY.md) for who owns what and how changes land.

---

## What it does

A vehicle crosses `cam04`. The worker decodes the frame, detects and tracks the
vehicle, reads its plate across several frames, scores how much to trust that
read, and emits a single **`EventEnvelope`** — camera, timestamp, track, plate
(or `null` when it is genuinely unreadable) and a cropped snapshot — to the
backend's ingest endpoint. The backend resolves identities, persists the
sighting, and pushes it live to the dashboard.

Nothing below the media layer knows whether a frame came from a live RTSP grid,
a recorded clip, a frame directory or the synthetic generator. That
**source-independence** is the invariant the whole system is built around: what
runs against synthetic ground truth on a laptop is the same pipeline that runs
against the live grid on demo day.

A plate of `null` is a correct, honest answer. Inventing a plate that was never
readable is the worst thing the system can do, and much of the design exists to
make that failure hard to reach.

---

## Architecture

Four lanes meet at a small set of contracts. The contract is the API between
people, not just between processes.

| Lane | Owner | Produces | Consumes |
|---|---|---|---|
| **AI worker** | Manas / Niklaus | `FrameEnvelope`, `EventEnvelope` | live grid / clips / synthetic |
| **Backend & data** | Mihir | REST + WebSocket, Postgres/PostGIS | `EventEnvelope` |
| **Frontend & demo** | Parth | operations dashboard | REST + WebSocket |
| **CV data & benchmarking** | Akshat | datasets, benchmark reports | dataset/manifest contracts |

The data path across the lanes:

```
media source ─▶ FrameEnvelope ─▶ 14-stage pipeline ─▶ EventEnvelope
                (the source                            (one per sighting)
                 boundary)                                     │
                                                     HTTP ingest ▼
                                              Postgres/PostGIS ─▶ REST/WS ─▶ dashboard
```

Every structure that crosses a person boundary — the two envelopes, the identity
model, the REST/WS surface — is defined **once** in
**[docs/TRINETRA_Canonical_Contracts.md](docs/TRINETRA_Canonical_Contracts.md)**.
That document is normative: if it and anything else (including this README)
disagree, it wins. This file is a front door, not a specification — it does not
restate a single contract, so it cannot drift from one.

---

## Repository layout

Lanes land incrementally. Today the repository carries the AI lane, the shared
specifications, and the governance files; the other lanes populate their
directories as they come online.

| Path | Owner | What | Status |
|---|---|---|---|
| `ai/` | Manas | media adapters + the 14-stage worker | present |
| `config/` | Manas | run configs (`offline` / `live` / `benchmark` / …) | present |
| `tests/` | Manas | worker suite, contract fixtures, the repo-wide guard | present |
| `docs/` | Manas | contracts, plans, ops manual, per-person manuals | present |
| `.github/` | Manas | CI, `CODEOWNERS`, PR template | present |
| `scripts/` | shared, per-file | operational scripts (owner named in each header) | partial |
| `backend/` | Mihir | REST/WS API, DB models, migrations | to land |
| `frontend/` | Parth | dashboard (Vite) | to land |
| `datasets/`, `benchmarks/` | Akshat | training data, accuracy reports | to land |

`artifacts/`, `datasets/` and model weights are gitignored — a large blob in git
history is in every clone forever.

---

## Quick start

The offline config runs on a fresh clone with **core dependencies only** — no
live grid, no backend, no GPU, no model weights:

```bash
pip install numpy PyYAML requests Pillow scipy
python -m ai.worker --config config/offline.yaml --camera cam04
```

Oracle and classical stages read the synthetic generator's own truth, so any
miscount is the pipeline's fault, not a model's — this proves the pipeline is
*correct*. Model accuracy is a separate question (`config/benchmark.yaml`). The
run guide, config matrix and full dependency tiers are in
**[ai/README.md](ai/README.md)**.

Run the tests (they need no model backend — model stages are exercised through
scripted fakes and oracles):

```bash
python -m pytest
```

---

## Documentation

| Document | What it is |
|---|---|
| [docs/TRINETRA_Canonical_Contracts.md](docs/TRINETRA_Canonical_Contracts.md) | **Normative.** Every cross-lane data structure. Read first. |
| [docs/REPOSITORY.md](docs/REPOSITORY.md) | Ownership, branch model, merge policy, CI, secret rules |
| [docs/TRINETRA_Main_Plan_REFINED_v2.md](docs/TRINETRA_Main_Plan_REFINED_v2.md) | Product and build plan |
| [docs/TRINETRA_Technical_Implementation_Master_Plan_REFINED_v2.md](docs/TRINETRA_Technical_Implementation_Master_Plan_REFINED_v2.md) | Technical implementation plan |
| [ai/README.md](ai/README.md) | The AI worker: run it, configure it, its dependency contract |

Per-person execution manuals:

- [AI lead — Manas / Niklaus](docs/manuals/TRINETRA_Manas_Niklaus_AI_Lead_Execution_Manual.md)
- [Backend & data platform — Mihir](docs/manuals/TRINETRA_Mihir_Backend_Data_Platform_Execution_Manual.md)
- [CV data & engineering — Akshat](docs/manuals/TRINETRA_Akshat_Computer_Vision_Data_Engineering_Execution_Manual.md)
- [Frontend & demo — Parth](docs/manuals/TRINETRA_Parth_Frontend_Dashboard_Demo_Execution_Manual.md)

---

## Ground rules

These are the cheap-to-follow, expensive-to-undo rules. The full set is in
[docs/REPOSITORY.md](docs/REPOSITORY.md).

- **Secrets live in the environment, never in the repo.** Copy `.env.example`
  to `.env` (gitignored) and fill it in. No stream password, database password
  or access token is ever committed. CI runs a repo-wide guard that fails the
  build on a leaked token or a committed credential.
- **Contracts change in one place.** Never edit a schema inside your own lane's
  copy; change it in the canonical contracts, in one commit, announced to every
  owner. Breaking changes bump `schema_version`.
- **Squash-merge to `main`; rebase your branch onto `main`.** One readable
  commit per feature, so a bad landing is one `git revert` away.
- **The credential a machine reads is set on that machine.** The worker takes
  the ingest URL as an origin only and appends the contract path itself, so a
  per-host setting can never quietly deliver events to the wrong place.
