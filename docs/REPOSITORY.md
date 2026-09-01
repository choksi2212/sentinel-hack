# TRINETRA — REPOSITORY OPERATIONS MANUAL

**Governs:** `https://github.com/choksi2212/sentinel-hack` — the **submission repository** · public · empty as of 2026-09-01, no default branch yet
**Version 1.0 · 2026-09-01 (D1) · 6 build days to submission**

> **Which repo am I in?** This document currently lives in `parthu-babyy` (the specification repo) but governs `sentinel-hack` (the code repo we submit). It moves to `sentinel-hack/docs/REPOSITORY.md` in the D1 bootstrap commit described in §3, at which point `parthu-babyy` becomes history — see §2.
>
> This document is not about *what* to build — that is in [`TRINETRA_Canonical_Contracts.md`](TRINETRA_Canonical_Contracts.md) and the two plans. It is about who touches what, which branch it lands on, and what must never be committed.
>
> **Read this before your first push.** Two of the rules below (contract changes, secrets) are cheap to follow and expensive to undo.

---

## 1. Why this document exists

Four people, six days, one repository, and a submission deadline on **7 September**. The failure modes we are guarding against are not exotic:

- Two people edit the same event schema in their own directory, both are internally consistent, and nothing works when they meet on D4.
- Someone commits `.env` with a Sentinel stream password to a **public** repo, and it is scraped within minutes.
- `main` breaks on D6 and nobody knows which of thirty commits did it.
- A 210 GB dataset or a 400 MB checkpoint enters git history and the repo becomes uncloneable on demo morning.
- Everyone waits on a code review that nobody has time to do, and the six days become four.

Each of those has a rule below. Nothing here is process for its own sake; if a rule does not prevent one of those outcomes, it is not in this document.

---

## 2. Repo identity and what moves here

`sentinel-hack` is the **submission repository**. Code and specification both live here, in one clone.

### 2.1 The docs move from `parthu-babyy`

Eight documents currently live in `choksi2212/parthu-babyy`: the canonical contracts, the two plans, the four per-person manuals, and this one. All eight move here on D1, as part of the bootstrap commit.

**Why they move rather than being linked:** the canonical contracts document is normative for the code in this repo. A judge, or a teammate, or an AI reading this repo must not have to find a differently-named repository to learn what an `EventEnvelope` is. A submission repo whose specification lives elsewhere is incomplete, and a link is one dead URL away from being useless.

After the move, `parthu-babyy` is **history**. Do not edit it. Add a line to its README pointing here.

### 2.2 Path changes during the move

The manuals sit at the root of `parthu-babyy`. Here they go under `docs/manuals/`, because a repo root that has to accommodate `ai/`, `backend/`, `frontend/`, `datasets/`, `config/`, and `scripts/` should not also carry four 25 KB markdown files.

That means the relative links inside `README.md` and inside each manual break on the move. Fixing them is a five-minute job and it is **part of the bootstrap commit**, not a follow-up:

| Link in | Was | Becomes |
|---|---|---|
`README.md` | `TRINETRA_Manas_...md` | `docs/manuals/TRINETRA_Manas_...md` |
`README.md` | `docs/TRINETRA_Canonical_Contracts.md` | unchanged |
Each manual | `docs/TRINETRA_Canonical_Contracts.md` | `../TRINETRA_Canonical_Contracts.md` |

Verify with a link check before pushing, not after.

---

## 3. D1 bootstrap — run once, in this order, by Manas

Nobody else pushes until step 8 is done. Two of the four lanes are blocked on the contracts landing, and a second person bootstrapping in parallel produces two root commits and a mess.

```bash
# 1. Fresh local repo — the remote has no default branch, so we create main
mkdir sentinel-hack && cd sentinel-hack
git init -b main
git remote add origin https://github.com/choksi2212/sentinel-hack.git

# 2. .gitignore FIRST, before any other file is added (see §12)
#    Adding it first means a stray .env or __pycache__ can never enter history.

# 3. Directory skeleton (see §4)
mkdir -p ai/{contracts,media,detect,track,plate,ocr,fusion,quality} \
         backend/app/{models,api,services,core} backend/alembic \
         frontend/src/{types,components,pages,hooks,lib} \
         datasets benchmarks/reports config scripts/synth \
         docs/manuals docs/demo tests/fixtures artifacts/snapshots \
         .github/workflows

# 4. Copy all 8 docs from the parthu-babyy clone, fixing paths per §2.2

# 5. Governance files: CODEOWNERS, PR template, this document
# 6. CI workflows (see §14)
# 7. Placeholder .gitkeep in every empty dir git would otherwise drop

git add -A
git commit -m "chore: bootstrap repository, specification docs, CI, ownership"
git push -u origin main

# 8. Announce in the group: "main is up, contracts branch next"
```

Then, still on D1 and still before anyone else pushes:

```bash
git switch -c contracts
# ai/contracts/frame.py, ai/contracts/stages.py, ai/contracts/event.py
# tests/fixtures/*.json  — all 12 fixtures
git push -u origin contracts
# PR → main, all three others ack, squash merge, delete branch
```

**The contracts branch is the gate for the whole project.** Mihir cannot write an ingest validator and Parth cannot write TypeScript types against a schema that does not exist in the repo. Get it merged D1 evening.

### 3.1 Repo settings to configure after the first push

Do these in the GitHub web UI once `main` exists. On a free public repo some protections are available and some are not — set what you can.

| Setting | Value | Why |
|---|---|---|
Default branch | `main` | |
Delete head branches on merge | **on** | Stops thirty stale branches by D6 |
Allow squash merge | **on** | §10 |
Allow merge commits | **off** | Keeps `main` linear and revertible |
Allow rebase merge | off | One strategy is enough |
Require status checks on `main` | **on** if available | A red CI merge on D5 costs an evening |
Require linear history | on if available | Makes §16 revert reliable |

Do **not** require pull-request reviews on `main` as a hard block. With four people and six days, a mandatory review on every merge produces a queue, and the queue produces direct-to-main commits at 1 am to get around it. §9 handles review by category instead.

---

## 4. Directory layout and owners

One owner per directory. The owner is who merges changes there and who is asked when something breaks.

```
sentinel-hack/
├── ai/                          MANAS
│   ├── contracts/               FrameEnvelope, EventEnvelope, stage dataclasses
│   ├── media/                   5 adapters: rtsp, hls, file, frames, synthetic
│   ├── detect/                  RF-DETR vehicle detection
│   ├── track/                   ByteTrack + TrackKey / session handling
│   ├── quality/                 vehicle gate + plate quality score
│   ├── plate/                   plate detection
│   ├── ocr/                     PaddleOCR + preprocessing variants
│   ├── fusion/                  temporal consensus, normalization, validation
│   └── worker.py                the per-camera worker loop
├── backend/                     MIHIR
│   ├── app/models/              SQLAlchemy models — the 8 tables
│   ├── app/api/                 /api/v1 routers
│   ├── app/services/            ingest, search, journey, watchlist, alerts
│   ├── app/core/                config, db session, redis
│   └── alembic/                 migrations — every schema change, no exceptions
├── frontend/                    PARTH
│   └── src/{types,components,pages,hooks,lib}
├── datasets/                    AKSHAT
│   ├── LICENSES.md              9 fields per asset, mandatory
│   └── manifests/               SHA-256 manifests
├── benchmarks/                  AKSHAT
│   └── reports/                 committed JSON reports
├── scripts/                     shared — see §5.1
│   └── synth/                   AKSHAT — synthetic corpus generator
├── config/                      MANAS
├── tests/fixtures/              MANAS — the 12 event fixtures
├── docs/                        MANAS
│   ├── TRINETRA_Canonical_Contracts.md      normative
│   ├── TRINETRA_Main_Plan_REFINED_v2.md
│   ├── TRINETRA_Technical_Implementation_Master_Plan_REFINED_v2.md
│   ├── REPOSITORY.md            this document
│   ├── manuals/                 four per-person manuals
│   └── demo/                    PARTH — demo script, projector checklist
├── artifacts/                   gitignored — snapshots, crops, weights
├── .github/                     MANAS
│   ├── workflows/               CI
│   ├── CODEOWNERS
│   └── pull_request_template.md
├── docker-compose.yml           MIHIR
├── .env.example                 split ownership — see §5.1
├── .gitignore                   MANAS
└── README.md                    MANAS
```

---

## 5. Ownership matrix — who does what

| Person | Role | Owns | Merges into `main` |
|---|---|---|---|
**Manas / Niklaus** | AI Lead & System Architect | `ai/**`, `config/**`, `tests/fixtures/**`, `docs/**`, `.github/**`, `.gitignore`, `README.md` | own lane + contract PRs after all-ack |
**Mihir** | Backend & Data Platform | `backend/**`, `docker-compose.yml`, `scripts/sync_cameras.py`, `scripts/seed_*.py` | own lane |
**Akshat** | CV Data & Benchmarking | `datasets/**`, `benchmarks/**`, `scripts/synth/**`, `scripts/check_split_leakage.py`, `scripts/freeze_manifest.py` | own lane |
**Parth** | Frontend & Demo | `frontend/**`, `docs/demo/**` | own lane |

Manas owns `docs/` and `.github/` because someone has to, and the architect is the person whose job already includes keeping the four lanes coherent. It is not a seniority claim — it is so that nobody has to ask who merges a CI fix.

### 5.1 Contested files — decided in advance

These are the files two people will reach for. Guessing at the time produces a conflict at the worst moment.

| File | Owner | Rule |
|---|---|---|
`.env.example` | **Mihir** | Mihir owns backend + DB + Redis keys. Parth appends only `VITE_*` keys and only in the marked block. Never any real value — placeholders only. |
`docker-compose.yml` | **Mihir** | Manas may add the `ai-worker` service; Parth may add the `frontend` service. Both via PR to Mihir, not direct. |
`config/*.yaml` | **Manas** | `offline.yaml` / `live.yaml` / `training.yaml`. Others request a key rather than adding one. |
`scripts/` | shared, per-file | Ownership is per file, per §5. A new script names its owner in a header comment on line 1. |
`requirements.txt` | **Mihir** | One Python environment for backend + AI. Manas requests AI dependencies via PR. Two requirements files means two environments and a lost afternoon. |
`frontend/package.json` | **Parth** | Sole owner. |
`README.md` | **Manas** | Anyone may PR a correction. |
`docs/TRINETRA_Canonical_Contracts.md` | **Manas**, but see §7 | The all-ack rule applies. |

### 5.2 Tie-break

Technical disagreement inside one lane: the lane owner decides.
Disagreement that crosses lanes, or touches a contract: **Manas decides**, as architect, and writes the reason into the doc so it is not re-argued on D5.

Six days is not enough time for consensus on everything. It is enough time for one person to decide quickly and be accountable for it.

---

## 6. Branch model

Deliberately small. GitFlow, release branches, and a `develop` branch are wrong for a six-day build — the overhead is real and the benefit assumes a release cadence we do not have.

```
main ────●────●────●────●────●────●────●───────●──── (always demoable)
          \    \        /    /             \
contracts  ●───●───────/    /               \
                          /                  \
ai/media-adapters   ●────●                    \
backend/ingest-endpoint  ●──●                  \
frontend/live-map          ●───●                ●  freeze/demo-sep09  → tag v1.0-demo
data/trinetra-hard              ●──●
```

### 6.1 The branches

| Branch | Lifetime | Rule |
|---|---|---|
**`main`** | permanent | **Always runnable, always demoable.** Direct commits only during D1 bootstrap (Manas) and the freeze (§15). |
**`contracts`** | D1 only | Merged before anything else, then deleted. Special-cased because everything depends on it. |
**Lane branches** | 1–2 days | Namespaced `ai/`, `backend/`, `frontend/`, `data/`, `docs/`, `ci/`. Squash-merged to `main`, then deleted. |
**`freeze/demo-sep09`** | Sep 9 → finals | Cut at the 24-hour freeze. Only cherry-picks. Tagged `v1.0-demo`. |

### 6.2 Naming

```
<lane>/<short-kebab-description>
```

Lanes: `ai` · `backend` · `frontend` · `data` · `docs` · `ci`

```
ai/media-adapters            backend/ingest-endpoint       frontend/live-map
ai/bytetrack-session         backend/journey-feasibility   frontend/journey-view
ai/temporal-fusion           backend/websocket-alerts      frontend/hls-preview
data/trinetra-hard           backend/hls-proxy             docs/repo-ops
data/synthetic-corpus        ci/frontend-typecheck
```

Lane, not person. `ai/temporal-fusion` stays correctly named when Akshat helps debug it; `manas/fusion` does not. It also makes `git branch -a` readable at a glance, which matters on D5 when there are fifteen of them.

Not acceptable: `test`, `fix`, `new`, `temp`, `manas2`, `final`, `final-final`.

### 6.3 Branch lifetime

**One to two days, maximum.** A branch open for four days is not a branch, it is a fork, and merging it on D5 means resolving four days of drift against three other people's changes during the window with no slack.

If a piece of work genuinely takes longer, split it and merge the parts. A half-finished media adapter behind a config flag on `main` is worth more than a complete one on a branch nobody has seen.

### 6.4 Do not create

| Not this | Because |
|---|---|
`develop` | A second integration branch means integrating twice |
`release/*` | We ship once; `freeze/demo-sep09` covers it |
Per-person long-lived branches | Becomes four forks; integration debt surfaces on D5 |
`main-backup`, `main-old` | Use tags. Branches named "backup" are never deleted |

---

## 7. Contract changes — the one rule to be slow about

A **contract change** is any edit to:

- `docs/TRINETRA_Canonical_Contracts.md`
- `ai/contracts/**`
- `backend/app/models/**` or `backend/alembic/**` (the DDL is a contract)
- `frontend/src/types/api.ts`
- any `EventEnvelope` field name, type, or enum value
- any `/api/v1` request or response shape

### 7.1 The protocol

1. **Never** edit a contract inside a feature branch alongside other work. A contract change gets its own branch and its own PR.
2. Branch `docs/contract-<what>` or `ai/contract-<what>`.
3. Change `docs/TRINETRA_Canonical_Contracts.md` **first**, in the same PR as the code.
4. Update every location that inlines it — the manuals carry `COPIED FROM CANONICAL — DO NOT EDIT HERE` blocks, and they are greppable:
   ```bash
   grep -rn "COPIED FROM CANONICAL" docs/
   ```
5. PR description states: what changed, which of the four lanes are affected, and whether it is additive or breaking.
6. **All three other people acknowledge** before merge. A 👍 on the PR is enough; a meeting is not required.
7. Announce the merge in the group chat with the one-line summary.

### 7.2 Why all-ack, when nothing else needs review

Because this specific failure is silent and expensive. If Mihir renames `first_seen_at` to `seen_at`, his tests pass, his lane is green, and nothing tells Parth until his build breaks — possibly not until a screen renders `undefined` during a rehearsal. Every other class of bug announces itself in the lane that caused it. Contract drift announces itself in someone else's lane, later, and usually under time pressure.

This already happened once in the specification documents: four manuals carried four incompatible event schemas, and it took a full audit to find. The all-ack rule costs about ten minutes per change and is the reason it will not happen in the code.

### 7.3 After D3: additive only

From D3 onward, `/api/v1` and the event schema are **additive-only**. A new optional field is fine. A renamed field, a removed field, a changed type, or a narrowed enum requires a new version and an explicit decision by Manas.

Parth's build breaking at 11 pm on 6 September because a field was renamed is a self-inflicted wound with no recovery time left.

---

## 8. Commit conventions

```
<type>(<scope>): <imperative summary under 72 chars>

<body: why, not what — the diff already says what>
```

**Types:** `feat` · `fix` · `refactor` · `test` · `docs` · `chore` · `perf` · `ci`
**Scopes:** `ai` · `media` · `detect` · `track` · `plate` · `ocr` · `fusion` · `backend` · `db` · `api` · `frontend` · `data` · `bench` · `contracts` · `repo`

```
feat(media): add VideoFileSource with PTS-driven sampling
fix(track): key tracker state on TrackKey, not (camera_id, track_id)
feat(db): add stream_sessions and uq_trackkey constraint
docs(contracts): bump EventEnvelope to 1.1, add plate_width_px
test(api): assert naive timestamps are rejected with 422
```

Not acceptable: `update`, `fix bug`, `changes`, `wip`, `asdf`, `final`.

Explain **why** in the body when the reason is not obvious from the diff. `fix(track): key on TrackKey` is a one-line diff whose reason — two vehicles merging across a reconnect and producing an impossible journey — is worth three sentences that save the next person an hour.

Commit freely on your own branch, including `wip` commits. Squash-merge means `main` only ever sees the clean summary, so nobody needs to police your local history.

---

## 9. Pull requests — required, or not, by category

Review is a real cost. Spend it where the failure mode is silent.

| Change | Branch + PR? | Approval needed |
|---|---|---|
Inside your own lane, no contract touched | Branch yes, PR yes | **None — self-merge on green CI** |
Touches another lane's directory | Yes | **That owner approves** |
**Contract change** (§7) | Yes | **All three others ack** |
Adds a dependency to `requirements.txt` | Yes | **Mihir approves** |
Edits `docker-compose.yml` | Yes | **Mihir approves** |
Edits `config/*.yaml` | Yes | **Manas approves** |
CI workflow | Yes | Manas approves |
Docs typo | Direct to `main` ok | None |
D1 bootstrap | Direct to `main` | Manas only |
During freeze (§15) | Cherry-pick only | **Two others agree it is demo-critical** |

Self-merge on your own lane is deliberate. With four people and six days, a mandatory review on every merge creates a queue, and a queue creates 1 am direct-to-main commits to bypass it. CI is the gate for own-lane work; humans are the gate for cross-lane and contracts.

Always open the PR anyway, even when self-merging. It gives the other three a place to see what landed without reading `main`'s log, and it gives CI something to run before the merge rather than after.

### 9.1 PR description

```markdown
## What
One or two sentences.

## Why
The reason, if not obvious.

## Contract impact
- [ ] No contract touched
- [ ] Additive only (new optional field)
- [ ] BREAKING — all-ack required, lanes affected: ___

## Verified
- [ ] CI green
- [ ] Ran locally
- [ ] Fixtures still pass (if backend/AI)

## Blocks / blocked by
```

The **Contract impact** section is the point of the template. It forces the author to think about the question before someone else has to discover the answer.

---

## 10. Merge and rebase policy

**Merging to `main`: squash.** One commit per feature. `main`'s history becomes a readable list of what landed, and §16's revert is a single `git revert`.

**Updating your branch from `main`: rebase.**

```bash
git switch main && git pull --ff-only
git switch ai/temporal-fusion
git rebase main
# resolve, then
git push --force-with-lease
```

`--force-with-lease`, never bare `--force`. It refuses the push if someone else has touched the branch, which is exactly the check you want.

**Rebase your own unmerged branch: yes.** **Rebase a branch someone else has checked out: no.** With lane-owned branches this is almost always safe; ask if you are unsure.

**Never force-push `main`.** Not once, not to fix a bad commit message, not at 2 am. Use `git revert` — it works even after everyone has pulled.

### 10.1 Pull before you start, every time

```bash
git switch main && git pull --ff-only
```

`--ff-only` is intentional. If it refuses, you have local commits on `main` that you did not intend to make, and you want to know now rather than discover them inside a merge commit.

### 10.2 Conflicts

Most conflicts here will be in `.env.example`, `requirements.txt`, or `docker-compose.yml` — all three are covered by §5.1 precisely to make conflicts rare.

If you hit one in a file you do not own: **stop and ask the owner.** Do not resolve it by guessing which side to keep. A wrongly-resolved conflict in a schema file is indistinguishable from a deliberate change, and it will be attributed to whoever merged it.

---

## 11. Daily rhythm

| When | What | Who |
|---|---|---|
**Morning** | `git switch main && git pull --ff-only`, then rebase your branch | everyone |
| | Read the group chat for merged contract changes | everyone |
**During the day** | Commit often on your branch; push at least once before lunch | everyone |
**Evening** | **Merge something to `main`.** Every day. | everyone |
| | Verify `main` still starts (§18) | last person to merge |
**Checkpoint** | Post one line: merged / blocked / needs from whom | everyone |

**Merging to `main` daily is the single most important habit in this document.** Not because `main` needs the code, but because it is the only thing that surfaces integration problems while there is still time to fix them. Four lanes that each work perfectly and meet for the first time on D5 is the standard way a hackathon project fails, and it fails with everyone having done their job.

### 11.1 Gate days

The plans define integration gates **G1–G7**. Three of them are repo events:

| Gate | Day | Means for the repo |
|---|---|---|
**G1** | D2 | Mihir's ingest accepts all 12 fixtures — fixtures are on `main` and CI runs them |
**G2/G3** | D4 | Manas's real event persists end to end — all three lanes merged and running together on `main` |
**G4/G5** | D4 | Alert reaches Parth's UI over WS — frontend on `main` talks to backend on `main` |

If a gate day arrives and the lanes have not merged, the gate cannot be evaluated. That is a repo problem, not an engineering one.

---

## 12. `.gitignore` — commit this first, before anything else

Adding this in the bootstrap commit, before any other file is staged, means a stray `.env` or checkpoint can never enter history. Removing a secret from history afterwards means a force-push and a rotated credential; preventing it costs nothing.

```gitignore
# ---- secrets ----
.env
.env.*
!.env.example
*.pem
*.key
credentials.json

# ---- python ----
__pycache__/
*.py[cod]
.venv/
venv/
env/
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# ---- node ----
node_modules/
dist/
.vite/
*.tsbuildinfo
npm-debug.log*

# ---- model weights & checkpoints ----
*.pt
*.pth
*.onnx
*.engine
*.pdparams
*.safetensors
weights/
checkpoints/

# ---- datasets & media ----
datasets/**/images/
datasets/**/labels/
datasets/raw/
datasets/**/*.zip
datasets/**/*.tar*
*.mp4
*.mkv
*.avi
*.ts
!tests/fixtures/**/*.json

# ---- runtime artifacts ----
artifacts/
snapshots/
logs/
*.log
benchmarks/reports/*.tmp.json

# ---- db dumps (except the demo snapshot, added with -f) ----
*.sql
*.dump

# ---- editors / os ----
.vscode/
.idea/
.DS_Store
Thumbs.db
*.swp
```

Note the four negations (`!.env.example`, `!tests/fixtures/**/*.json`). Fixtures are small, essential, and would otherwise be caught by the JSON and media patterns.

---

## 13. Never commit

**This repository is public.** Anything pushed is world-readable immediately and is scraped by credential harvesters within minutes. Deleting it later does not help — it stays in the history, in forks, and in caches.

| Never | Instead | If it happens anyway |
|---|---|---|
`.env` with real values | `.env.example` with placeholders | **Rotate the credential first**, then clean history. In that order. |
Sentinel stream passwords | Backend `.env`, gitignored; browser reaches streams via Mihir's HLS proxy | Same |
A credential in a `VITE_*` variable | Backend proxy — Vite inlines these into the public bundle | Rotate, remove the variable |
Model weights (`*.pt`, `*.pdparams`) | Shared drive + SHA-256 in `datasets/LICENSES.md` | `git rm --cached`, add to `.gitignore` |
Datasets (thundarstrom, CCPD, 210 GB thirdeyelabs) | Download script + frozen manifest | Same |
Video clips | Shared drive; document the SHA-256 | Same |
`node_modules/`, `__pycache__/`, `.venv/` | `.gitignore` | `git rm -r --cached` |
`artifacts/snapshots/` output | Gitignored; regenerated by running the pipeline | Same |
Commented-out code blocks | Delete it. Git remembers. | — |
A 400 MB checkpoint "just this once" | It is in history forever and everyone clones it | Requires history rewrite — expensive |

### 13.1 Large files

Nothing over **10 MB** without asking. No Git LFS unless we decide we need it — LFS on a free public repo has bandwidth limits that are exactly the kind of thing that fails on demo morning.

Model weights and datasets are referenced, not stored: a download script in `scripts/`, plus a SHA-256 in the manifest so we can prove which artifact produced which benchmark number.

### 13.2 The one intentional exception

The demo database snapshot. Small, compressed, and worth having in the repo so a clean clone can produce a populated dashboard without running the pipeline:

```bash
pg_dump -Fc trinetra > docs/demo/demo_snapshot.dump
git add -f docs/demo/demo_snapshot.dump     # -f overrides the *.dump ignore
```

Only Mihir adds this, only once, on D6, and only if it is under 10 MB.

---

## 14. CI

Four cheap jobs with path filters. The filters matter: a frontend PR must not wait on backend tests when there are six days on the clock.

```yaml
# .github/workflows/ci.yml
name: CI
on:
  push: { branches: [main] }
  pull_request:

jobs:
  contracts:
    # runs on EVERY PR — this is the cross-lane safety net
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: pytest tests/test_contracts.py -q     # 12 fixtures vs schemas

  backend:
    if: contains(github.event.pull_request.changed_files, 'backend/') || github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgis/postgis:16-3.4              # pinned, same as compose
        env: { POSTGRES_PASSWORD: postgres }
        options: >-
          --health-cmd pg_isready --health-interval 10s
          --health-timeout 5s --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: alembic upgrade head                  # from an EMPTY database
      - run: pytest backend/tests -q

  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20', cache: npm, cache-dependency-path: frontend/package-lock.json }
      - run: npm ci --prefix frontend
      - run: npm run --prefix frontend typecheck   # tsc --noEmit
      - run: npm run --prefix frontend build       # dev server tolerates what build rejects

  data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: python scripts/check_split_leakage.py
      - run: python scripts/check_licenses.py      # every asset has 9 fields
```

### 14.1 Why these four

| Job | Catches |
|---|---|
`contracts` | The silent cross-lane failure. Runs on everything, no filter. |
`backend` | A migration that only works on Mihir's already-populated database. `alembic upgrade head` from empty is the real test. |
`frontend` | Type errors the Vite dev server tolerates and `vite build` rejects. Discovering these on D6 evening is avoidable. |
`data` | Test-set contamination and unlicensed assets — both invisible until someone asks, and both unfixable late. |

No linting job. Not because style does not matter, but because a red CI on a formatting nit at 11 pm on D6 will get CI disabled, and then the four jobs above stop running too. Run formatters locally.

### 14.2 Red CI on `main`

Fix it or revert it, **within the hour**. A red `main` that stays red for a day trains everyone to ignore CI, and after that the four jobs above are decoration.

---

## 15. Tags, freeze, and the demo branch

| Tag | When | What |
|---|---|---|
`v0.1-submission` | **Sep 7**, at submit | Exact state submitted. Never moved. |
`v1.0-demo` | **Sep 9**, at freeze | What runs at the finals |

```bash
git tag -a v0.1-submission -m "Submitted to Sentinel, 7 Sep 2026"
git push origin v0.1-submission
```

Tag the submission even if you plan to keep working. If a question arises about what was submitted, the answer should be a git ref rather than a memory.

### 15.1 The 24-hour freeze — from Sep 9

```bash
git switch -c freeze/demo-sep09 main
git push -u origin freeze/demo-sep09
git tag -a v1.0-demo -m "Demo build, Gujarat Sentinel finals"
git push origin v1.0-demo
```

From the cut, on `freeze/demo-sep09`:

| Allowed | Not allowed |
|---|---|
Crash fix on a demo path | New features |
Wrong-label fix | Refactoring |
Demo data correction | Dependency upgrades |
Adding a missing env var | Schema changes |
| | "Quick" improvements |

Every change: cherry-pick, **two other people agree it is demo-critical**, and the full demo runs once afterwards.

`main` stays open for post-hackathon work. The demo runs from the frozen branch. The reason for the freeze is not that the code is finished — it is that a change on demo morning has no time to be tested, and an untested change is more likely to break the demo than the bug it fixes.

---

## 16. When `main` breaks

Squash merges make this straightforward — one commit per feature, so one revert.

```bash
git log --oneline -10                  # find the offending squash commit
git revert <sha>                       # creates a revert commit
git push origin main
```

Then fix it on a branch and re-merge. Do not fix forward on `main` while it is broken — a second person pulling a broken `main` loses time to a problem that is already known.

**Never** `git reset --hard` + force-push on `main`. Three other people have already pulled it; a rewritten `main` gives them a divergence they did not cause and cannot easily resolve.

### 16.1 Demo-day emergency

1. `git switch freeze/demo-sep09`
2. Fix it there, minimally
3. Two others confirm it is demo-critical
4. Run the full 8-minute demo once
5. Commit with `fix(demo):` and push

If a fix cannot be verified by a full demo run, do not ship it. Parth's recovery drills exist so that a known broken thing can be worked around live — a scripted workaround is safer than an unverified fix.

---

## 17. Onboarding — clone to running

Put this in `README.md` and keep it working. A teammate reinstalling on D5, or a judge cloning the repo, should reach a running system without asking anyone.

```bash
git clone https://github.com/choksi2212/sentinel-hack.git
cd sentinel-hack
cp .env.example .env                   # fill in local values

docker compose up -d postgres redis
pip install -r requirements.txt
alembic upgrade head
python scripts/sync_cameras.py --input data/cameras.json

uvicorn backend.app.main:app --reload
curl http://localhost:8000/health/ready          # must be 200 before continuing

python -m ai.worker --config config/offline.yaml --camera cam04
npm ci --prefix frontend && npm run --prefix frontend dev
```

Order matters: the worker before migrations produces foreign-key errors that read like a bug in the AI code. `/health/ready` is the gate for the last two steps.

---

## 18. Definition of done — repo hygiene

Check on D6, before the submission tag.

- [ ] `.gitignore` was the first file in history; no `.env`, weights, datasets, or `node_modules` in any commit
- [ ] `git log --all --oneline | grep -iE 'wip|asdf|final'` on `main` returns nothing
- [ ] No branch older than 2 days still open
- [ ] Merged branches deleted
- [ ] Every merged PR states its contract impact
- [ ] `docs/TRINETRA_Canonical_Contracts.md` matches the code in all four lanes
- [ ] Every `COPIED FROM CANONICAL` block matches the canonical text (§7 step 4)
- [ ] CI green on `main`; all four jobs ran within the last 24 h
- [ ] `alembic upgrade head` verified from an **empty** database
- [ ] `README.md` clone-to-running verified on a machine that is not the author's
- [ ] `CODEOWNERS` matches §5
- [ ] All 8 docs present under `docs/` (contracts, 2 plans, this manual, 4 under `manuals/`), every internal link resolving
- [ ] `parthu-babyy` README points here
- [ ] `v0.1-submission` tagged and pushed
- [ ] `freeze/demo-sep09` cut, `v1.0-demo` tagged

---

## 19. Anti-patterns

| Do not | Consequence |
|---|---|
Wait to merge until your lane is "done" | Integration debt lands on D5 with no slack |
Keep a branch open four days | It is a fork; merging costs a day |
Edit a contract inside a feature PR | Silent cross-lane break, found late |
Commit `.env` to a **public** repo | Credential scraped in minutes; rotation required |
Put a secret in a `VITE_*` variable | Compiled into the public JS bundle |
Force-push `main` | Three people get a divergence they cannot resolve |
`git reset --hard` on shared history | Same, worse |
Commit a checkpoint "just this once" | In history forever; everyone clones it |
Resolve a conflict in a file you do not own | Wrong resolution is indistinguishable from intent |
Leave `main` red overnight | Everyone learns to ignore CI |
Add a second `requirements.txt` | Two environments, one lost afternoon |
Rename an API field after D3 | Frontend breaks with no recovery time |
Add a feature during the freeze | Untested change on demo morning |
Skip Alembic and alter tables by hand | Nobody can reproduce your database |
Branch named `final` / `final-final` | You will not know which is which |
Mandatory review on every merge | Queue forms, then 1 am direct-to-main to bypass it |

---

## 20. The short version

Nine rules. If you remember only these:

1. **`main` is always demoable.** Merge to it daily.
2. **Branch per task, lane-namespaced, 1–2 days maximum.**
3. **Contracts change in their own PR, with all three others acknowledging.**
4. **Squash to `main`, rebase your own branch, never force-push `main`.**
5. **Nothing secret, nothing large, nothing generated goes in git.** The repo is public.
6. **Own-lane self-merge on green CI. Cross-lane needs the owner.**
7. **Alembic for every schema change.** No exceptions.
8. **Additive-only API after D3.**
9. **Freeze on Sep 9.** Cherry-picks only, two-person agreement, full demo run after each one.
