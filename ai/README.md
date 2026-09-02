# TRINETRA AI worker

The boundary between messy reality and clean data. Five media adapters collapse
into one `FrameEnvelope`; a fourteen-stage pipeline turns that into one
`EventEnvelope` per vehicle sighting and POSTs it to the backend's ingest
endpoint. No stage below the media layer knows whether its frames came from a
live grid, a recorded file, a frame directory or the synthetic generator — that
source-independence is the invariant the whole package is built around.

This file is two things: the run guide for the worker, and **the dependency
contract for the backend lane.** The pinned `requirements.txt` and the container
image are owned there; the list below is the source of truth for what those must
contain. Nothing here reads a credential from a config file or a flag — see
[`.env.example`](../.env.example).

For the full system, read [`docs/TRINETRA_Canonical_Contracts.md`](../docs/TRINETRA_Canonical_Contracts.md);
it outranks this file on any conflict.

---

## Run it

The offline config runs on a fresh clone with only the core dependencies — no
live grid, no backend, no model weights, no GPU:

```bash
python -m ai.worker --config config/offline.yaml --camera cam04
```

That proves the pipeline is *correct* (oracle stages read the synthetic
generator's own truth, so any miscount is the pipeline's fault, not a model's).
It does not prove the models are *good* — `config/benchmark.yaml` is for that.

| Config | Source | Stages | Sink | For |
|---|---|---|---|---|
| `offline.yaml` | synthetic | oracle / classical | `file` | correctness on a bare clone |
| `benchmark.yaml` | recorded (`file`) | real models | `null` | model accuracy, reproducible — no delivery |
| `live.yaml` | live RTSP | real models | `http` | the demo |
| `training.yaml` | — | — | — | fine-tune go/no-go record (`kind: training`, not runnable yet) |
| `base.yaml` | — | — | `file` | shared defaults the others extend |

Validate a config without running:

```bash
python scripts/validate_config.py config/live.yaml
```

---

## Dependencies, in tiers

Everything heavy is imported lazily, inside the stage that needs it. So the
package **imports and the oracle/classical path runs on `numpy` alone**; a
missing model backend is a clear error from that one stage, never an import
failure that takes down the worker.

### Core — always required
| Package (pip) | Imports as | Used by |
|---|---|---|
| `numpy` | `numpy` | every stage; the only hard third-party import |
| `PyYAML` | `yaml` | `ai/config.py` — loading any config file |
| `requests` | `requests` | `ai/emit/http_sink.py` — only when the sink is HTTP |

`.env` is parsed by the loader in `ai/config.py`; **`python-dotenv` is
deliberately not a dependency** — do not add it.

### Real media decode — for anything but the synthetic source
| Package (pip) | Imports as | Used by |
|---|---|---|
| `opencv-python-headless` | `cv2` | `ai/media/{file_source,frames_source,live_base}.py` |
| `Pillow` | `PIL` | frame I/O, snapshot encoding |
| **ffmpeg** (system binary) | — | RTSP-over-TCP / HLS / container decode behind OpenCV |

### Tracking
| Package (pip) | Imports as | Used by |
|---|---|---|
| `scipy` | `scipy` | `ai/track/{assignment,kalman}.py` — Hungarian assignment + Kalman motion |

### Real model backends — only the ones you enable
| Stage | Real backend | Needs |
|---|---|---|
| detect | `rfdetr` | `onnxruntime` (or `onnxruntime-gpu`); optional `torch` + the `rfdetr` pip package to load a fine-tuned `.pth` |
| plate | `rtdetr` | `torch` **+ `transformers>=4.49`** (see below) |
| ocr | `paddle` | `paddleocr` + `paddlepaddle` (or `paddlepaddle-gpu`) |
| — hub — | | `huggingface_hub` pulls checkpoints; token from `HF_TOKEN` |

`torch` is also used by `ai/metrics.py` purely for optional GPU telemetry, all
guarded by `torch.cuda.is_available()` — absent GPU or absent torch just leaves
those fields empty.

**`transformers>=4.49` is a hard floor for the plate detector.** The checkpoint
[`justjuu/rtdetr-v2-license-plate-detection`](https://huggingface.co/justjuu/rtdetr-v2-license-plate-detection)
declares `model_type: rt_detr_v2`, and the `RTDetrV2ForObjectDetection` class
that loads it only exists from 4.49. The stage refuses to guess: it raises a
message naming the installed version rather than silently upgrading a package
the backend lane also imports. Fix explicitly:

```bash
pip install "transformers>=4.49"
```

Or set the plate backend to `edge` (weightless, needs neither torch nor
transformers) to run the whole pipeline without plate weights.

### The weightless path
`detect: motion`, `plate: edge`, `ocr: template` are classical-CV backends that
need no weights and no GPU — just `numpy` (plus `cv2`/`Pillow` for real frames).
This is the stage set that can cross a source boundary, so it is what the
source-independence acceptance test drives across modes. Oracle stages cannot:
they read the synthetic generator's truth and are coupled to it by construction.

### Backend names per stage
Two groups: **ships** to the demo, and **never ships** (measurement/test only).

| Stage | Backends |
|---|---|
| detect | `rfdetr`, `motion` · `oracle`, `scripted` |
| track | `bytetrack`, `iou` · `oracle`, `scripted` |
| plate | `rtdetr`, `edge` · `oracle`, `scripted` |
| ocr | `paddle`, `template` · `oracle`, `scripted` |

---

## Model weights — manual steps

`weights/` is gitignored. Checkpoints are never committed; a weight in the
history is a weight in every clone forever.

- **Plate detector** — pulled automatically from the Hub on first run given
  `transformers>=4.49` and `HF_TOKEN`. Override with a local path if working
  offline (`local_weights` on the backend).
- **PaddleOCR** — models download on first use, or point `model_dir` at a local
  copy.
- **RF-DETR (COCO-pretrained ONNX)** — fetched from the Hub via
  `huggingface_hub` (see `ONNX_REPO_TEMPLATE` in `ai/detect/rfdetr.py`).

- **RF-DETR fine-tuned checkpoint — MANUAL, does not exist until trained.**
  A COCO-pretrained detector is not reportable on this fleet; the accuracy rows
  need a checkpoint fine-tuned on the Gujarat dataset (owned by the CV/data
  lane). When it exists, drop it here — the backend picks it up automatically:

  ```
  weights/rfdetr/trinetra_gujarat_<date>.pth     # torch, via the rfdetr package
  weights/rfdetr/trinetra_gujarat_<date>.onnx    # exported, via onnxruntime
  ```

  > **MANUAL STEP — training command.** The exact command that produces the
  > checkpoint belongs here, beside the dataset it was trained on, once the
  > dataset split is frozen. Left as a marked placeholder until then:
  >
  > ```
  > # TODO(fill in when the dataset is frozen):
  > #   <rfdetr train invocation> --dataset <path> --epochs <n> \
  > #     --output weights/rfdetr/trinetra_gujarat_<date>.pth
  > ```

---

## Credentials & environment

Environment only. Copy `.env.example` to `.env` (gitignored) and fill it in. The
worker reads the ingest token from `TRINETRA_INGEST_TOKEN` and the hub token from
`HF_TOKEN`; media credentials come from the `SENTINEL_*` vars.

`TRINETRA_INGEST_URL` is an **origin only, no path** — `HttpEventSink` appends
the contract path `/api/v1/ingest/events` itself. A per-machine setting that
could point the worker at a different path is one that can deliver events nowhere
while a 404 reads as "the server answered."

---

## Tests

```bash
python -m pytest                       # full suite
python -m pytest tests/test_pipeline.py -q
```

The suite runs without any model backend installed — model stages are exercised
through scripted fakes and oracles. `tests/test_no_attribution.py` is a
repo-wide guard: no tracked file may carry an authorship marker or an access
token. Keep it green.

---

## Measured environment (this machine)

Recorded so a mismatch is visible, not a surprise mid-demo:

```
python 3.11.0
numpy 2.4.6 · PyYAML 6.0.3 · requests 2.34.2 · Pillow 12.2.0 · scipy 1.17.1
torch 2.14.0.dev+cu130 · onnxruntime 1.28.0 · huggingface_hub 0.36.2
transformers 4.46.1        # below the 4.49 the plate detector needs — upgrade, or use plate: edge
cv2 absent                 # install opencv-python-headless for real media
paddleocr absent           # install for the paddle OCR backend
```
