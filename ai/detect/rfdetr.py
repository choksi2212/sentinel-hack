"""RF-DETR -- the detector that ships. Apache-2.0, Roboflow.

Contracts section 11 locks this choice. The reasoning is licensing before
accuracy: Ultralytics YOLO is AGPL-3.0, and shipping it would oblige us to
release the whole system under AGPL. RF-DETR is Apache-2.0 and carries no such
term. It is also, at Nano size, faster than the alternatives at comparable
accuracy on small objects, which is the regime this deployment actually lives in.

Two loading paths, tried in order:

  1. **ONNX export** (onnx-community/rfdetr_<variant>-ONNX, Apache-2.0). Needs only
     onnxruntime and numpy. This is the default because it works today on this
     machine and on a reviewer's, with no build step and no cv2.
  2. **The rfdetr package** (Roboflow's own, torch). Preferred once installed --
     it supports fine-tuned checkpoints, which the ONNX path cannot load.

A third slot is left for a locally fine-tuned checkpoint; see MANUAL WEIGHTS
below. Fine-tuning is not optional for this fleet and the reason is measured, not
assumed -- see the auto-rickshaw note in _COCO_FLEET_CAVEAT.

Every import in this module is deferred into _load(). Importing it costs nothing.
"""

import json
import os
from typing import Any, Optional, Sequence

import numpy as np

from ai.detect.base import BaseDetector, map_class_name
from ai.detect.blobs import suppress_overlaps
from ai.contracts.stages import DetectorResult

# Variants that may ship. Contracts section 11: Nano through Large are Apache-2.0.
# There is no XL in that list and adding one without re-checking its licence would
# be exactly the mistake this constant exists to prevent.
RFDETR_VARIANTS: tuple[str, ...] = ("nano", "small", "medium", "base", "large")

ONNX_REPO_TEMPLATE = "onnx-community/rfdetr_{variant}-ONNX"

# MEASURED, AND THE SINGLE MOST IMPORTANT NUMBER IN THIS FILE.
#
# The ONNX export's input is declared dynamic -- ['batch_size', 3, 'height',
# 'width'] -- and it is not. The DINOv2 backbone's windowed position embeddings
# are baked for a 384x384 grid. Measured on the nano export, real 1920x1080 night
# frame from cam04:
#
#     384x384   max score over 300 queries   0.794   3 vehicles found
#     512x512   max score over 300 queries   0.381   0 vehicles found
#     640..896  same, degrading
#     1008x1008 hard failure in Reshape
#
# So feeding a larger frame does not buy resolution: it silently produces a
# confident-looking nothing. A silent accuracy collapse that only shows up as
# "the model found no cars tonight" is the worst failure mode available here,
# which is why _preprocess asserts rather than trusting the declared shape.
RFDETR_INPUT_SIZE = 384

# Also measured on that frame, and the reason tiling is on by default.
#
# Five real cam04 frames, threshold 0.30, detections bucketed by box width -- the
# same buckets the benchmark reports in, because a single average would hide
# exactly this:
#
#                 >100  80-100  60-80  40-60  30-40  <30   total
#     single-pass   10       2      1      0      0     0      13
#     tiled 3x2     17      10      6     24     14    11      82
#
# The headline is not "6x more detections". It is that single-pass found **nothing
# below 60 px**, and 49 of the 82 tiled detections are below 60 px. The camera is
# elevated and wide: squashing 1920x1080 to 384 divides every dimension by five, so
# a 50 px vehicle at the far end of the junction becomes 10 px and is gone. Tiling
# runs the model near native scale, where those vehicles are the size it was
# trained to see.
#
# Half the junction is therefore invisible without tiling, and a benchmark run
# single-pass would report good precision on the near lane and call it the system's
# accuracy.
#
# The cost is one forward pass per tile: measured 41 ms per tile, 288 ms median for
# seven, on CPU. The sampling interval is 100 ms. So tiling is mandatory for
# accuracy and unaffordable on CPU at frame rate -- which is the whole argument for
# the GPU provider, not a preference. See ai/README.md.
DEFAULT_TILE_GRID = (3, 2)      # cols, rows

# Overlap sweep on the same frame, cross-tile NMS at IoU 0.55:
#
#     0.00 -> 15 detections     0.12 -> 16     0.30 -> 18
#
# More overlap keeps finding more, so boundary-straddling vehicles are genuinely
# being recovered rather than double-counted. 0.12 is the default because it is
# cheap; whether 0.30 is worth its larger tiles has not been separated from the
# possibility of duplicates surviving at IoU just under the threshold, and is not
# claimed either way.
DEFAULT_TILE_OVERLAP = 0.12

# COCO normalization. preprocessor_config.json for these exports sets
# do_normalize false and do_rescale true, so the only step is divide by 255.
# Applying ImageNet mean/std on top -- the obvious guess -- shifts the input
# distribution and quietly costs accuracy.
RFDETR_RESCALE_ONLY = True

_COCO_FLEET_CAVEAT = """
Measured on cam04, Paldi Junction, night: auto-rickshaws are returned as "truck"
(0.37-0.52) and "bus" (0.41-0.72). COCO has no auto-rickshaw class, so the model
has nowhere correct to put one.

Consequences, both of which are stated in the benchmark rather than smoothed over:
  - Per-class accuracy from a COCO-pretrained checkpoint is not reportable for
    this fleet. Vehicle *detection* works; vehicle *classification* does not.
  - Fine-tuning on Indian traffic is required, not a nice-to-have. Until it lands,
    the honest configuration is coco_classes_are_advisory=True, which maps every
    detection to "other" and lets the pipeline count vehicles without claiming to
    have identified them.
"""

# MANUAL WEIGHTS -----------------------------------------------------------------
# A fine-tuned checkpoint cannot be fetched automatically -- it does not exist
# until someone trains it. Drop it at the path below (or point local_weights at
# it) and this backend will prefer it over the pretrained export.
#
#     weights/rfdetr/trinetra_gujarat_<date>.pth      torch, via the rfdetr package
#     weights/rfdetr/trinetra_gujarat_<date>.onnx     exported, via onnxruntime
#
# weights/ is gitignored. The training command that produces it belongs in
# ai/README.md next to the dataset it was trained on, so the checkpoint is
# reproducible rather than a mystery binary someone found on a laptop.
DEFAULT_LOCAL_WEIGHTS_DIR = "weights/rfdetr"


class RFDETRDetector(BaseDetector):
    """RF-DETR over onnxruntime, with the torch package as an upgrade path.

    **Measured on five real cam04 night frames** (Paldi Junction, threshold 0.30,
    nano ONNX export, CPU):

        single-pass   13 detections   52 ms median    nothing below 60 px wide
        tiled 3x2     82 detections  288 ms median    49 below 60 px, 11 below 30

    Tiling is on by default because of that second column, not the first: without
    it the model cannot see the far half of the junction at all. Turn it off only
    for a like-for-like comparison against a published single-pass benchmark, and
    say which was used when reporting the number.

    Detections below roughly 30 px of *plate* width are not OCR-able (Contracts
    section 6), so some of the small end here is a counted vehicle with no readable
    plate. That is a correct outcome -- `plate: null` is a valid answer -- and it is
    still worth detecting, because a vehicle count and a movement direction do not
    need a plate.
    """

    warmup_frames = 2

    def __init__(
        self,
        *,
        variant: str = "nano",
        device: str = "auto",
        precision: str = "fp32",
        confidence_threshold: float = 0.35,
        allowed_classes: Optional[frozenset[str]] = None,
        tile: bool = True,
        tile_grid: tuple[int, int] = DEFAULT_TILE_GRID,
        tile_overlap: float = DEFAULT_TILE_OVERLAP,
        include_full_frame: bool = True,
        merge_iou: float = 0.55,
        max_detections: int = 128,
        coco_classes_are_advisory: bool = False,
        local_weights: Optional[str] = None,
        backend: str = "auto",
        cache_dir: str = "weights/hf",
    ) -> None:
        if variant not in RFDETR_VARIANTS:
            raise ValueError(
                f"variant {variant!r} is not one of {RFDETR_VARIANTS}. Sizes outside "
                "this list have not been licence-checked (Contracts section 11)."
            )
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"precision must be fp32 or fp16, got {precision!r}")
        if backend not in ("auto", "onnx", "torch"):
            raise ValueError(f"backend must be auto, onnx or torch, got {backend!r}")

        super().__init__(
            confidence_threshold=confidence_threshold,
            allowed_classes=allowed_classes,
        )
        self.variant = variant
        self.device = device
        self.precision = precision
        self.tile = tile
        self.tile_grid = tile_grid
        self.tile_overlap = tile_overlap
        self.include_full_frame = include_full_frame
        self.merge_iou = merge_iou
        self.max_detections = max_detections
        self.coco_classes_are_advisory = coco_classes_are_advisory
        self.local_weights = local_weights
        self.requested_backend = backend
        self.cache_dir = cache_dir

        self._session: Any = None
        self._torch_model: Any = None
        self._id2label: dict[int, str] = {}
        self._resolved_backend = "none"
        self._providers: list[str] = []
        self.tiles_per_frame = 0
        self.detections_dropped_unmappable = 0

    # ---------------------------------------------------------------- loading

    def _load(self) -> None:
        if self.requested_backend in ("auto", "torch") and self._try_load_torch():
            return
        if self.requested_backend in ("auto", "onnx") and self._try_load_onnx():
            return
        raise RuntimeError(
            "could not load RF-DETR. Install one of:\n"
            "    pip install onnxruntime          (CPU, works everywhere)\n"
            "    pip install onnxruntime-gpu      (CUDA, needed for 10 fps)\n"
            "    pip install rfdetr               (torch, needed for fine-tuned weights)\n"
            "See ai/README.md for the versions this has been run against."
        )

    def _try_load_torch(self) -> bool:
        """Roboflow's own package. Only path that can load a fine-tuned .pth."""
        try:
            import rfdetr  # noqa: F401
        except ImportError:
            return False

        from rfdetr import RFDETRBase, RFDETRLarge  # type: ignore

        checkpoint = self._find_local_weights((".pth", ".pt"))
        builder = RFDETRLarge if self.variant == "large" else RFDETRBase
        kwargs: dict[str, Any] = {}
        if checkpoint:
            kwargs["pretrain_weights"] = checkpoint

        self._torch_model = builder(**kwargs)
        self._resolved_backend = "torch"
        self._id2label = _coco_id2label()
        return True

    def _try_load_onnx(self) -> bool:
        try:
            import onnxruntime as ort
        except ImportError:
            return False

        local = self._find_local_weights((".onnx",))
        if local:
            model_path, config_path = local, None
        else:
            model_path, config_path = self._fetch_onnx_export()

        providers = self._select_providers(ort)
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._providers = list(self._session.get_providers())
        self._resolved_backend = "onnx"

        if config_path:
            with open(config_path, encoding="utf-8") as handle:
                config = json.load(handle)
            self._id2label = {int(k): v for k, v in config.get("id2label", {}).items()}
        if not self._id2label:
            self._id2label = _coco_id2label()
        return True

    def _fetch_onnx_export(self) -> tuple[str, str]:
        from huggingface_hub import hf_hub_download

        # The token is read from the environment and never from a config file.
        # HF_TOKEN in .env, which is gitignored -- see .env.example.
        token = os.environ.get("HF_TOKEN") or None
        repo = ONNX_REPO_TEMPLATE.format(variant=self.variant)
        filename = (
            "onnx/model_fp16.onnx" if self.precision == "fp16" else "onnx/model.onnx"
        )
        model_path = hf_hub_download(
            repo, filename, cache_dir=self.cache_dir, token=token
        )
        config_path = hf_hub_download(
            repo, "config.json", cache_dir=self.cache_dir, token=token
        )
        return model_path, config_path

    def _find_local_weights(self, suffixes: tuple[str, ...]) -> Optional[str]:
        """Look for a fine-tuned checkpoint. See MANUAL WEIGHTS above."""
        if self.local_weights:
            if not os.path.exists(self.local_weights):
                raise FileNotFoundError(
                    f"local_weights={self.local_weights!r} does not exist. Remove the "
                    "setting to fall back to the pretrained export, or put the "
                    "checkpoint there."
                )
            return self.local_weights

        directory = DEFAULT_LOCAL_WEIGHTS_DIR
        if not os.path.isdir(directory):
            return None
        candidates = sorted(
            name for name in os.listdir(directory) if name.endswith(suffixes)
        )
        return os.path.join(directory, candidates[-1]) if candidates else None

    def _select_providers(self, ort: Any) -> list[str]:
        available = set(ort.get_available_providers())
        wanted: list[str] = []

        if self.device in ("auto", "cuda"):
            for provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider"):
                if provider in available:
                    wanted.append(provider)
            if self.device == "cuda" and not wanted:
                raise RuntimeError(
                    "device='cuda' but onnxruntime has no CUDA provider. Available: "
                    f"{sorted(available)}. Install onnxruntime-gpu, or pass "
                    "device='cpu' and accept ~43 ms per tile."
                )
        wanted.append("CPUExecutionProvider")
        return wanted

    def _close(self) -> None:
        self._session = None
        self._torch_model = None

    # ------------------------------------------------------------- inference

    def _detect(self, frame_bgr: np.ndarray) -> Sequence[DetectorResult]:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR, got shape {frame.shape}")

        height, width = frame.shape[:2]
        windows = self._windows(width, height)
        self.tiles_per_frame = len(windows)

        boxes: list[tuple[int, int, int, int]] = []
        scores: list[float] = []
        labels: list[str] = []

        for x0, y0, x1, y1 in windows:
            crop = frame[y0:y1, x0:x1]
            for box, score, label in self._infer_window(crop):
                bx1, by1, bx2, by2 = box
                boxes.append((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0))
                scores.append(score)
                labels.append(label)

        if not boxes:
            return []

        # One suppression pass across every tile and the full frame together. Per
        # tile would leave a vehicle in an overlap region detected twice, which
        # becomes two tracks and two sighting events for one car.
        kept = suppress_overlaps(boxes, scores, iou_threshold=self.merge_iou)
        if len(kept) > self.max_detections:
            kept = kept[: self.max_detections]

        return [
            DetectorResult(
                bbox_xyxy=boxes[i],
                class_name=labels[i],
                confidence=round(scores[i], 4),
            )
            for i in kept
        ]

    def _windows(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        """The regions to run inference over, in frame coordinates."""
        if not self.tile:
            return [(0, 0, width, height)]

        cols, rows = self.tile_grid
        tile_w, tile_h = width // cols, height // rows
        pad_x, pad_y = int(tile_w * self.tile_overlap), int(tile_h * self.tile_overlap)

        windows: list[tuple[int, int, int, int]] = []
        if self.include_full_frame:
            # Kept alongside the tiles: it is the only window that sees a vehicle
            # large enough to straddle several tiles, such as a bus in the
            # foreground, as one object.
            windows.append((0, 0, width, height))

        for row in range(rows):
            for col in range(cols):
                x0 = max(0, col * tile_w - pad_x)
                y0 = max(0, row * tile_h - pad_y)
                x1 = min(width, (col + 1) * tile_w + pad_x)
                y1 = min(height, (row + 1) * tile_h + pad_y)
                windows.append((x0, y0, x1, y1))
        return windows

    def _infer_window(
        self, crop: np.ndarray
    ) -> list[tuple[tuple[int, int, int, int], float, str]]:
        if self._resolved_backend == "onnx":
            raw_boxes, class_ids, confidences = self._infer_onnx(crop)
        else:
            raw_boxes, class_ids, confidences = self._infer_torch(crop)

        crop_h, crop_w = crop.shape[:2]
        results: list[tuple[tuple[int, int, int, int], float, str]] = []

        for box, class_id, confidence in zip(raw_boxes, class_ids, confidences):
            if confidence < self.confidence_threshold:
                continue

            label = self._map_label(int(class_id))
            if label is None:
                self.detections_dropped_unmappable += 1
                continue

            # cxcywh, normalized to the crop. Converting in the wrong order gives
            # boxes that look plausible and crop the wrong pixels, which surfaces
            # two stages later as an OCR problem.
            cx, cy, box_w, box_h = (float(v) for v in box)
            x1 = int(round((cx - box_w / 2) * crop_w))
            y1 = int(round((cy - box_h / 2) * crop_h))
            x2 = int(round((cx + box_w / 2) * crop_w))
            y2 = int(round((cy + box_h / 2) * crop_h))

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(crop_w, x2), min(crop_h, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            results.append(((x1, y1, x2, y2), float(confidence), label))
        return results

    def _infer_onnx(self, crop: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pixel_values = _preprocess(crop, precision=self.precision)
        boxes, logits = self._session.run(
            ["pred_boxes", "logits"], {"pixel_values": pixel_values}
        )
        # Sigmoid, not softmax. RF-DETR is DETR-family: each query scores every
        # class independently and there is no background class competing for
        # probability mass. Softmax here suppresses real detections in crowded
        # frames, which is where they matter most.
        scores = _sigmoid(logits[0].astype(np.float32))
        class_ids = scores.argmax(axis=1)
        confidences = scores.max(axis=1)
        return boxes[0], class_ids, confidences

    def _infer_torch(self, crop: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from PIL import Image

        rgb = Image.fromarray(crop[:, :, ::-1])
        detections = self._torch_model.predict(rgb, threshold=self.confidence_threshold)

        crop_h, crop_w = crop.shape[:2]
        # The package returns absolute xyxy; _infer_window expects normalized
        # cxcywh, so convert here rather than branching downstream.
        boxes = []
        for x1, y1, x2, y2 in np.asarray(detections.xyxy, dtype=np.float32):
            boxes.append(
                (
                    ((x1 + x2) / 2) / crop_w,
                    ((y1 + y2) / 2) / crop_h,
                    (x2 - x1) / crop_w,
                    (y2 - y1) / crop_h,
                )
            )
        return (
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(detections.class_id, dtype=np.int64),
            np.asarray(detections.confidence, dtype=np.float32),
        )

    def _map_label(self, class_id: int) -> Optional[str]:
        raw = self._id2label.get(class_id)
        if raw is None:
            return None
        mapped = map_class_name(raw)
        if mapped is None:
            return None
        # See _COCO_FLEET_CAVEAT: on this fleet a COCO checkpoint calls an
        # auto-rickshaw a truck. When the caller has acknowledged that, the class
        # is discarded and the detection is kept -- a counted vehicle of unknown
        # type is true, and "truck" is not.
        if self.coco_classes_are_advisory:
            return "other"
        return mapped

    # ---------------------------------------------------------------- metadata

    @property
    def model_name(self) -> str:
        return f"rf-detr-{self.variant}"

    @property
    def model_version(self) -> str:
        suffix = "tiled" if self.tile else "single"
        return f"{self._resolved_backend}-{self.precision}-{suffix}"

    @property
    def license_name(self) -> str:
        return "Apache-2.0"

    @property
    def input_size(self) -> int:
        return RFDETR_INPUT_SIZE

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update(
            {
                "backend": self._resolved_backend,
                "providers": self._providers,
                "input_size": RFDETR_INPUT_SIZE,
                "tiled": self.tile,
                "tiles_per_frame": self.tiles_per_frame,
                "classes_advisory": self.coco_classes_are_advisory,
                "dropped_unmappable_class": self.detections_dropped_unmappable,
            }
        )
        return base


# --------------------------------------------------------------------- helpers


def _preprocess(crop: np.ndarray, *, precision: str = "fp32") -> np.ndarray:
    """BGR uint8 HxWx3 -> NCHW float, resized to exactly RFDETR_INPUT_SIZE.

    A plain squash to square, not a letterbox: RTDetrImageProcessor is configured
    with do_pad false, so this matches how the checkpoint was trained. Letterboxing
    would introduce grey bars the model never saw.

    Uses PIL for the resize. numpy has no good bilinear resampler, and nearest
    neighbour visibly costs accuracy on small objects -- which are the whole
    reason tiling exists.
    """
    from PIL import Image

    size = RFDETR_INPUT_SIZE
    rgb = np.ascontiguousarray(crop[:, :, ::-1])
    resized = np.asarray(
        Image.fromarray(rgb).resize((size, size), Image.BILINEAR), dtype=np.float32
    )

    # do_rescale true, do_normalize false. See RFDETR_RESCALE_ONLY.
    resized /= 255.0
    tensor = np.transpose(resized, (2, 0, 1))[None]

    if tensor.shape[2:] != (size, size):
        raise AssertionError(
            f"input is {tensor.shape[2:]}, must be ({size}, {size}). The ONNX graph "
            "declares a dynamic input and is not -- other sizes return a confident "
            "empty result. See RFDETR_INPUT_SIZE."
        )
    return tensor.astype(np.float16 if precision == "fp16" else np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic. Overflows to inf on the naive form for x < -700."""
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _coco_id2label() -> dict[int, str]:
    """The 80 COCO class names in contiguous order.

    A fallback for when config.json is unavailable, which happens with a local
    export. Only the handful of vehicle rows actually matter, but a wrong offset
    silently relabels everything, so the whole list is spelled out.
    """
    names = (
        "person bicycle car motorcycle airplane bus train truck boat traffic_light "
        "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
        "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
        "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
        "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
        "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
        "potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
        "cell_phone microwave oven toaster sink refrigerator book clock vase "
        "scissors teddy_bear hair_drier toothbrush"
    ).split()
    return dict(enumerate(names))
