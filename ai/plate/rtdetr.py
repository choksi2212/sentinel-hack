"""RT-DETRv2 plate detection. justjuu/rtdetr-v2-license-plate-detection, Apache-2.0.

Technical Implementation C5 names this checkpoint as the starting point. Its published
figures -- 0.97 mAP, 0.88 on small objects -- are **on its own test set**, and are
recorded here as the reason to start with it and not as a claim about TRINETRA. Ours
comes from the width-bucket breakdown in ai/metrics.py against TRINETRA-HARD, and the
gap between the two numbers is the interesting quantity: a plate detector trained on
mostly-frontal well-lit plates meeting an elevated junction camera at night is the
single most likely place for a published number to fail to transfer.

**Licence.** Apache-2.0, verified against the model card, which is why this is the
plate backend and not one of the several YOLO-based plate detectors with better
demo videos. Those are AGPL-3.0 through Ultralytics and cannot ship. Contracts
section 11 permits them for benchmark comparison only.

**Version gap, and it is a real one on this machine.** The checkpoint declares
model_type "rt_detr_v2" and architecture RTDetrV2ForObjectDetection, which
transformers gained in 4.49. Installed here is 4.46.1, which has RTDetrForObjectDetection
(v1) but not the v2 class, so this backend cannot run until transformers is upgraded.
That is a one-line install and it is recorded in ai/README.md for the dependency
manifest rather than fixed by pinning a new version from inside AI code, since the
environment is shared with the backend service. _load raises a message naming the
exact command; the pipeline stays runnable in the meantime on the edge and oracle
backends, which is the reason those exist.
"""

from typing import Any, Optional, Sequence

import numpy as np

from ai.contracts.stages import BBox, TrackResult
from ai.plate.base import BasePlateDetector

# The model card's repository id. Not configurable by default: a benchmark row that
# does not say which checkpoint produced it is not a benchmark row, and the version
# reported in ModelProvenance is derived from this plus the resolved commit hash.
DEFAULT_REPO_ID = "justjuu/rtdetr-v2-license-plate-detection"

# Minimum transformers version carrying RTDetrV2ForObjectDetection.
MIN_TRANSFORMERS = (4, 49)

# Inference resolution. RT-DETR is trained at 640x640 and the processor resizes to it.
# Left at the trained resolution deliberately: a vehicle crop is typically far smaller
# than 640 px, so this is an upscale, and upscaling to the trained scale is what lets
# the model see a 40 px plate at the size its features expect.
INPUT_SIZE = 640

# Detections below this are not returned. Lower than the vehicle detector's 0.35
# because a plate is a small object in a crop that is known to contain a vehicle --
# the prior that something plate-shaped is present is much stronger here than the
# prior that a given region of road contains a car.
DEFAULT_THRESHOLD = 0.25


class RTDETRPlateDetector(BasePlateDetector):
    """Transformer plate detector, run per vehicle crop.

    Batches the crops of one frame into a single forward pass. At a junction with
    eight vehicles in view that is 8 crops per sampled frame, and eight sequential
    forward passes on a 12 GB laptop GPU costs more in launch overhead than in
    arithmetic -- the crops are small, so the GPU is idle between kernels. Batching
    them is the difference between the plate stage being a rounding error in the
    frame budget and being the largest item in it.

    The batch is padded to a common size by the processor rather than by resizing each
    crop to a square, which would change every plate's aspect ratio by a different
    amount and is exactly the distortion the aspect check in geometry.py would then
    have to be loosened to tolerate.
    """

    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_REPO_ID,
        device: str = "auto",
        precision: str = "fp16",
        local_weights: Optional[str] = None,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        batch_crops: bool = True,
        max_batch: int = 16,
        confidence_threshold: float = DEFAULT_THRESHOLD,
        **kwargs: Any,
    ) -> None:
        super().__init__(confidence_threshold=confidence_threshold, **kwargs)
        self.repo_id = repo_id
        self.device = device
        self.precision = precision
        self.local_weights = local_weights
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.batch_crops = bool(batch_crops)
        self.max_batch = int(max_batch)

        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._resolved_device: str = "cpu"
        self._revision: Optional[str] = None
        self._batch: list[tuple[int, np.ndarray]] = []
        self._batch_results: dict[int, Sequence[tuple[BBox, float]]] = {}
        self.forward_passes = 0
        self.crops_inferred = 0

    # ------------------------------------------------------------------- loading

    def _load(self) -> None:
        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "RT-DETRv2 plate detection needs torch and transformers. Install "
                "them, or run with plate backend 'edge' which needs neither and is "
                "why the pipeline is testable without weights."
            ) from exc

        version = tuple(int(p) for p in transformers.__version__.split(".")[:2])
        if not hasattr(transformers, "RTDetrV2ForObjectDetection"):
            # ---------------------------------------------------------------------
            # MANUAL STEP REQUIRED -- not a code defect, an environment gap.
            #
            # The checkpoint is rt_detr_v2 and this transformers build predates the
            # class. Nothing in this file can work around it: the architecture is
            # defined in transformers, not here. Resolve with the command below,
            # which is also in ai/README.md's dependency list for Mihir.
            #
            #     pip install "transformers>=4.49"
            #
            # Deliberately not attempted automatically. transformers is shared with
            # the backend service, and an AI stage silently upgrading a dependency
            # that another lane's code imports is how an integration break appears
            # on the morning of the demo.
            # ---------------------------------------------------------------------
            raise RuntimeError(
                f"transformers {transformers.__version__} has no "
                f"RTDetrV2ForObjectDetection; the "
                f"{DEFAULT_REPO_ID} checkpoint declares model_type 'rt_detr_v2', "
                f"which needs transformers >= {'.'.join(str(v) for v in MIN_TRANSFORMERS)}. "
                f"Run: pip install \"transformers>={'.'.join(str(v) for v in MIN_TRANSFORMERS)}\" "
                f"-- or set the plate backend to 'edge' to run the pipeline without "
                f"plate weights. Installed transformers is {version}."
            )

        from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

        self._torch = torch
        source = self.local_weights or self.repo_id
        kwargs: dict[str, Any] = {}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        if self.hf_token and not self.local_weights:
            kwargs["token"] = self.hf_token

        self._processor = AutoImageProcessor.from_pretrained(source, **kwargs)
        self._model = RTDetrV2ForObjectDetection.from_pretrained(source, **kwargs)

        self._resolved_device = self._resolve_device(torch)
        self._model = self._model.to(self._resolved_device)
        self._model.eval()

        # fp16 on CUDA only. On CPU half precision is emulated and slower than fp32,
        # so asking for it there is a request that should be quietly corrected rather
        # than honoured -- but it is recorded in stats so a benchmark says which ran.
        if self.precision == "fp16" and self._resolved_device.startswith("cuda"):
            self._model = self._model.half()
        else:
            self.precision = "fp32"

        self._revision = getattr(
            getattr(self._model, "config", None), "_commit_hash", None
        )

    def _resolve_device(self, torch: Any) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _close(self) -> None:
        self._model = None
        self._processor = None
        if self._torch is not None and self._resolved_device.startswith("cuda"):
            # Frees the 12 GB budget for the next stage. On a shared laptop GPU this
            # is the difference between the OCR stage loading and an allocator error.
            self._torch.cuda.empty_cache()
        self._torch = None

    # ----------------------------------------------------------------- inference

    def detect_plates(
        self, frame_bgr: np.ndarray, tracks: Sequence[TrackResult]
    ) -> dict[int, "Any"]:
        """Batched override: crops are gathered, inferred together, then dispatched.

        The base class calls _detect_in_crop once per vehicle, which is the right
        shape for every other backend and the wrong one here. So the batch is built
        first, run in one forward pass, and _detect_in_crop becomes a lookup.
        """
        if not self.batch_crops:
            return super().detect_plates(frame_bgr, tracks)

        self._batch_results = self._run_batch(frame_bgr, tracks)
        return super().detect_plates(frame_bgr, tracks)

    def _run_batch(
        self, frame_bgr: np.ndarray, tracks: Sequence[TrackResult]
    ) -> dict[int, Sequence[tuple[BBox, float]]]:
        from ai.plate.geometry import crop_vehicle

        crops: list[np.ndarray] = []
        keys: list[int] = []
        for track in tracks:
            crop, _ = crop_vehicle(
                frame_bgr, track.bbox_xyxy, pad_fraction=self.pad_fraction
            )
            if crop.size == 0:
                continue
            crops.append(crop[:, :, ::-1])  # BGR -> RGB, which the processor expects
            keys.append(track.track_id)

        results: dict[int, Sequence[tuple[BBox, float]]] = {}
        for start in range(0, len(crops), self.max_batch):
            chunk = crops[start : start + self.max_batch]
            chunk_keys = keys[start : start + self.max_batch]
            for key, boxes in zip(chunk_keys, self._infer(chunk)):
                results[key] = boxes
        return results

    def _infer(
        self, crops: Sequence[np.ndarray]
    ) -> list[Sequence[tuple[BBox, float]]]:
        torch = self._torch
        inputs = self._processor(images=list(crops), return_tensors="pt")
        inputs = {k: v.to(self._resolved_device) for k, v in inputs.items()}
        if self.precision == "fp16":
            inputs = {
                k: (v.half() if v.dtype == torch.float32 else v)
                for k, v in inputs.items()
            }

        with torch.no_grad():
            outputs = self._model(**inputs)
        self.forward_passes += 1
        self.crops_inferred += len(crops)

        # target_sizes in (height, width) per crop, so the processor returns boxes in
        # each crop's own pixel coordinates rather than in the padded batch's. Getting
        # this wrong scales every box by the ratio between the two, which on a batch of
        # similarly sized crops is a near-constant factor and therefore looks like a
        # calibration issue rather than a bug.
        target_sizes = torch.tensor(
            [[c.shape[0], c.shape[1]] for c in crops], device=self._resolved_device
        )
        processed = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.0
        )

        batch: list[Sequence[tuple[BBox, float]]] = []
        for entry in processed:
            boxes = entry["boxes"].detach().float().cpu().numpy()
            scores = entry["scores"].detach().float().cpu().numpy()
            batch.append(
                [
                    (
                        (
                            int(round(box[0])),
                            int(round(box[1])),
                            int(round(box[2])),
                            int(round(box[3])),
                        ),
                        float(score),
                    )
                    for box, score in zip(boxes, scores)
                ]
            )
        return batch

    def _detect_in_crop(
        self, crop_bgr: np.ndarray, track: TrackResult
    ) -> Sequence[tuple[BBox, float]]:
        if self.batch_crops:
            return self._batch_results.get(track.track_id, ())
        return self._infer([crop_bgr[:, :, ::-1]])[0]

    # ------------------------------------------------------------------ metadata

    @property
    def model_name(self) -> str:
        return "rtdetr-v2-plate"

    @property
    def model_version(self) -> str:
        """Repo id plus resolved commit, so a benchmark row identifies the weights.

        The commit hash is what makes the row reproducible. "rtdetr-v2-plate" alone
        does not: the upstream repository can be updated, and a number attributed to
        a name rather than a revision cannot be re-derived six weeks later.
        """
        if self._revision:
            return f"{self.repo_id}@{self._revision[:12]}"
        return self.repo_id

    @property
    def license_name(self) -> str:
        return "Apache-2.0"

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update(
            {
                "device": self._resolved_device,
                "precision": self.precision,
                "batched": self.batch_crops,
                "forward_passes": self.forward_passes,
                "crops_inferred": self.crops_inferred,
                "crops_per_pass": round(
                    self.crops_inferred / self.forward_passes, 2
                ) if self.forward_passes else 0.0,
            }
        )
        return base
