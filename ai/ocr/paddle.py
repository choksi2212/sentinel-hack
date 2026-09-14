"""PaddleOCR -- the shipping OCR backend. Apache-2.0, stage 8 of the 14.

Locked in Technical Implementation C7 and chosen over the alternatives on licence and
on script coverage:

    PaddleOCR         Apache-2.0   ships. Latin + digits, quantised models, ONNX export
    EasyOCR           Apache-2.0   viable fallback; heavier, no per-character scores
    Tesseract         Apache-2.0   built for documents, weak on 40 px plate crops
    TrOCR             MIT          transformer, far too slow per crop at 8 vehicles/frame
    commercial ALPR   proprietary  excluded outright, this has to be reproducible

**paddleocr is not installed in this environment**, which is a deliberate state rather
than an oversight -- see _load. The pipeline runs end to end without it on the
template backend, so nothing is blocked, and ai/README.md carries the install command
for Mihir to fold into requirements.txt.

**Detection is disabled.** PaddleOCR ships a text *detector* and a text *recogniser*
and this stage only wants the second: the plate box is already known, from ai/plate,
and running Paddle's detector inside a 60x15 crop asks it to find text regions in an
image that is entirely one text region. It costs a model's worth of latency to
rediscover what the caller already passed in, and it fails in a specific way -- on a
tight crop it returns a box slightly inside the characters and clips the first one.
So rec_only, with the crop passed straight to the recogniser.
"""

from typing import Any, Optional

import numpy as np

from ai.contracts.stages import PlateCandidate
from ai.ocr.base import OCRRead, BaseOCR

# The English recognition model. Latin script and digits, which is all an Indian plate
# needs -- the Devanagari variants that appear on some state plates are not legally
# valid on the registration plate itself, and a Devanagari-capable model would enlarge
# the character set the recogniser has to discriminate for no gain in what it can read.
DEFAULT_LANG = "en"

# PaddleOCR's own confidence floor. Set low here on purpose: this stage's job is to
# report what it read together with how sure it is, and the decision about whether a
# 0.31 read is worth keeping belongs to ai/fusion, which can see the other four frames
# of the same track. Discarding it here throws away a vote before anything has had a
# chance to count it.
DEFAULT_DROP_SCORE = 0.10


class PaddleOCR(BaseOCR):
    """PaddleOCR's recogniser on an already-located plate crop.

    Per-character confidences are requested where the installed version exposes them,
    because they are what makes character-level temporal fusion possible: with them,
    ai/fusion can take character 3 from the frame that read character 3 well and
    character 7 from a different frame. Without them the whole string is one vote. The
    field is Optional on OCRRead for exactly this reason -- older PaddleOCR builds
    return only a string and a score, and the pipeline degrades to whole-string voting
    rather than failing.
    """

    def __init__(
        self,
        *,
        lang: str = DEFAULT_LANG,
        use_gpu: bool = True,
        drop_score: float = DEFAULT_DROP_SCORE,
        model_dir: Optional[str] = None,
        use_angle_cls: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lang = lang
        self.use_gpu = bool(use_gpu)
        self.drop_score = float(drop_score)
        self.model_dir = model_dir
        # Off by default. The angle classifier decides whether a crop is upside down,
        # and a plate on a vehicle is never upside down -- it is rotated by at most the
        # camera's roll. Enabling it adds a model and a chance to flip a correct read
        # 180 degrees, which produces a confident wrong string.
        self.use_angle_cls = bool(use_angle_cls)
        self._engine: Any = None
        self._version: str = "unknown"
        self._has_char_scores = False
        self.backend_errors = 0

    def _load(self) -> None:
        try:
            import paddleocr
        except ImportError as exc:  # pragma: no cover - environment dependent
            # -----------------------------------------------------------------------
            # MANUAL STEP REQUIRED -- not a code defect, an environment gap.
            #
            # paddleocr and its paddlepaddle runtime are not installed here. Nothing
            # in this file can work around that; the recogniser weights and the
            # inference runtime both live in those packages.
            #
            #     pip install paddlepaddle-gpu paddleocr
            #
            # Deliberately not attempted automatically, for the same reason
            # ai/plate/rtdetr.py does not silently upgrade transformers: paddlepaddle
            # pins numpy and protobuf ranges, and an AI stage quietly installing it
            # can break the backend service's imports. It belongs in Mihir's
            # requirements.txt, which is why ai/README.md lists it rather than this
            # lane creating a requirements file of its own.
            #
            # Until then, run with ocr backend 'template', which needs nothing and is
            # why the 14 stages are measurable end to end today.
            # -----------------------------------------------------------------------
            raise RuntimeError(
                "PaddleOCR is not installed. Run: pip install paddlepaddle-gpu "
                "paddleocr -- or set the ocr backend to 'template', which needs no "
                "weights and no download. See ai/README.md for the full dependency "
                "list."
            ) from exc

        self._version = getattr(paddleocr, "__version__", "unknown")
        kwargs: dict[str, Any] = {
            "lang": self.lang,
            "use_angle_cls": self.use_angle_cls,
            "drop_score": self.drop_score,
            "show_log": False,
        }
        if self.model_dir:
            kwargs["rec_model_dir"] = self.model_dir

        # PaddleOCR renamed its device flag between 2.x and 3.x. Try the current name,
        # fall back to the old one, and if neither is accepted construct without it --
        # running on CPU is slower but correct, and a stage that refuses to start
        # because of a keyword rename is worse than a stage that runs slowly.
        for device_kwargs in (
            {"device": "gpu" if self.use_gpu else "cpu"},
            {"use_gpu": self.use_gpu},
            {},
        ):
            try:
                self._engine = paddleocr.PaddleOCR(**kwargs, **device_kwargs)
                break
            except (TypeError, ValueError):
                continue
        if self._engine is None:
            raise RuntimeError(
                f"paddleocr {self._version} rejected every known device keyword. "
                f"Set ocr backend 'template' to keep the pipeline running and report "
                f"the installed version."
            )

    def _close(self) -> None:
        self._engine = None

    def _read_crop(
        self, crop_bgr: np.ndarray, candidate: PlateCandidate
    ) -> Optional[OCRRead]:
        if self._engine is None:
            raise RuntimeError("PaddleOCR.load() was not called")

        # Contiguous because paddle's binding reads the buffer directly and a strided
        # view -- which _crop_plate returns and the raw variant preserves -- either
        # copies silently or reads the wrong pixels depending on the build.
        image = np.ascontiguousarray(crop_bgr)

        try:
            result = self._engine.ocr(image, det=False, cls=self.use_angle_cls)
        except Exception:  # pragma: no cover - backend dependent
            # A backend exception is one unreadable plate, not a dead worker. A single
            # malformed crop must not take down a stream that is otherwise running:
            # the counter makes it visible in stats() if it starts happening often.
            self.backend_errors += 1
            return None

        parsed = _parse_result(result)
        if parsed is None:
            return None
        text, confidence, char_scores = parsed
        if char_scores is not None:
            self._has_char_scores = True

        # Uppercased and stripped of separators, but NOT grammar-corrected. Plate
        # grammar is ai/normalize/plate.py's job, and keeping them apart is what makes
        # it possible to report how often OCR was right against how often the grammar
        # rules rescued or ruined it.
        cleaned = "".join(ch for ch in text.upper() if ch.isalnum())
        if not cleaned:
            return None

        return OCRRead(
            text=cleaned,
            confidence=float(confidence),
            variant="paddle",
            char_confidences=char_scores,
        )

    @property
    def model_name(self) -> str:
        return f"paddleocr-rec-{self.lang}"

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def license_name(self) -> str:
        return "Apache-2.0"

    @property
    def ships(self) -> bool:
        return True

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update(
            {
                "lang": self.lang,
                "use_gpu": self.use_gpu,
                "angle_cls": self.use_angle_cls,
                "char_confidences_available": self._has_char_scores,
                "backend_errors": self.backend_errors,
            }
        )
        return base


def _parse_result(
    result: Any,
) -> Optional[tuple[str, float, Optional[tuple[float, ...]]]]:
    """Pull (text, confidence, per-character scores) out of whatever Paddle returned.

    PaddleOCR's return shape has changed across releases and differs between det=True
    and det=False. Observed forms, all of which have to work because this code cannot
    pin the installed version:

        [[('GJ01AB1234', 0.98)]]        2.x, det=False, per-image nesting
        [('GJ01AB1234', 0.98)]          2.x, det=False, flattened
        [{'rec_texts': [...], ...}]     3.x predict-style dict

    Written defensively on purpose. The alternative -- indexing result[0][0][0] and
    trusting it -- turns a version bump into a TypeError inside the frame loop, which
    surfaces as the whole stream dying rather than as one unreadable plate.
    """
    if not result:
        return None

    node: Any = result
    # Descend through list nesting until something that is not a bare list appears.
    for _ in range(4):
        if isinstance(node, (list, tuple)) and len(node) == 1:
            node = node[0]
        else:
            break

    if isinstance(node, dict):
        texts = node.get("rec_texts") or node.get("rec_text")
        scores = node.get("rec_scores") or node.get("rec_score")
        text = texts[0] if isinstance(texts, (list, tuple)) and texts else texts
        score = scores[0] if isinstance(scores, (list, tuple)) and scores else scores
        if not isinstance(text, str) or score is None:
            return None
        return text, float(score), _as_char_scores(node.get("rec_char_scores"))

    if isinstance(node, (list, tuple)) and len(node) >= 2:
        text, score = node[0], node[1]
        if isinstance(text, (list, tuple)):
            # det=True shape: [box, (text, score)]. node[0] is the polygon, so the
            # payload is node[1]. Not expected here since det is off, but a version
            # that ignores det= and detects anyway would otherwise read the box
            # coordinates as a plate string.
            pair = node[1]
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                return None
            text, score = pair[0], pair[1]
        if not isinstance(text, str):
            return None
        try:
            return text, float(score), None
        except (TypeError, ValueError):
            return None

    return None


def _as_char_scores(raw: Any) -> Optional[tuple[float, ...]]:
    if not raw:
        return None
    try:
        values = raw[0] if isinstance(raw[0], (list, tuple)) else raw
        return tuple(float(v) for v in values)
    except (TypeError, ValueError, IndexError):
        return None
