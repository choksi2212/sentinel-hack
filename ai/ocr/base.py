"""The OCR stage boundary -- stage 8 of the 14. One plate box in, one string out.

Two rules govern this file and both come from Contracts section 12, and both are
rules about *refusing* rather than about reading:

**plate: null is a valid, correct, frequently-right answer.** A vehicle facing away,
a plate behind a tow bar, a motorcycle at 30 px -- the right output is nothing. A
stage that always returns a string will be right more often on a benchmark that only
counts reads and catastrophically wrong in an investigation, because a fabricated
plate does not merely lose information, it points at a real vehicle that was never
there. Refusing to read is implemented as a width floor here, before any backend runs.

**Confidence is never multiplied into something new and presented.** This stage emits
one number per read and it is the backend's own. ai/fusion combines them by evidence
share, ai/quality weights by image quality, and neither is permitted to produce a
product and call it a probability.

**Preprocessing variants are tried one at a time, never composed.** See
ai/ocr/preprocess.py for why -- it is the single most tempting mistake in this stage.
"""

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from ai.contracts.stages import BBox, PlateCandidate

# Pixels of margin taken around the plate box before reading.
#
# 3 px rather than a fraction of the box, because what this compensates for is a fixed
# quantity: the plate detector's box tends to sit just inside the true plate edge --
# the outermost character stroke is what the detector locks onto, not the plate's
# painted border. A fractional pad would take 2 px on a small plate, where the error
# is the same 1-2 px it is on a large one, and clip the first character exactly where
# clipping is most expensive.
PLATE_CROP_PAD_PX = 3

# Below this plate width, refuse to read and return None.
#
# The arithmetic: an Indian plate carries up to ten characters, so per-character width
# is roughly plate_width / 12 once inter-character gaps are counted. The 5x7 glyph
# shapes that distinguish 8 from B need about one pixel per glyph column to survive at
# all, so five pixels per character -- meaning full fidelity starts around 60 px, and
# everything below that is interpolation. At 24 px there are two pixels per character
# and the distinction between 0, O, D, 8 and B is simply not present in the data.
#
# 24 rather than 30 or 60 deliberately. Refusing at 60 would discard the 24-60 px band
# where most real junction plates live and where temporal fusion earns its keep: many
# individually-unreliable reads of the same track, agreeing on most characters. The
# floor is set where the signal is absent, not where it is weak. The synthetic
# generator's own legibility threshold is 30 px, which is an independent estimate of
# the same boundary from the rendering side, and the two agreeing to within 6 px is
# the only cross-check available.
MIN_OCR_PLATE_WIDTH_PX = 24


@dataclass(frozen=True)
class OCRRead:
    """One backend's reading of one plate crop.

    Lane-internal: the pipeline turns this plus the track and the frame's timing into
    the PlateObservation that ai/fusion consumes. Deliberately not added to
    ai/contracts/stages.py -- the contract boundary carries what other lanes depend on,
    and no other lane needs to know that this stage tries six preprocessing variants.

    text is the raw read, uppercased and stripped of separators but NOT grammar-checked
    or normalized. That happens in ai/normalize/plate.py, and keeping the two apart is
    what makes it possible to report how often OCR was right versus how often the
    grammar rules rescued or ruined it.
    """

    text: str
    confidence: float
    variant: str
    variants_tried: int = 1
    variants_agreeing: int = 1
    char_confidences: Optional[tuple[float, ...]] = None

    @property
    def agreement(self) -> float:
        """Share of preprocessing variants that produced this same string.

        The honest companion to `confidence`. This stage selects the highest-confidence
        variant out of several, and max-of-N is biased upward by construction: six
        independent noisy reads will produce one that looks good by luck. That bias
        cannot be removed by choosing differently -- taking the mean would throw away
        the variant that actually worked -- so it is disclosed instead. A read where
        five of six variants agree is worth more than an equally confident read where
        none do, and ai/quality is where that gets used.
        """
        if self.variants_tried <= 0:
            return 0.0
        return self.variants_agreeing / self.variants_tried


@dataclass(frozen=True)
class FrameRef:
    """Which frame a deferred plate crop came from. Identity only, no pixels.

    Exists because the pipeline reads a track's crops when the track finishes rather
    than when each frame arrives, and by then the frame is gone. A backend that needs
    frame identity -- the oracle, to look up ground truth; the scripted stub, to index
    its script -- gets it from this instead of from the FrameEnvelope it can no longer
    see.

    Deliberately not the envelope itself and deliberately not holding frame_bgr. Four
    crops per track across eight open tracks would pin eight whole frames, which at
    1920x1080 is 48 MB retained for four integers' worth of information.
    """

    camera_id: str
    stream_session_id: str
    frame_index: int
    pts_ms: int


@runtime_checkable
class OCREngine(Protocol):
    """What the pipeline requires of an OCR backend."""

    def load(self) -> None: ...

    def close(self) -> None: ...

    def read(
        self, frame_bgr: np.ndarray, candidate: PlateCandidate
    ) -> Optional[OCRRead]: ...

    def cut_crop(
        self, frame_bgr: np.ndarray, candidate: PlateCandidate
    ) -> Optional[np.ndarray]: ...

    def read_crop(
        self,
        crop_bgr: Optional[np.ndarray],
        candidate: PlateCandidate,
        *,
        frame_ref: Optional["FrameRef"] = None,
    ) -> Optional[OCRRead]: ...

    @property
    def model_name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def license_name(self) -> str: ...

    @property
    def ships(self) -> bool: ...


class BaseOCR(ABC):
    """Crop, refuse-if-too-small, run each preprocessing variant, take the best.

    Subclasses implement _read_crop against a single prepared image and know nothing
    about cropping, the width floor, or the variant loop. That division is what makes
    the width floor unskippable: a backend cannot accidentally read a 9 px plate,
    because it is never handed one.
    """

    def __init__(
        self,
        *,
        min_plate_width_px: int = MIN_OCR_PLATE_WIDTH_PX,
        pad_px: int = PLATE_CROP_PAD_PX,
        variants: Optional[Sequence[str]] = None,
    ) -> None:
        self.min_plate_width_px = int(min_plate_width_px)
        self.pad_px = int(pad_px)
        self._variant_names = tuple(variants) if variants is not None else None
        self._loaded = False

        self.plates_seen = 0
        self.refused_small = 0
        self.crops_empty = 0
        self.reads_attempted = 0
        self.reads_empty = 0
        self.reads_returned = 0
        self.variant_wins: Counter = Counter()

    # ------------------------------------------------------------------- lifecycle

    def load(self) -> None:
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def close(self) -> None:
        if not self._loaded:
            return
        self._close()
        self._loaded = False

    @abstractmethod
    def _load(self) -> None:
        """Acquire whatever the backend needs. May raise; the worker reports it."""

    def _close(self) -> None:
        """Release resources. Most backends have none."""

    # ---------------------------------------------------------------------- reading

    @property
    def variants(self) -> tuple[str, ...]:
        """Preprocessing variants this backend will try, in order."""
        from ai.ocr.preprocess import DEFAULT_VARIANTS

        return self._variant_names if self._variant_names is not None else DEFAULT_VARIANTS

    def read(
        self, frame_bgr: np.ndarray, candidate: PlateCandidate
    ) -> Optional[OCRRead]:
        """Read one plate. None means "no plate could be read", which is an answer."""
        return self.read_crop(self.cut_crop(frame_bgr, candidate), candidate)

    def cut_crop(
        self, frame_bgr: np.ndarray, candidate: PlateCandidate
    ) -> Optional[np.ndarray]:
        """Exactly the crop read() would read, for a caller that defers the read.

        The pipeline runs OCR on a track's top-K crops when the track finishes, not on
        every sampled frame -- that is the whole reason ai/fusion keeps a top-K buffer.
        By then the frame is gone, so the crop has to be cut at offer time and carried.
        Cutting it here rather than in the pipeline is what keeps the deferred read
        identical to the immediate one: the padding is this backend's pad_px, and a
        caller that guessed 4 px where the backend uses 3 would produce reads that
        cannot be compared against a non-deferred run.

        Returns a VIEW. A view keeps the entire frame alive as its base, so a caller
        that stores one for later must copy it -- four views per track across eight
        tracks is eight whole frames pinned in memory, which at 1920x1080 is 48 MB of
        invisible retention that looks like a leak in the decoder.
        """
        return self._crop_plate(frame_bgr, candidate.plate_bbox_xyxy)

    def read_crop(
        self,
        crop_bgr: Optional[np.ndarray],
        candidate: PlateCandidate,
        *,
        frame_ref: Optional["FrameRef"] = None,
    ) -> Optional[OCRRead]:
        """Read an already-cut, already-padded plate crop.

        `frame_ref` restores frame identity for a deferred read, and omitting it when the
        backend needs it is a silent failure rather than a loud one. OracleOCR resolves
        ground truth per frame and ScriptedOCR indexes its script by frame_index; read at
        flush time with no reference, both would answer against whichever frame happened
        to be last, find no vehicle whose box matches, and return None. Every plate comes
        back unread, every event carries plate: null, and nothing raises -- so the offline
        run that exists to verify the pipeline reports a clean, plausible zero.

        The width floor is checked against the candidate rather than the crop, and that
        is deliberate: the candidate carries the plate's width in the scene, and the
        crop's own width is that plus padding, or double it after an upscale variant.
        Checking the crop would let a 20 px plate through as soon as somebody enlarged
        it, which is the exact failure MIN_OCR_PLATE_WIDTH_PX exists to prevent -- the
        characters are not in the data, and interpolation cannot put them there.
        """
        if frame_ref is not None:
            self._begin_frame(frame_ref)

        self.plates_seen += 1

        if candidate.plate_width_px < self.min_plate_width_px:
            # Refused before any backend sees it. See MIN_OCR_PLATE_WIDTH_PX.
            self.refused_small += 1
            return None

        if crop_bgr is None or crop_bgr.size == 0:
            self.crops_empty += 1
            return None

        return self._read_variants(crop_bgr, candidate)

    def read_all(
        self, frame_bgr: np.ndarray, candidates: dict[int, PlateCandidate]
    ) -> dict[int, OCRRead]:
        """Read every plate in one frame. Missing keys mean no read, as in ai/plate."""
        out: dict[int, OCRRead] = {}
        for track_id, candidate in candidates.items():
            read = self.read(frame_bgr, candidate)
            if read is not None:
                out[track_id] = read
        return out

    def read_all_envelope(
        self, envelope: Any, candidates: dict[int, PlateCandidate]
    ) -> dict[int, OCRRead]:
        """read_all, for a backend that needs to know which frame this is.

        Only the oracle needs it, and it needs it for the same reason the oracle plate
        detector does -- to look up ground truth. Kept as a separate entry point rather
        than threading the envelope through read(), so that a real backend's signature
        does not advertise access to information it must never use.
        """
        self._begin_frame(envelope)
        return self.read_all(envelope.frame_bgr, candidates)

    def _begin_frame(self, envelope: Any) -> None:
        """Hook for backends that need frame identity. Default is to ignore it."""

    def _read_variants(
        self, crop: np.ndarray, candidate: PlateCandidate
    ) -> Optional[OCRRead]:
        from ai.ocr.preprocess import apply_variant

        reads: list[OCRRead] = []
        for name in self.variants:
            prepared = apply_variant(crop, name)
            if prepared is None or prepared.size == 0:
                continue
            self.reads_attempted += 1
            read = self._read_crop(prepared, candidate)
            if read is None or not read.text:
                self.reads_empty += 1
                continue
            reads.append(
                OCRRead(
                    text=read.text,
                    confidence=read.confidence,
                    variant=name,
                    char_confidences=read.char_confidences,
                )
            )

        if not reads:
            return None

        tried = len(self.variants)
        best = max(reads, key=lambda r: r.confidence)
        agreeing = sum(1 for r in reads if r.text == best.text)
        self.reads_returned += 1
        self.variant_wins[best.variant] += 1

        return OCRRead(
            text=best.text,
            confidence=best.confidence,
            variant=best.variant,
            variants_tried=tried,
            variants_agreeing=agreeing,
            char_confidences=best.char_confidences,
        )

    def _crop_plate(self, frame_bgr: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        """The plate box plus a fixed margin, clamped to the frame.

        Returns a view, not a copy. Every preprocessing variant copies before writing,
        which is checked in tests/test_ocr.py -- a variant that wrote in place would
        corrupt the frame for every stage after this one, including the snapshot.
        """
        height, width = frame_bgr.shape[:2]
        x1 = max(0, bbox[0] - self.pad_px)
        y1 = max(0, bbox[1] - self.pad_px)
        x2 = min(width, bbox[2] + self.pad_px)
        y2 = min(height, bbox[3] + self.pad_px)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2]

    @abstractmethod
    def _read_crop(
        self, crop_bgr: np.ndarray, candidate: PlateCandidate
    ) -> Optional[OCRRead]:
        """Read one prepared plate crop.

        The crop is already padded, already width-checked, and already run through one
        preprocessing variant -- which may have converted it to single-channel or
        doubled its size. Handle both: a backend that assumes three channels will fail
        on the grayscale variant and succeed on the others, which shows up as that
        variant simply never winning and is very hard to notice.

        candidate carries the plate box in full-frame coordinates. Use it for the plate
        width if the backend adapts to scale; do not use it to index anything outside
        the crop. Note that the crop's own dimensions no longer match the candidate's
        after upscale_2x, so the candidate's width is the *scene* width and the crop's
        is the pixels actually being read -- they are different numbers and the width
        buckets in ai/metrics.py mean the former.

        Return None for "nothing legible here". The variant and agreement fields on the
        returned OCRRead are filled in by the caller; leave them at their defaults.
        """

    # ------------------------------------------------------------------- properties

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def model_version(self) -> str: ...

    @property
    @abstractmethod
    def license_name(self) -> str: ...

    @property
    def ships(self) -> bool:
        """Whether this backend's numbers may appear in a published claim."""
        return True

    def stats(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "license": self.license_name,
            "ships": self.ships,
            "variants": list(self.variants),
            "plates_seen": self.plates_seen,
            "refused_small": self.refused_small,
            "crops_empty": self.crops_empty,
            "reads_attempted": self.reads_attempted,
            "reads_empty": self.reads_empty,
            "reads_returned": self.reads_returned,
            "variant_wins": dict(self.variant_wins),
            # Named a proxy, not a rate: the denominator counts every plate box handed
            # to this stage, including the ones it was right to refuse.
            "read_proxy": round(self.reads_returned / self.plates_seen, 4)
            if self.plates_seen
            else 0.0,
        }
