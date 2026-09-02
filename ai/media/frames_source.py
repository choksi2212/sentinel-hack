"""FrameSequenceSource -- a directory of images as a stream.

Contracts section 2.3, source_mode "frames". This is how annotated datasets,
extracted hard cases and regression fixtures enter the pipeline. When a plate is
misread on one specific frame, that frame goes in a directory next to the twenty
around it and becomes a test that runs in milliseconds without a video decoder.

A directory has no timestamps. Rather than pretend otherwise, the timeline is
DECLARED: pts_ms = index * interval_ms, with interval_ms an explicit argument.
That is a different thing from the synthetic clock in ai/media/pts.py -- there,
a real clock broke and we patched around it; here there was never a clock and we
say so. Either way the session is marked pts_unreliable, because no latency
number derived from a timeline we chose is a measurement.

Ordering is by trailing frame number, not by filename. Contracts section 9
specifies %06d.jpg, which happens to sort correctly as text, but a dataset
exported as 1.jpg .. 300.jpg would otherwise be read 1, 10, 100, 101 -- and a
tracker fed a shuffled sequence produces confident nonsense rather than an error.
"""

import os
import re
from typing import Any, Optional, Sequence

import numpy as np

from ai.media.base import BaseMediaSource
from ai.media.pacing import ReplayPacer
from ai.media.sampler import TARGET_INTERVAL_MS

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Default declared spacing between consecutive images. 25 fps, matching the
# common case on these feeds. Override it whenever the true rate is known.
DEFAULT_FRAME_INTERVAL_MS = 40

_TRAILING_NUMBER = re.compile(r"(\d+)(?=\D*$)")


def _pick_decoder() -> Any:
    """Return a `path -> BGR ndarray | None` reader, preferring OpenCV.

    OpenCV first because imread returns BGR already and is the faster of the two, and every
    other adapter in this package needs it anyway. Pillow second because this adapter is the
    one that genuinely does not: a directory of PNGs needs an image decoder, not a video
    codec, and requiring a 60 MB wheel to replay twenty extracted hard cases is what turns
    "reproduce this misread" into an afternoon.

    That matters more than it looks. config/offline.yaml claims to work on a fresh clone with
    nothing installed but the dependencies -- and ai/README.md hands the dependency list to
    the backend lane, so every hard requirement removed here is one fewer thing that has to
    be installed on a machine that will never touch a camera.

    Both are optional and the failure names both, because "No module named 'cv2'" from inside
    a frame reader sends people to install OpenCV when Pillow would have done.
    """
    try:
        import cv2

        return cv2.imread
    except ImportError:
        pass

    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError(
            "frames mode needs an image decoder and found neither OpenCV nor Pillow. "
            "Install either: `pip install pillow` is the smaller of the two."
        ) from None

    def read(path: str) -> Optional[np.ndarray]:
        try:
            with Image.open(path) as handle:
                # convert() before asarray, so a palette PNG or a 16-bit TIFF becomes the
                # HxWx3 uint8 every downstream stage indexes into rather than an array whose
                # shape depends on how the file was exported.
                rgb = np.asarray(handle.convert("RGB"))
        except Exception:  # noqa: BLE001 - matches cv2.imread: unreadable file, not a crash
            return None
        # RGB to BGR. The contract says BGR (OpenCV order) and a stage that measures exposure
        # or draws a crop does not ask which decoder produced the array; swapped channels
        # would show up as a quality score that is subtly wrong and never as an error.
        return np.ascontiguousarray(rgb[:, :, ::-1])

    return read


class FrameSequenceSource(BaseMediaSource):
    """Emits FrameEnvelopes from an ordered directory of images."""

    source_mode = "frames"

    def __init__(
        self,
        camera_id: str,
        directory: str,
        *,
        interval_ms: int = DEFAULT_FRAME_INTERVAL_MS,
        loop: bool = False,
        max_loops: Optional[int] = None,
        speed: Optional[float] = None,
        target_interval_ms: int = TARGET_INTERVAL_MS,
        detect_discontinuity: bool = False,
        discontinuity_threshold: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        # Discontinuity detection defaults OFF here. A hand-picked directory of
        # hard cases is often visually unrelated frame to frame, and rotating the
        # session on every image would make the fixture untestable.
        super().__init__(
            camera_id,
            target_interval_ms=target_interval_ms,
            detect_discontinuity=detect_discontinuity,
            discontinuity_threshold=discontinuity_threshold,
            max_frames=max_frames,
            pacer=ReplayPacer(speed) if speed is not None else None,
        )
        if interval_ms <= 0:
            raise ValueError(f"interval_ms must be > 0, got {interval_ms}")

        self.directory = directory
        self.interval_ms = interval_ms
        self.loop = loop
        self.max_loops = max_loops
        self.loops_played = 0

        self._files: list[str] = []
        self._cursor = 0
        self._decoded = 0
        self._unreadable: list[str] = []
        # Chosen at open(), not here: a source is constructed by the factory before anything
        # decides to run it, and refusing to construct because Pillow is missing would make
        # `validate_config.py` fail on a machine that only ever needed to check the YAML.
        self._decode: Any = None

    # ------------------------------------------------------------------ capture

    def _open_capture(self) -> None:
        self._decode = _pick_decoder()

        if not os.path.isdir(self.directory):
            raise NotADirectoryError(f"frame directory not found: {self.directory}")

        self._files = sort_frame_files(
            entry
            for entry in os.listdir(self.directory)
            if entry.lower().endswith(IMAGE_EXTENSIONS)
        )
        if not self._files:
            raise RuntimeError(
                f"no images in {self.directory}; expected one of {IMAGE_EXTENSIONS}"
            )
        self._cursor = 0

    def _read_raw(self) -> Optional[tuple[np.ndarray, Optional[float]]]:
        while self._cursor < len(self._files):
            name = self._files[self._cursor]
            index = self._cursor
            self._cursor += 1

            frame = self._decode(os.path.join(self.directory, name))
            if frame is None:
                # A truncated or non-image file. Record it and move on -- one bad
                # file must not abort a 300-frame fixture, but it must not vanish
                # silently either, or the fixture's frame count quietly drifts.
                self._unreadable.append(name)
                continue

            self._decoded += 1
            return frame, float(index * self.interval_ms)

        return None

    def _close_capture(self) -> None:
        self._files = []
        self._cursor = 0

    # -------------------------------------------------------------------- loops

    def _handle_exhausted(self, detail: str) -> bool:
        if not self.loop:
            return False
        if self.max_loops is not None and self.loops_played + 1 >= self.max_loops:
            return False

        self.loops_played += 1
        self._cursor = 0
        self._start_session(
            reason="discontinuity",
            detail=f"sequence restart (loop {self.loops_played})",
        )
        return True

    # --------------------------------------------------------------------- misc

    @property
    def pts_unreliable(self) -> bool:
        """Always True. The timeline was declared, not measured.

        Deliberately unconditional. A frame directory can support an accuracy
        claim -- the pixels are real -- but never a timing one, and the only way
        to guarantee that is to refuse the claim at the source rather than trust
        every downstream consumer to remember.
        """
        return True

    @property
    def file_count(self) -> int:
        return len(self._files)

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update(
            {
                "directory": self.directory,
                "files": len(self._files),
                "frames_decoded": self._decoded,
                "unreadable_files": list(self._unreadable),
                "declared_interval_ms": self.interval_ms,
                "loops_played": self.loops_played,
            }
        )
        return base


def sort_frame_files(names: "Sequence[str] | Any") -> list[str]:
    """Order filenames by trailing frame number, falling back to the name.

    Exposed rather than private because the fixture tooling in scripts/ sorts the
    same directories and the two orderings must not diverge.
    """

    def key(name: str) -> tuple[int, int, str]:
        match = _TRAILING_NUMBER.search(os.path.splitext(name)[0])
        if match is None:
            return (1, 0, name.lower())
        return (0, int(match.group(1)), name.lower())

    return sorted(names, key=key)
