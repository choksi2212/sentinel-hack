"""Write the two evidence stills an operator needs to confirm or reject an event.

Every sighting carries `snapshot_uri` (the vehicle in context) and `plate_crop_uri` (the
plate, close up). They exist for one purpose: a human deciding whether to act on a hit
needs to see what the pipeline saw. An event whose plate cannot be looked at is an
assertion.

**This is the file where the USP could quietly be broken.** "We don't centralize every
video. We centralize intelligence." A still per sighting is intelligence; a still per frame
is video with extra steps. So the arithmetic is written down rather than assumed, and the
measurement that produced it is named rather than implied -- these numbers are the mean over
the 1920x1080 synthetic camera at seed 42, 134 emitted frames and 610 plate crops, at the
defaults below (see tests/test_emit.py, which re-measures them):

    snapshot, max_width 1280, q85     58,892 bytes    57.5 KB
    plate crop, no resize, q92         2,080 bytes     2.0 KB
    per event                         60,972 bytes    59.5 KB

Thirty cameras producing one sighting a second each is 1.74 MB/s -- **147 GB a day, from
thirty cameras.** The grid has 80,000. Hence: one pair of stills per *event*, never per
frame; downscaled; and a writer that can be switched off entirely, which is what benchmark
runs do.

(The plate crop being 28x smaller than the snapshot is worth noticing, because it is the
one an operator actually decides on. If storage ever has to be cut, the snapshot is the
expensive half and the useful half costs two kilobytes. The ratio is resolution-dependent
in the direction that helps: a bigger frame grows the snapshot quadratically and the plate
crop only as fast as the plate.)

**Staging, not retention of frames.** The snapshot has to come from the best frame of the
track, but `event_id` is not minted until the track finishes, and holding a 1080p BGR frame
per open track costs 6,220,800 bytes -- 118.6 MB at twenty tracks, on a 12 GB budget where
the detector wants most of it. So `stage_frame` encodes to JPEG bytes the moment a frame
becomes the best one seen (a 106x reduction, 1.12 MB held at twenty open tracks) and
`commit` writes those bytes under the event's own name. Memory is bounded and reported in
stats().

Encoding is not free -- 10.6 ms per staged frame, against a 100 ms sampling interval -- and
`_encode_jpeg` documents where that goes and what was done about it. It is paid only when a
track's best frame improves, which is a handful of times per vehicle rather than once per
sampled frame, and `write_snapshot: false` removes it entirely while keeping the plate crop
that costs 1 KB and ~0.2 ms.

**A URI is minted only after the bytes are on disk.** Any failure returns None. A dangling
URI is worse than a null: null renders as "no snapshot available", which is true, while a
URI to a file that was never written renders as a broken image and reads as a storage
fault rather than as the encode failure it was.
"""

import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ai.contracts.ids import TrackKey

DEFAULT_SNAPSHOT_ROOT = "artifacts/snapshots"

# 85 is where JPEG stops paying for itself on traffic stills -- 95 is roughly double the
# bytes for a difference nobody can see on a vehicle body, and below about 75 the ringing
# around plate characters starts to matter, which is the one thing in the frame a person
# will be squinting at.
DEFAULT_JPEG_QUALITY = 85

# The snapshot is downscaled to this width. It is for human confirmation, not for re-running
# detection, and 1280 is wide enough that a plate legible in the source is still legible
# here. Downscaling before encoding is also most of why the encode is cheap.
DEFAULT_MAX_SNAPSHOT_WIDTH = 1280

# The plate crop is NOT downscaled and NOT upscaled. It is the evidence that actually
# decides the operator's question, it is a few hundred pixels wide at most, and any
# resampling here would be inventing detail in the one image whose whole value is that it
# shows exactly what OCR was given. A 60 px plate should look like a 60 px plate.
PLATE_CROP_QUALITY = 92

# Ids become filenames, so they are constrained to what is safe in one. EvidenceBlock's
# docstring tells the backend never to build a path from these strings; the reason that
# warning is necessary is that a URI built from an id is only trustworthy if somebody
# checked the id, and this is the only place that can. camera_id comes from the camera
# catalogue -- a JSON file fetched over the network -- so it is input, not a constant.
_UNSAFE_IN_FILENAME = re.compile(r"[^A-Za-z0-9._-]")


def safe_component(value: str, *, fallback: str = "unknown") -> str:
    """One path component, with everything that could escape it removed.

    Two mechanisms, and it is worth being precise about which does what, because the
    obvious reading of this function is wrong. Characters outside `[A-Za-z0-9._-]` become
    `_`, which is what disarms a separator: `a/b` is `a_b`. The dot is *not* in that set --
    filenames need it -- so `..` survives the substitution untouched, and what actually
    disarms it is the `lstrip(".")`: `..` becomes the empty string and falls through to the
    fallback, and `../../etc/passwd` becomes `_.._etc_passwd`, which is one component with
    no separators in it and therefore harmless wherever it lands.

    Rejects rather than escapes, so there is no input that produces a path outside the
    root. Returning a mangled-but-safe name beats raising, because a camera whose id has an
    odd character in the catalogue should still produce evidence -- just in a
    differently-named folder.
    """
    cleaned = _UNSAFE_IN_FILENAME.sub("_", (value or "").strip())
    cleaned = cleaned.lstrip(".")  # no hidden files, no leading-dot traversal games
    return cleaned or fallback


class SnapshotWriter:
    """Stages the best frame per track, then writes both stills when the event is built.

    Thread-safe: the pipeline may run a worker per camera in one process, and they share a
    root.
    """

    def __init__(
        self,
        root: str = DEFAULT_SNAPSHOT_ROOT,
        *,
        enabled: bool = True,
        quality: int = DEFAULT_JPEG_QUALITY,
        max_width: int = DEFAULT_MAX_SNAPSHOT_WIDTH,
        uri_prefix: Optional[str] = None,
        write_snapshot: bool = True,
        write_plate_crop: bool = True,
    ) -> None:
        self.root = Path(root)
        self.enabled = bool(enabled)
        self.quality = int(quality)
        self.max_width = int(max_width)
        # Where the frontend will fetch these from, when that is not the filesystem. With
        # no prefix the URI is a file:// URL, which is correct for offline runs and for the
        # demo, and wrong the moment the API serves them from another host -- so it is a
        # config value rather than a constant.
        self.uri_prefix = uri_prefix.rstrip("/") if uri_prefix else None
        self.write_snapshot = bool(write_snapshot)
        self.write_plate_crop = bool(write_plate_crop)

        self._lock = threading.Lock()
        # TrackKey -> (quality, jpeg_bytes) for the best frame seen so far.
        self._staged: dict[TrackKey, tuple[float, bytes]] = {}

        self.frames_encoded = 0
        self.snapshots_written = 0
        self.plate_crops_written = 0
        self.encode_failures = 0
        self.write_failures = 0

    # ------------------------------------------------------------------- staging

    def stage_frame(self, track_key: TrackKey, frame_bgr: Any, quality: float) -> bool:
        """Remember this frame as the track's snapshot if it is the best one yet.

        Called with the frame in hand, which is the only moment it exists -- the
        accumulator keeps plate crops, not frames. Returns True if it was staged.

        Encoding here rather than at commit time is what bounds the memory, and the cost is
        paid only when the best frame changes -- for a top-4 buffer, a handful of times per
        vehicle rather than once per sampled frame. It is 10.6 ms per encode, not free; see
        _encode_jpeg.
        """
        if not self.enabled or not self.write_snapshot:
            return False
        with self._lock:
            staged = self._staged.get(track_key)
            if staged is not None and quality <= staged[0]:
                return False
        encoded = self._encode_jpeg(frame_bgr, max_width=self.max_width, quality=self.quality)
        if encoded is None:
            return False
        with self._lock:
            # Re-checked under the lock: two camera workers never share a TrackKey, but a
            # future batching change could, and losing the better frame to a race would be
            # invisible in the output -- the snapshot would simply be slightly worse.
            staged = self._staged.get(track_key)
            if staged is not None and quality <= staged[0]:
                return False
            self._staged[track_key] = (float(quality), encoded)
            self.frames_encoded += 1
        return True

    def has_staged(self, track_key: TrackKey) -> bool:
        with self._lock:
            return track_key in self._staged

    def drop(self, track_key: TrackKey) -> None:
        """Forget a track's staged frame without writing it."""
        with self._lock:
            self._staged.pop(track_key, None)

    def drop_session(self, stream_session_id: str) -> None:
        """Forget every staged frame for a session.

        Driven from the same place as EvidenceAccumulator.take_session, and for the same
        reason: state keyed on a track that no longer exists is how the cross-session
        track-merge bug comes back, and here it would attach one vehicle's photograph to
        another vehicle's event.
        """
        with self._lock:
            for key in [k for k in self._staged if k.stream_session_id == stream_session_id]:
                self._staged.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._staged.clear()

    # ------------------------------------------------------------------- writing

    def commit(
        self,
        track_key: TrackKey,
        *,
        event_id: str,
        observed_at: Optional[str] = None,
        plate_crop_bgr: Any = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Write the staged snapshot and the plate crop. Returns (snapshot_uri, crop_uri).

        Named by event_id, so a re-run or a retry overwrites rather than accumulating a
        second copy -- the same one decision in builder.py that makes the POST retry and
        the disk spool idempotent also makes this idempotent, for free.

        Either URI may be None independently: no frame was staged, no plate was ever
        located, the encode failed, the disk is full. Each is reported separately in
        stats() because "no plate crop" and "could not write the plate crop" are different
        problems and only the second one is a bug here.
        """
        if not self.enabled:
            return None, None

        with self._lock:
            staged = self._staged.pop(track_key, None)

        snapshot_uri: Optional[str] = None
        crop_uri: Optional[str] = None
        directory = self._directory_for(track_key.camera_id, observed_at)
        safe_event = safe_component(event_id, fallback="event")

        if staged is not None:
            path = directory / f"{safe_event}.jpg"
            if self._write_bytes(path, staged[1]):
                snapshot_uri = self._uri_for(path)
                with self._lock:
                    self.snapshots_written += 1

        if self.write_plate_crop and plate_crop_bgr is not None:
            encoded = self._encode_jpeg(plate_crop_bgr, max_width=0, quality=PLATE_CROP_QUALITY)
            if encoded is not None:
                path = directory / f"{safe_event}_plate.jpg"
                if self._write_bytes(path, encoded):
                    crop_uri = self._uri_for(path)
                    with self._lock:
                        self.plate_crops_written += 1

        return snapshot_uri, crop_uri

    def _directory_for(self, camera_id: str, observed_at: Optional[str]) -> Path:
        """<root>/<camera_id>/<YYYY-MM-DD>/.

        Partitioned by camera and day so that retention is a directory removal rather than
        a scan, and so that no single directory reaches the size where listing it becomes
        the slow part of an investigation. The date is sliced off the ISO timestamp rather
        than taken from the clock: a replay of yesterday's recording must file under
        yesterday, or the stills and the events disagree about when the vehicle passed.
        """
        day = "unknown-date"
        if observed_at and len(observed_at) >= 10:
            candidate = observed_at[:10]
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate):
                day = candidate
        return self.root / safe_component(camera_id, fallback="unknown-camera") / day

    def _encode_jpeg(self, image_bgr: Any, *, max_width: int, quality: int) -> Optional[bytes]:
        """BGR ndarray -> JPEG bytes, downscaled to max_width (0 disables).

        PIL rather than cv2, because cv2 is not installed and this does not justify the
        dependency -- see ai/README.md.

        **Where the time actually goes**, measured on a 1920x1080 frame, per call:

            BGR->RGB via Image.fromarray(arr[:, :, ::-1])    7.2 ms
            resize 1920 -> 1280 BILINEAR                     7.4 ms
            JPEG q85 encode at 1280x720                      1.1 ms
            total                                           15.7 ms

        The encode is almost free; the cost is two full-resolution pixel passes, and one of
        them was doing nothing but reordering three bytes. `arr[:, :, ::-1]` is a view and
        costs nothing, but it is non-contiguous, so Image.fromarray has to make a strided
        copy of the whole 6,220,800-byte frame to get a buffer it can use. PIL's raw decoder
        takes "BGR" as a rawmode, which folds the reorder into the buffer read that had to
        happen anyway:

            frombuffer(RGB, raw BGR) -> resize -> encode     10.6 ms

        Byte-identical output -- verified, max absolute pixel difference 0 -- for a third
        less time. Worth having because this runs in the frame loop, where the budget is
        100 ms per sampled frame and the detector wants most of it.

        Two conditions on the fast path, both checked rather than assumed. The buffer must
        be C-contiguous and exactly w*h*3 bytes, or frombuffer reads whatever follows it in
        memory and produces a sheared image -- so a non-contiguous input falls back. And
        because rawmode "BGR" differs from mode "RGB", PIL must convert during the read and
        therefore copies; it does not alias the caller's frame. That matters: the decoder
        reuses frame buffers, and an image sharing memory with the next frame would
        silently stage the wrong photograph.
        """
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover - environment dependent
            # ---------------------------------------------------------------------
            # MANUAL STEP REQUIRED -- not a code defect, an environment gap.
            #
            #     pip install pillow
            #
            # Listed in ai/README.md. Until then run with snapshots disabled: every
            # event is still valid, with both URIs null, which is exactly what the
            # schema means by null.
            # ---------------------------------------------------------------------
            with self._lock:
                self.encode_failures += 1
            return None

        import io

        try:
            array = np.asarray(image_bgr)
            if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
                with self._lock:
                    self.encode_failures += 1
                return None
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            height, width = array.shape[0], array.shape[1]

            rgb = None
            if array.flags["C_CONTIGUOUS"] and array.nbytes == width * height * 3:
                try:
                    rgb = Image.frombuffer("RGB", (width, height), array, "raw", "BGR", 0, 1)
                except (ValueError, TypeError):  # pragma: no cover - old Pillow
                    rgb = None
            if rgb is None:
                rgb = Image.fromarray(array[:, :, ::-1])

            if max_width > 0 and rgb.width > max_width:
                target_h = max(1, round(rgb.height * max_width / rgb.width))
                rgb = rgb.resize((max_width, target_h), Image.BILINEAR)
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=quality, optimize=False)
            return buffer.getvalue()
        except Exception:  # noqa: BLE001 - an unwritable still must not kill the worker
            with self._lock:
                self.encode_failures += 1
            return None

    def _write_bytes(self, path: Path, payload: bytes) -> bool:
        """Temp file plus rename, so a killed process never leaves a truncated JPEG.

        Same reasoning as the event spool: a half-written file is worse than a missing one,
        because it exists. A truncated JPEG renders as a grey band in the operator's UI and
        looks like a camera fault.
        """
        tmp = path.with_suffix(".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            return True
        except OSError:
            with self._lock:
                self.write_failures += 1
            try:
                tmp.unlink()
            except OSError:
                pass
            return False

    def _uri_for(self, path: Path) -> str:
        """The URI that goes in the event. file:// locally, prefix + relative path otherwise.

        Relative to the root, not absolute, when a prefix is configured -- otherwise the
        URI leaks the machine's directory layout into a row the frontend renders and the
        backend stores, and moving the artifact directory would invalidate every event
        already ingested.
        """
        if self.uri_prefix:
            try:
                relative = path.resolve().relative_to(self.root.resolve())
            except (OSError, ValueError):  # pragma: no cover - defensive
                relative = Path(path.name)
            return f"{self.uri_prefix}/" + "/".join(relative.parts)
        try:
            return path.resolve().as_uri()
        except (OSError, ValueError):  # pragma: no cover - defensive
            return path.as_posix()

    # --------------------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        """Counters, including the staged bytes.

        staged_bytes is here because it is this class's contribution to the memory budget
        and the whole staging design exists to keep it small. If it ever reads in the
        hundreds of megabytes then tracks are not being dropped when their sessions end,
        which is a correctness bug -- stale staged frames mean an event can carry a
        photograph of a different vehicle -- wearing the costume of a memory leak.
        """
        with self._lock:
            staged_count = len(self._staged)
            staged_bytes = sum(len(payload) for _, payload in self._staged.values())
            return {
                "root": str(self.root),
                "enabled": self.enabled,
                "staged_tracks": staged_count,
                "staged_bytes": staged_bytes,
                "frames_encoded": self.frames_encoded,
                "snapshots_written": self.snapshots_written,
                "plate_crops_written": self.plate_crops_written,
                "encode_failures": self.encode_failures,
                "write_failures": self.write_failures,
            }


class NullSnapshotWriter:
    """Stages nothing, writes nothing, returns (None, None). The benchmark's writer.

    Separate class rather than `enabled=False`, so that a config can express "no evidence
    artifacts" without the reader having to check a flag three files away to find out
    whether the run produced any. Both URIs null is a valid event, and a benchmark that
    measures the pipeline plus a JPEG encode plus an fsync is not measuring the pipeline.
    """

    def stage_frame(self, track_key: TrackKey, frame_bgr: Any, quality: float) -> bool:
        return False

    def has_staged(self, track_key: TrackKey) -> bool:
        return False

    def drop(self, track_key: TrackKey) -> None:
        return None

    def drop_session(self, stream_session_id: str) -> None:
        return None

    def clear(self) -> None:
        return None

    def commit(
        self,
        track_key: TrackKey,
        *,
        event_id: str,
        observed_at: Optional[str] = None,
        plate_crop_bgr: Any = None,
    ) -> tuple[Optional[str], Optional[str]]:
        return None, None

    def stats(self) -> dict[str, Any]:
        return {"enabled": False, "staged_tracks": 0, "staged_bytes": 0}


def build_snapshot_writer(config: Optional[dict[str, Any]]) -> Any:
    """Config dict -> writer. Absent, empty or {"enabled": false} gives the null writer."""
    config = dict(config or {})
    if not config or not config.get("enabled", True):
        return NullSnapshotWriter()
    return SnapshotWriter(
        root=str(config.get("root", DEFAULT_SNAPSHOT_ROOT)),
        enabled=True,
        quality=int(config.get("quality", DEFAULT_JPEG_QUALITY)),
        max_width=int(config.get("max_width", DEFAULT_MAX_SNAPSHOT_WIDTH)),
        uri_prefix=config.get("uri_prefix"),
        write_snapshot=bool(config.get("write_snapshot", True)),
        write_plate_crop=bool(config.get("write_plate_crop", True)),
    )
