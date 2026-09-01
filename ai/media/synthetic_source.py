"""SyntheticReplaySource -- deterministic frames with known answers.

Contracts section 2.3, source_mode "synthetic". Owner's manual sections 4.9 and 8.

Three jobs, none of which any other adapter can do.

**A pipeline test with no models and no video files.** Every frame is a pure
function of (seed, frame_index), so the whole 14-stage path runs on a CI machine
with numpy and nothing else. Seed 42 produces byte-identical frames on every run
and every machine -- which is the only reason a test can assert an exact plate,
an exact confidence and an exact event count.

**Ground truth.** The generator knows what it drew: which vehicle, which plate,
which pixels, how wide. So the primary metric -- correct final plate events over
eligible vehicle events -- is computable without a single hand-labelled frame.
That is what makes it possible to know whether a change to the fusion weights
helped, on the same afternoon the change was made.

**Fault injection.** Every rule in ai/media/pts.py describes a failure that is
hard to reproduce on demand: a decoder that freezes its clock, a stream that
jumps backwards, a scene that cuts. Here they are constructor arguments. The
alternative is unplugging a network cable at the right moment and hoping, which is
not a test.

Plates are rendered with the bitmap font in ai/media/glyphs.py, at a size that
scales with the vehicle. Plate widths therefore span the reporting buckets from
Contracts section 7 naturally -- including the sub-30-pixel bucket where
plate: null is the correct answer and any string at all is the worst outcome.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ai.contracts.enums import VEHICLE_TYPES
from ai.contracts.frame import FrameEnvelope
from ai.media.base import BaseMediaSource
from ai.media.glyphs import draw_text, text_extent, text_mask
from ai.media.pacing import ReplayPacer
from ai.media.sampler import TARGET_INTERVAL_MS

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_STEP_MS = 40          # 25 fps
DEFAULT_SEED = 42             # the reproducibility seed the tests pin

# Plate pool. Deliberately mixed: standard Gujarat formats, a BH-series plate
# that fails the grammar check in ai/normalize/plate.py, and a short older-format
# plate. A generator that only emits plates matching our own regex would validate
# nothing about how the pipeline handles the plates it will actually meet.
DEFAULT_PLATES = (
    "GJ01AB1234",
    "GJ18XY7788",
    "GJ05JK4521",
    "MH12DE9812",
    "GJ27AA0001",
    "22BH1234AA",
    "GJ3C4567",
)

# Fraction of vehicle width occupied by the plate. Roughly true of a rear plate
# viewed square-on, and the number that puts plate widths in the right buckets.
PLATE_WIDTH_RATIO = 0.42
PLATE_ASPECT = 0.24          # plate height / plate width

# Legibility floor, matching the 30 px bucket boundary in Contracts section 7.
# Below it the glyph strokes are fewer pixels wide than the strokes they
# represent, and plate: null is the correct answer.
MIN_LEGIBLE_PLATE_WIDTH_PX = 30

# Fraction of a plate that must still be unoccluded for it to count as readable.
# 0.9 rather than 1.0: a couple of pixels clipped off an edge does not stop a
# plate being read, and demanding perfection would drop most of the useful frames
# from the denominator.
MIN_PLATE_VISIBLE_FRACTION = 0.9


@dataclass(frozen=True)
class SyntheticFaults:
    """Failures to inject, each keyed to a frame index.

    Frame indices here count RAW generated frames, not emitted ones, because the
    failures being reproduced happen at the decoder, below sampling.
    """

    end_after_frames: Optional[int] = None       # stream simply ends
    raise_after_frames: Optional[int] = None     # transport error mid-read
    freeze_pts_from: Optional[int] = None        # stalled decoder clock
    pts_backwards_at: Optional[int] = None       # timeline goes backwards
    pts_jump_at: Optional[int] = None            # large forward jump
    pts_jump_ms: int = 60_000
    pts_unavailable_from: Optional[int] = None   # decoder reports no PTS
    black_frames: tuple[int, ...] = ()           # fully black frames
    scene_cut_at: Optional[int] = None           # hard cut to unrelated footage

    def touches(self, frame: int) -> bool:
        return frame in self.black_frames or frame in {
            self.end_after_frames,
            self.raise_after_frames,
            self.pts_backwards_at,
            self.pts_jump_at,
            self.scene_cut_at,
        }


@dataclass(frozen=True)
class SyntheticVehicle:
    """One vehicle's whole trajectory, decided at construction time."""

    vehicle_id: int
    plate: str
    vehicle_type: str
    colour: tuple[int, int, int]
    spawn_frame: int
    frames_visible: int
    lane_y: int
    start_x: float
    end_x: float
    start_height: int
    end_height: int

    def progress(self, frame: int) -> Optional[float]:
        """0.0 at spawn to 1.0 at exit, or None when not on screen."""
        if frame < self.spawn_frame:
            return None
        offset = frame - self.spawn_frame
        if offset >= self.frames_visible:
            return None
        if self.frames_visible <= 1:
            return 0.0
        return offset / (self.frames_visible - 1)


@dataclass
class VehicleTruth:
    """What was actually drawn, for one vehicle on one frame."""

    vehicle_id: int
    plate: str
    vehicle_type: str
    vehicle_bbox_xyxy: tuple[int, int, int, int]
    plate_bbox_xyxy: Optional[tuple[int, int, int, int]]
    plate_width_px: int
    plate_legible: bool
    plate_visible_fraction: float = 1.0


@dataclass
class FrameTruth:
    """Ground truth for one generated frame."""

    frame_index: int
    pts_ms: int
    vehicles: list[VehicleTruth] = field(default_factory=list)

    @property
    def legible_plates(self) -> set[str]:
        """Plates a correct pipeline should be able to read on this frame."""
        return {v.plate for v in self.vehicles if v.plate_legible}


class SyntheticReplaySource(BaseMediaSource):
    """Procedurally generated traffic with known answers and injectable faults."""

    source_mode = "synthetic"

    def __init__(
        self,
        camera_id: str,
        *,
        seed: int = DEFAULT_SEED,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        total_frames: int = 500,
        step_ms: int = DEFAULT_STEP_MS,
        plates: tuple[str, ...] = DEFAULT_PLATES,
        vehicles_per_100_frames: float = 3.0,
        faults: Optional[SyntheticFaults] = None,
        speed: Optional[float] = None,
        target_interval_ms: int = TARGET_INTERVAL_MS,
        detect_discontinuity: bool = True,
        max_frames: Optional[int] = None,
    ) -> None:
        super().__init__(
            camera_id,
            target_interval_ms=target_interval_ms,
            detect_discontinuity=detect_discontinuity,
            max_frames=max_frames,
            pacer=ReplayPacer(speed) if speed is not None else None,
        )
        self.seed = seed
        self.width = width
        self.height = height
        self.total_frames = total_frames
        self.step_ms = step_ms
        self.plates = plates
        self.faults = faults or SyntheticFaults()

        self._raw_index = 0
        self._background: Optional[np.ndarray] = None
        self._cut_background: Optional[np.ndarray] = None
        self._vehicles: list[SyntheticVehicle] = []
        self._truth: dict[int, FrameTruth] = {}
        self._truth_by_pts: dict[int, FrameTruth] = {}
        self._raised = False

        # One RNG, consumed in one fixed order, never touched during generation.
        # Drawing randomness per frame would make the output depend on how many
        # frames had been read, and a test that reads 50 frames would disagree
        # with one that reads 500.
        rng = random.Random(seed)
        self._vehicles = self._plan_vehicles(rng, vehicles_per_100_frames)
        self._background_seed = rng.randrange(1 << 30)
        self._cut_seed = rng.randrange(1 << 30)

    # ------------------------------------------------------------------ capture

    def _open_capture(self) -> None:
        self._raw_index = 0
        self._raised = False
        self._background = _road_background(
            self.width, self.height, self._background_seed
        )
        self._cut_background = None

    def _read_raw(self) -> Optional[tuple[np.ndarray, Optional[float]]]:
        index = self._raw_index

        if self.faults.end_after_frames is not None and index >= self.faults.end_after_frames:
            return None
        if index >= self.total_frames:
            return None
        if (
            self.faults.raise_after_frames is not None
            and index >= self.faults.raise_after_frames
            and not self._raised
        ):
            self._raised = True
            raise RuntimeError(
                f"injected transport failure at raw frame {index} for {self.camera_id}"
            )

        self._raw_index += 1
        frame, truth = self._render(index)
        self._truth[index] = truth
        # Indexed by the frame's own timeline position, which is what a caller
        # holding a FrameEnvelope can actually match on. truth.pts_ms is the
        # honest generator clock; _pts_for below may deliberately corrupt the
        # reported value under fault injection, and the two must not be confused.
        self._truth_by_pts[truth.pts_ms] = truth
        return frame, self._pts_for(index)

    def _close_capture(self) -> None:
        self._background = None
        self._cut_background = None

    # -------------------------------------------------------------------- clock

    def _pts_for(self, index: int) -> Optional[float]:
        faults = self.faults

        if faults.pts_unavailable_from is not None and index >= faults.pts_unavailable_from:
            return None

        if faults.freeze_pts_from is not None and index >= faults.freeze_pts_from:
            return float(faults.freeze_pts_from * self.step_ms)

        pts = index * self.step_ms

        if faults.pts_backwards_at is not None and index >= faults.pts_backwards_at:
            pts -= faults.pts_backwards_at * self.step_ms + 1_000

        if faults.pts_jump_at is not None and index >= faults.pts_jump_at:
            pts += faults.pts_jump_ms

        return float(pts)

    # ------------------------------------------------------------------ planning

    def _plan_vehicles(
        self, rng: random.Random, per_100_frames: float
    ) -> list[SyntheticVehicle]:
        count = max(1, int(round(self.total_frames * per_100_frames / 100.0)))
        lanes = 3
        lane_height = self.height // (lanes + 1)

        vehicles: list[SyntheticVehicle] = []
        for vehicle_id in range(1, count + 1):
            plate = self.plates[(vehicle_id - 1) % len(self.plates)]
            lane = rng.randrange(lanes)
            frames_visible = rng.randint(24, 70)
            spawn = rng.randrange(0, max(1, self.total_frames - 5))

            # Approaching: the box grows as it crosses, so one vehicle's plate
            # passes through several width buckets during its own track. That is
            # what makes fusion worth measuring -- an early frame is unreadable
            # and a late one is not, and the answer has to come from the whole
            # sequence rather than from whichever frame happened to be sampled.
            start_height = rng.randint(34, 70)
            end_height = start_height + rng.randint(60, 150)

            left_to_right = rng.random() < 0.5
            margin = 260.0
            start_x = -margin if left_to_right else self.width + margin
            end_x = self.width + margin if left_to_right else -margin

            vehicles.append(
                SyntheticVehicle(
                    vehicle_id=vehicle_id,
                    plate=plate,
                    vehicle_type=VEHICLE_TYPES[rng.randrange(len(VEHICLE_TYPES))],
                    colour=(
                        rng.randint(40, 215),
                        rng.randint(40, 215),
                        rng.randint(40, 215),
                    ),
                    spawn_frame=spawn,
                    frames_visible=frames_visible,
                    lane_y=lane_height * (lane + 1),
                    start_x=start_x,
                    end_x=end_x,
                    start_height=start_height,
                    end_height=end_height,
                )
            )

        vehicles.sort(key=lambda v: (v.spawn_frame, v.vehicle_id))
        return vehicles

    # ----------------------------------------------------------------- rendering

    def _render(self, index: int) -> tuple[np.ndarray, FrameTruth]:
        truth = FrameTruth(frame_index=index, pts_ms=index * self.step_ms)

        if index in self.faults.black_frames:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8), truth

        cut_at = self.faults.scene_cut_at
        if cut_at is not None and index >= cut_at:
            if self._cut_background is None:
                self._cut_background = _road_background(
                    self.width, self.height, self._cut_seed, night=True
                )
            # After a cut, the scene is unrelated and carries no vehicles. An
            # empty frame is the honest depiction: whatever was being tracked is
            # gone, which is precisely the condition the tracker must not paper over.
            return self._cut_background.copy(), truth

        assert self._background is not None
        frame = self._background.copy()

        # Which vehicle owns each plate pixel, so occlusion can be measured after
        # all drawing is done. 0 means no plate.
        owner = np.zeros((self.height, self.width), dtype=np.int32)

        visible = [
            (vehicle, progress)
            for vehicle in self._vehicles
            if (progress := vehicle.progress(index)) is not None
        ]
        # Nearer vehicles last, so they occlude the ones further away. Drawing in
        # spawn order instead would let a distant car paint over a close one,
        # which is both wrong to look at and wrong in the ground truth.
        visible.sort(key=lambda pair: _height_at(pair[0], pair[1]))

        for vehicle, progress in visible:
            drawn = _draw_vehicle(frame, vehicle, progress, owner)
            if drawn is not None:
                truth.vehicles.append(drawn)

        _resolve_occlusion(truth, owner)
        return frame, truth

    # ------------------------------------------------------------------ truth API
    #
    # Two indices exist and they are not interchangeable. The generator produces
    # raw frames 0, 1, 2, ... at step_ms apart; the sampler emits roughly every
    # third one, and BaseMediaSource stamps each emitted frame with its own
    # counter starting at 0 per session. So raw frame 42 is emitted as
    # FrameEnvelope.frame_index 14.
    #
    # Looking truth up by envelope.frame_index therefore compares the pipeline's
    # output against a DIFFERENT frame's ground truth -- and because neighbouring
    # frames look similar, the resulting accuracy number is wrong in a way that
    # still looks reasonable. That is the worst kind of measurement bug.
    #
    # pts_ms is the one identifier that survives sampling: it is on the envelope,
    # it is on FrameTruth, and it means the same thing on both. Everything that
    # pairs a frame with its truth goes through it.

    def truth_for_envelope(self, envelope: FrameEnvelope) -> Optional[FrameTruth]:
        """Ground truth for the frame this envelope carries. The safe lookup.

        Use this, not truth_for(). Returns None if the envelope did not come from
        this source, or if PTS was suppressed by a fault -- in which case there is
        no way to identify the frame and pretending otherwise would be a guess.
        """
        if envelope.pts_ms is None:
            return None
        return self.truth_at_pts(envelope.pts_ms)

    def truth_at_pts(self, pts_ms: int) -> Optional[FrameTruth]:
        """Ground truth for a source timestamp.

        Exact match only. A faulted run whose PTS was shifted backwards or jumped
        forward will not resolve, which is correct: the pixels are real but the
        timeline is a lie, and a fault-injection run is testing recovery, not
        accuracy.
        """
        return self._truth_by_pts.get(int(pts_ms))

    def truth_for(self, raw_frame_index: int) -> Optional[FrameTruth]:
        """Ground truth by RAW GENERATOR index -- not by envelope.frame_index.

        Kept because the generator's own tests address frames by the index they
        were rendered at. If you are holding a FrameEnvelope, you want
        truth_for_envelope; see the note above this method.
        """
        return self._truth.get(raw_frame_index)

    def expected_plates(self) -> set[str]:
        """Every plate that is legible on at least one frame of the whole run.

        This is the denominator of the primary metric for a synthetic run: the
        plates a correct pipeline should end up reporting. A plate that is never
        legible on any frame is deliberately excluded -- counting it as a miss
        would penalise the pipeline for refusing to invent an answer, which is
        the behaviour we most want to keep.
        """
        legible: set[str] = set()
        for truth in self._truth.values():
            legible |= truth.legible_plates
        return legible

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update(
            {
                "seed": self.seed,
                "resolution": f"{self.width}x{self.height}",
                "raw_frames_generated": self._raw_index,
                "planned_vehicles": len(self._vehicles),
                "faults_configured": [
                    name
                    for name, value in vars(self.faults).items()
                    if value not in (None, (), 0)
                ],
            }
        )
        return base


# ------------------------------------------------------------------- generation


def _road_background(
    width: int, height: int, seed: int, *, night: bool = False
) -> np.ndarray:
    """A static road scene. Deterministic in seed, and computed once per session.

    Not noise. A flat or random background would give the discontinuity detector
    in ai/media/discontinuity.py nothing stable to correlate against, and the
    histogram threshold that separates a real scene cut from ordinary traffic
    would stop meaning anything.

    night=True renders the same geometry in a completely different exposure
    regime. Used for the injected scene cut: two daytime road scenes correlate at
    0.72 to 0.95 even with different seeds -- above the detector's threshold, and
    correctly so, since a different view of a similar road is not a discontinuity.
    Only a genuinely unrelated image is, and that is what this produces.
    """
    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    horizon = int(height * 0.28)

    if night:
        sky_top, sky_bottom = rng.integers(8, 18), rng.integers(18, 34)
        road_top, road_bottom = rng.integers(10, 20), rng.integers(22, 38)
        marking, building = 70, (12, 40)
    else:
        sky_top, sky_bottom = rng.integers(120, 190), rng.integers(190, 235)
        road_top, road_bottom = rng.integers(58, 78), rng.integers(88, 112)
        marking, building = 205, (70, 150)

    sky = np.linspace(sky_top, sky_bottom, horizon, dtype=np.float64)
    frame[:horizon] = sky[:, None, None].astype(np.uint8)

    road = np.linspace(road_top, road_bottom, height - horizon, dtype=np.float64)
    frame[horizon:] = road[:, None, None].astype(np.uint8)

    # Lane markings, so the scene has real structure and real edges.
    lanes = 3
    lane_height = height // (lanes + 1)
    for lane in range(1, lanes + 1):
        y = lane_height * lane + lane_height // 2
        if not (horizon < y < height - 4):
            continue
        dash = 46
        for x in range(0, width, dash * 2):
            frame[y : y + 3, x : min(x + dash, width)] = marking

    # Fixed roadside blocks: buildings above the horizon, which is where most of
    # the histogram mass sits and therefore what makes a cut detectable.
    for _ in range(int(rng.integers(5, 10))):
        bx = int(rng.integers(0, max(1, width - 120)))
        bw = int(rng.integers(48, 130))
        bh = int(rng.integers(30, horizon))
        shade = int(rng.integers(*building))
        frame[horizon - bh : horizon, bx : bx + bw] = shade

    grain = rng.integers(-4, 5, size=(height, width, 1), dtype=np.int16)
    return np.clip(frame.astype(np.int16) + grain, 0, 255).astype(np.uint8)


def _height_at(vehicle: SyntheticVehicle, progress: float) -> int:
    return int(
        vehicle.start_height + (vehicle.end_height - vehicle.start_height) * progress
    )


def _resolve_occlusion(truth: FrameTruth, owner: np.ndarray) -> None:
    """Downgrade plates that a later-drawn vehicle covered up.

    Without this the ground truth claims a plate was readable on a frame where it
    was behind a bus, the pipeline correctly reports nothing, and the metric
    records a miss. That penalises exactly the behaviour Contracts section 3.2
    exists to protect -- refusing to invent a plate -- and it would push tuning in
    the worst possible direction.
    """
    for vehicle in truth.vehicles:
        if vehicle.plate_bbox_xyxy is None:
            vehicle.plate_visible_fraction = 0.0
            vehicle.plate_legible = False
            continue

        x0, y0, x1, y1 = vehicle.plate_bbox_xyxy
        window = owner[y0:y1, x0:x1]
        if window.size == 0:
            vehicle.plate_visible_fraction = 0.0
            vehicle.plate_legible = False
            continue

        fraction = float((window == vehicle.vehicle_id).mean())
        vehicle.plate_visible_fraction = round(fraction, 4)
        vehicle.plate_legible = (
            vehicle.plate_width_px >= MIN_LEGIBLE_PLATE_WIDTH_PX
            and fraction >= MIN_PLATE_VISIBLE_FRACTION
        )


def _draw_vehicle(
    frame: np.ndarray,
    vehicle: SyntheticVehicle,
    progress: float,
    owner: np.ndarray,
) -> Optional[VehicleTruth]:
    height_px = _height_at(vehicle, progress)
    width_px = int(height_px * 1.5)
    centre_x = int(vehicle.start_x + (vehicle.end_x - vehicle.start_x) * progress)

    x0 = centre_x - width_px // 2
    y0 = vehicle.lane_y - height_px // 2
    x1, y1 = x0 + width_px, y0 + height_px

    box = _fill_rect(frame, (x0, y0, x1, y1), vehicle.colour)
    if box is None:
        return None

    # A vehicle body clears any plate behind it. Without this, a plate occluded by
    # a body rather than by another plate would still read as fully visible.
    _fill_region(owner, (x0, y0, x1, y1), 0)

    # Windscreen band, so the vehicle is not a flat rectangle. A detector trained
    # on real vehicles will not fire on a solid block, and a synthetic frame that
    # only a stub can handle is worth much less than one a real model can.
    band_y = y0 + int(height_px * 0.18)
    _fill_rect(
        frame,
        (x0 + width_px // 8, band_y, x1 - width_px // 8, band_y + max(3, height_px // 5)),
        tuple(int(c * 0.45) for c in vehicle.colour),  # type: ignore[arg-type]
    )

    plate_width = max(6, int(width_px * PLATE_WIDTH_RATIO))
    plate_height = max(4, int(plate_width * PLATE_ASPECT))
    plate_x0 = centre_x - plate_width // 2
    plate_y0 = y1 - int(height_px * 0.22) - plate_height

    plate_box = _draw_plate(
        frame, vehicle.plate, (plate_x0, plate_y0, plate_width, plate_height)
    )
    if plate_box is not None:
        _fill_region(owner, plate_box, vehicle.vehicle_id)

    return VehicleTruth(
        vehicle_id=vehicle.vehicle_id,
        plate=vehicle.plate,
        vehicle_type=vehicle.vehicle_type,
        vehicle_bbox_xyxy=box,
        plate_bbox_xyxy=plate_box,
        plate_width_px=plate_width,
        plate_legible=plate_box is not None
        and plate_width >= MIN_LEGIBLE_PLATE_WIDTH_PX,
    )


def _draw_plate(
    frame: np.ndarray,
    plate: str,
    rect: tuple[int, int, int, int],
) -> Optional[tuple[int, int, int, int]]:
    """Draw a white plate with dark text, scaled to fit the given box."""
    x0, y0, width, height = rect

    box = _fill_rect(frame, (x0, y0, x0 + width, y0 + height), (238, 238, 238))
    if box is None:
        return None
    _stroke_rect(frame, (x0, y0, x0 + width, y0 + height), (30, 30, 30))

    inner_w = max(1, width - 4)
    inner_h = max(1, height - 4)
    mask = text_mask(plate, scale=1)
    if mask.size == 0:
        return box

    scaled = _resize_mask_nearest(mask, inner_w, inner_h)
    _apply_mask(frame, scaled, (x0 + 2, y0 + 2), (26, 26, 26))
    return box


def _resize_mask_nearest(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resample of a boolean mask.

    Nearest-neighbour deliberately: it degrades to unreadable at small sizes the
    same way a real plate does, and it introduces no interpolated grey that a
    sharpness score would read as blur that is not there.
    """
    src_h, src_w = mask.shape
    rows = (np.arange(height) * src_h // max(1, height)).clip(0, src_h - 1)
    cols = (np.arange(width) * src_w // max(1, width)).clip(0, src_w - 1)
    return mask[rows[:, None], cols[None, :]]


def _apply_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    origin: tuple[int, int],
    colour: tuple[int, int, int],
) -> None:
    x0, y0 = origin
    frame_h, frame_w = frame.shape[:2]
    mask_h, mask_w = mask.shape

    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    copy_w = min(mask_w - src_x0, frame_w - dst_x0)
    copy_h = min(mask_h - src_y0, frame_h - dst_y0)
    if copy_w <= 0 or copy_h <= 0:
        return

    window = mask[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]
    region = frame[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w]
    region[window] = np.array(colour, dtype=frame.dtype)


def _fill_rect(
    frame: np.ndarray,
    xyxy: tuple[int, int, int, int],
    colour: tuple[int, int, int],
) -> Optional[tuple[int, int, int, int]]:
    """Fill a rectangle, clipped to the frame. Returns the clipped xyxy or None."""
    x0, y0, x1, y1 = xyxy
    frame_h, frame_w = frame.shape[:2]

    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(frame_w, x1), min(frame_h, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return None

    frame[cy0:cy1, cx0:cx1] = np.array(colour, dtype=frame.dtype)
    return (cx0, cy0, cx1, cy1)


def _fill_region(buffer: np.ndarray, xyxy: tuple[int, int, int, int], value: int) -> None:
    """Write a scalar into a clipped rectangle of a 2-D buffer."""
    x0, y0, x1, y1 = xyxy
    h, w = buffer.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w, x1), min(h, y1)
    if cx1 > cx0 and cy1 > cy0:
        buffer[cy0:cy1, cx0:cx1] = value


def _stroke_rect(
    frame: np.ndarray,
    xyxy: tuple[int, int, int, int],
    colour: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    x0, y0, x1, y1 = xyxy
    _fill_rect(frame, (x0, y0, x1, y0 + thickness), colour)
    _fill_rect(frame, (x0, y1 - thickness, x1, y1), colour)
    _fill_rect(frame, (x0, y0, x0 + thickness, y1), colour)
    _fill_rect(frame, (x1 - thickness, y0, x1, y1), colour)


__all__ = [
    "DEFAULT_PLATES",
    "DEFAULT_SEED",
    "FrameTruth",
    "SyntheticFaults",
    "SyntheticReplaySource",
    "SyntheticVehicle",
    "VehicleTruth",
    "draw_text",
    "text_extent",
]
