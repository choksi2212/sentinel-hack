"""Detection -- stage 3 of the fourteen in Contracts section 10, and the origin of
every downstream number.

Detection is where a vehicle first enters the record, so it is where the three ways
the system can lie are first reachable. Weighted worst first, they are:

  1. A vehicle that was never there. Stage 4 groups detections into a track, stage 8
     votes a plate across that track's frames, stage 13 emits one event per track. A
     phantom detection -- a below-threshold box the filter let through, an oracle
     inventing a vehicle the truth never held -- becomes a track, then an event, then a
     plate belonging to no real car. The filter drops into counters so every loss is
     visible; the oracle reproduces truth verbatim and stamps the false positives it is
     *asked* to inject with a confidence a real read never carries. Both are pinned here.

  2. Two vehicles merged into one. blobs.label_mask is 4-connected on purpose: two cars
     whose boxes touch only at a corner are two components, not one. A merge is the
     silent form of failure 1 -- one event whose plate belongs to neither vehicle, and
     nothing throws. test_track.py carries the same warning one stage downstream.

  3. A sighting silently dropped -- an undercount. The confidence/class filter drops
     into counters, never into silence. RF-DETR's preprocess is asserted to always
     produce the 384-pixel square, because a wrong input size is a confident empty
     result -- the undercount that reads as a quiet road. The motion backend returns
     nothing during its warmup by construction: a bounded, documented silence that must
     not be "fixed" into frame-one garbage.

The suite runs with no model backend installed. The one detector that ships,
RFDETRDetector, defers every heavy import into load(); its constructor, tiling geometry
and preprocess are exercised here with onnxruntime, torch, cv2 and transformers all
poisoned, because the package's whole promise is that swapping the detector is a config
change testable without a GPU or a 108 MB download (ai/detect/__init__.py). The
classical and oracle backends are numpy only, so they are tested as themselves rather
than through fakes.
"""

import contextlib
import sys

import numpy as np
import pytest

from ai.contracts.frame import FrameEnvelope
from ai.contracts.stages import DetectorResult
from ai.detect import (
    DETECTOR_NAMES,
    SHIPPABLE_DETECTORS,
    VEHICLE_CLASSES,
    BaseDetector,
    DetectorConfigError,
    MotionBlobDetector,
    OracleDegradation,
    OracleDetector,
    ScriptedDetector,
    blobs_from_mask,
    build_detector,
    describe_detector,
    detector_ships,
    iou,
    is_shippable_class,
    label_mask,
    map_class_name,
    normalize_detector_config,
    resolve_allowed_classes,
    suppress_overlaps,
)

# --------------------------------------------------------------------------- helpers

CAM = "cam04"


def frame(index, *, pixels=None, width=200, height=100, mode="synthetic"):
    """A FrameEnvelope for the oracle, which reads the envelope rather than pixels.

    The classical and scripted backends take raw pixel arrays, not envelopes, so this
    is only needed where truth_for_envelope is consulted.
    """
    if pixels is None:
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        height, width = pixels.shape[:2]
    return FrameEnvelope(
        camera_id=CAM,
        stream_session_id="sess-detect",
        frame_index=index,
        pts_ms=index * 40,
        wallclock_utc=None,
        frame_bgr=pixels,
        width=width,
        height=height,
        source_mode=mode,
    )


def blank():
    """A throwaway pixel buffer for backends that ignore frame contents."""
    return np.zeros((10, 10, 3), dtype=np.uint8)


class _Vehicle:
    """One row of synthetic ground truth, shaped like what OracleDetector reads."""

    def __init__(self, bbox, vehicle_type, plate):
        self.vehicle_bbox_xyxy = bbox
        self.vehicle_type = vehicle_type
        self.plate = plate


class _Truth:
    def __init__(self, frame_index, vehicles):
        self.frame_index = frame_index
        self.vehicles = vehicles


class _TruthSource:
    """A media source exposing only truth_for_envelope -- the oracle's whole dependency."""

    def __init__(self, truth):
        self._truth = truth

    def truth_for_envelope(self, envelope):  # noqa: ARG002 -- fixed truth, envelope ignored
        return self._truth


class _ListDetector(BaseDetector):
    """Emits a fixed list so BaseDetector._filter is the only behaviour under test."""

    def __init__(self, results, **kwargs):
        super().__init__(**kwargs)
        self._results = results

    def _load(self):
        pass

    def _detect(self, frame_bgr):  # noqa: ARG002
        return list(self._results)

    @property
    def model_name(self):
        return "list-fake"

    @property
    def model_version(self):
        return "test"


# Heavy model backends. Poisoned wholesale in test_rfdetr_construction_and_geometry_
# need_no_backend to prove the shipping detector's constructor and pure geometry touch
# none of them. numpy, PIL, scipy, yaml and requests are core and deliberately absent.
_HEAVY_BACKENDS = frozenset(
    {
        "onnxruntime",
        "onnxruntime_gpu",
        "torch",
        "rfdetr",
        "cv2",
        "paddleocr",
        "paddlepaddle",
        "transformers",
        "huggingface_hub",
    }
)


class _ImportBlocker:
    def find_spec(self, name, path=None, target=None):  # noqa: ARG002
        if name.split(".")[0] in _HEAVY_BACKENDS:
            raise ImportError(f"blocked for test: {name}")
        return None


@contextlib.contextmanager
def backends_blocked():
    """Make every heavy backend import raise ImportError for the duration of the block."""
    blocker = _ImportBlocker()
    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.split(".")[0] in _HEAVY_BACKENDS
    }
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


# --------------------------------------------- fabrication: nothing invented, nothing lost

def test_perfect_oracle_reproduces_truth_exactly():
    truth = _Truth(
        0,
        [
            _Vehicle((10, 10, 60, 50), "car", "GJ01AB1234"),
            _Vehicle((80, 20, 110, 60), "motorcycle", None),
        ],
    )
    out = OracleDetector(_TruthSource(truth)).detect_envelope(frame(0))
    # Boxes and classes are the truth, verbatim -- nothing invented, nothing dropped.
    assert [(d.class_name, d.bbox_xyxy) for d in out] == [
        ("car", (10, 10, 60, 50)),
        ("motorcycle", (80, 20, 110, 60)),
    ]
    # Even a perfect oracle reports a calibrated 0.92, never a fake 1.0.
    assert all(d.confidence == 0.92 for d in out)
    det = OracleDetector(_TruthSource(truth))
    assert det.is_oracle is True and det.ships is False


def test_oracle_can_honestly_report_nothing():
    # Miss everything, invent nothing -> an empty list. The oracle is allowed to see a
    # quiet road; what it must never do is fill that silence with a guess.
    truth = _Truth(3, [_Vehicle((10, 10, 60, 50), "car", "A")])
    det = OracleDetector(
        _TruthSource(truth),
        degradation=OracleDegradation(miss_rate=1.0, false_positives_per_frame=0.0, seed=1),
    )
    assert det.detect_envelope(frame(3)) == []


def test_oracle_marks_false_positives_with_a_distinct_confidence():
    # Asked to inject false positives, the oracle stamps them at 0.55, never the 0.92 of
    # a real read -- so a benchmark can separate invented boxes from true ones.
    truth = _Truth(
        3,
        [_Vehicle((10, 10, 60, 50), "car", "A"), _Vehicle((80, 20, 140, 70), "truck", "B")],
    )
    det = OracleDetector(
        _TruthSource(truth),
        degradation=OracleDegradation(miss_rate=0.0, false_positives_per_frame=3.0, seed=1),
    )
    out = det.detect_envelope(frame(3))
    assert len(out) > 2
    confidences = {round(d.confidence, 3) for d in out}
    assert 0.55 in confidences and 0.92 in confidences


def test_oracle_degradation_is_deterministic():
    # Same seed, same frame -> identical output, so a degraded-oracle benchmark is
    # reproducible rather than a different number every run.
    deg = OracleDegradation(miss_rate=0.3, jitter_px=4, false_positives_per_frame=0.5, seed=7)

    def run():
        truth = _Truth(
            4,
            [_Vehicle((10, 10, 60, 50), "car", "A"), _Vehicle((80, 20, 140, 70), "truck", "B")],
        )
        out = OracleDetector(_TruthSource(truth), degradation=deg).detect_envelope(frame(4))
        return [(d.class_name, d.bbox_xyxy, d.confidence) for d in out]

    assert run() == run()


def test_oracle_jitter_perturbs_boxes():
    truth_boxes = {(10, 10, 60, 50), (80, 20, 140, 70)}
    truth = _Truth(
        3,
        [_Vehicle((10, 10, 60, 50), "car", "A"), _Vehicle((80, 20, 140, 70), "truck", "B")],
    )
    det = OracleDetector(
        _TruthSource(truth),
        degradation=OracleDegradation(jitter_px=5, false_positives_per_frame=0.0, seed=1),
    )
    boxes = [d.bbox_xyxy for d in det.detect_envelope(frame(3))]
    assert any(box not in truth_boxes for box in boxes)  # jitter actually displaced a box


def test_oracle_low_confidence_knob_lowers_reads():
    # A calibration lever for the quality gate: force every read down to the low value.
    truth = _Truth(
        3,
        [_Vehicle((10, 10, 60, 50), "car", "A"), _Vehicle((80, 20, 140, 70), "truck", "B")],
    )
    det = OracleDetector(
        _TruthSource(truth),
        degradation=OracleDegradation(low_confidence_rate=1.0, low_confidence_value=0.3, seed=1),
    )
    assert all(d.confidence == 0.3 for d in det.detect_envelope(frame(3)))


def test_filter_drops_are_counted_never_silent():
    # car 0.9 kept; car 0.2 below the default 0.35 threshold; person 0.99 wrong class.
    det = _ListDetector(
        [
            DetectorResult((0, 0, 40, 40), "car", 0.9),
            DetectorResult((0, 0, 10, 10), "car", 0.2),
            DetectorResult((0, 0, 30, 30), "person", 0.99),
        ]
    )
    kept = det.detect(np.zeros((100, 100, 3), np.uint8))
    assert [(k.class_name, k.confidence) for k in kept] == [("car", 0.9)]
    # Every removed detection lands in a counter -- an undercount you can see.
    assert det.detections_emitted == 1
    assert det.detections_dropped_confidence == 1  # the 0.2 car
    assert det.detections_dropped_class == 1  # the 0.99 person
    assert det.frames_seen == 1


def test_filter_threshold_is_kept_at_the_boundary():
    # A detection exactly at threshold is kept (the drop is confidence < threshold), so
    # a calibrated cutoff is not quietly one notch stricter than configured.
    det = _ListDetector(
        [DetectorResult((0, 0, 20, 20), "car", 0.5)], confidence_threshold=0.5
    )
    assert len(det.detect(np.zeros((40, 40, 3), np.uint8))) == 1
    assert det.detections_dropped_confidence == 0


# ------------------------------------------ identity: two vehicles must not merge into one

def test_diagonal_touch_is_two_vehicles_not_one():
    # 4-connectivity: boxes touching only at a corner are separate components. A merge
    # here becomes one track, one event, one plate belonging to neither vehicle.
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True  # diagonal neighbour only
    _, count = label_mask(mask)
    assert count == 2


def test_edge_adjacent_pixels_are_one_vehicle():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = mask[1, 2] = mask[2, 1] = True  # share edges
    _, count = label_mask(mask)
    assert count == 1


def test_label_mask_empty_and_rejects_non_2d():
    _, count = label_mask(np.zeros((4, 4), dtype=bool))
    assert count == 0  # empty mask, no components -- not one background component
    with pytest.raises(ValueError):
        label_mask(np.zeros((2, 2, 2), dtype=bool))


def test_blobs_returned_largest_first_and_area_filtered():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:12, 2:12] = True  # 100 px
    mask[15:18, 15:18] = True  # 9 px
    found = blobs_from_mask(mask, min_area=1)
    assert [b.area for b in found] == [100, 9]  # largest first
    assert tuple(found[0][:4]) == (2, 2, 12, 12)  # xyxy of the big one
    assert found[0].fill_ratio == 1.0  # a solid rectangle fills its box
    # min_area removes the small blob entirely -- a size gate, not a merge.
    assert len(blobs_from_mask(mask, min_area=100)) == 1


def test_iou_is_exclusive_on_shared_edges():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # Edge-touching boxes do not overlap -- exclusive xyxy, so NMS won't fuse neighbours.
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3, abs=1e-6)


def test_nms_keeps_distinct_boxes_and_suppresses_duplicates():
    boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (100, 100, 110, 110)]
    scores = [0.9, 0.8, 0.5]
    # Highest-scoring duplicate kept, its near-twin suppressed, the far box survives:
    # neither a double-count of one vehicle nor a dropped distinct one.
    assert suppress_overlaps(boxes, scores) == [0, 2]


def test_nms_rejects_length_mismatch():
    with pytest.raises(ValueError):
        suppress_overlaps([(0, 0, 1, 1)], [0.1, 0.2])


# --------------------------- undercount: the shipping detector's 384 invariant + warmup

def test_rfdetr_construction_and_geometry_need_no_backend():
    # The package promise: swapping to the shipping detector is a config change,
    # testable with no GPU and no 108 MB download. Constructor + pure geometry must work
    # with every heavy backend import poisoned; the failure is deferred to load().
    with backends_blocked():
        from ai.detect.rfdetr import RFDETRDetector

        det = RFDETRDetector(variant="nano", tile=True, include_full_frame=True)
        assert det.ships is True and det.license_name == "Apache-2.0"
        assert det.input_size == 384
        # Full frame + a 3x2 grid = 7 windows, all pure numpy geometry.
        assert len(det._windows(1280, 720)) == 7
        # Deferred failure: asking it to actually load a model is where the missing
        # backend is reported -- a clean RuntimeError, not an ImportError at construction.
        with pytest.raises(RuntimeError):
            det.load()


def test_preprocess_always_produces_the_384_square():
    # "The single most important number." A wrong input size is a confident empty
    # result -- the undercount that reads as a quiet road. Any input resizes to 384.
    from ai.detect import rfdetr

    for height, width in [(100, 200), (720, 1280), (17, 5), (384, 384)]:
        tensor = rfdetr._preprocess(np.zeros((height, width, 3), np.uint8))
        assert tensor.shape == (1, 3, 384, 384)
        assert tensor.dtype == np.float32
    # Normalised into [0, 1]: a solid-white crop is all 1.0, not 255.
    white = rfdetr._preprocess(np.full((50, 50, 3), 255, np.uint8))
    assert float(white.max()) == 1.0 and float(white.min()) == 1.0
    # The fp16 path keeps the shape and only changes dtype.
    half = rfdetr._preprocess(np.zeros((50, 50, 3), np.uint8), precision="fp16")
    assert half.shape == (1, 3, 384, 384) and half.dtype == np.float16


def test_rfdetr_rejects_bad_construction_eagerly():
    from ai.detect.rfdetr import RFDETRDetector

    for kwargs in ({"variant": "xl"}, {"precision": "int8"}, {"backend": "tensorflow"}):
        with pytest.raises(ValueError):
            RFDETRDetector(**kwargs)  # rejected before any load(), so a typo fails fast


def test_windows_single_when_untiled():
    from ai.detect.rfdetr import RFDETRDetector

    assert RFDETRDetector(tile=False)._windows(1280, 720) == [(0, 0, 1280, 720)]
    # Tiling without the full frame drops to just the grid.
    assert len(RFDETRDetector(tile=True, include_full_frame=False)._windows(1280, 720)) == 6


def test_sigmoid_is_stable_at_extremes():
    from ai.detect import rfdetr

    out = rfdetr._sigmoid(np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0], dtype=np.float32))
    assert out[0] == pytest.approx(0.0) and out[-1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(0.5)
    assert np.all(np.isfinite(out))  # no overflow at +/-1000


def test_coco_label_offsets_match_the_vehicle_ids():
    # The COCO ids the detector folds into vehicle classes. An off-by-one here silently
    # relabels every detection -- cars would read as buses.
    from ai.detect import rfdetr

    labels = rfdetr._coco_id2label()
    assert len(labels) == 80
    assert (labels[2], labels[3], labels[5], labels[7]) == ("car", "motorcycle", "bus", "truck")


def test_motion_warmup_returns_nothing_by_construction():
    # The background model needs frames before it can subtract. Those early empties are
    # a bounded, documented silence -- not a bug to "fix" into frame-one garbage.
    det = MotionBlobDetector()
    static = np.full((100, 200, 3), 100, np.uint8)
    assert det.detect(static) == []  # first frame builds the background
    assert det.frames_suppressed_warmup == 1


def test_motion_detects_a_moving_blob_after_warmup():
    det = MotionBlobDetector()
    static = np.full((100, 200, 3), 100, np.uint8)
    for _ in range(6):  # exhaust the warmup window
        det.detect(static)
    moved = static.copy()
    moved[30:90, 60:140] = (0, 0, 255)  # a bright block appears against the background
    out = det.detect(moved)
    assert len(out) == 1
    assert out[0].bbox_xyxy == (60, 30, 140, 90)  # the block, in xyxy


def test_motion_reset_clears_the_background():
    det = MotionBlobDetector()
    det.detect(np.full((100, 200, 3), 100, np.uint8))
    det.reset()
    assert det._background is None
    # After reset the next frame is a first frame again: empty, background rebuilt.
    assert det.detect(np.full((100, 200, 3), 100, np.uint8)) == []


def test_motion_classifies_by_geometry_only():
    from ai.detect.stub import _class_from_geometry

    assert _class_from_geometry(200, 60) == "truck"  # wide and long
    assert _class_from_geometry(30, 80) == "motorcycle"  # tall and narrow
    assert _class_from_geometry(50, 50) == "other"  # an unremarkable box


# ------------------------------------------------------------ oracle & scripted wiring

def test_oracle_requires_a_truth_source():
    # No truth_for_envelope -> refuse at construction. An oracle bolted onto a live RTSP
    # source has no ground truth to read and must not silently return nothing.
    with pytest.raises(TypeError):
        OracleDetector(object())


def test_oracle_refuses_raw_pixels():
    # detect(pixels) has no truth to consult; the oracle works from the envelope only.
    det = OracleDetector(_TruthSource(_Truth(0, [])))
    with pytest.raises(TypeError):
        det.detect(np.zeros((10, 10, 3), np.uint8))


def test_oracle_unresolved_frame_is_empty_and_counted():
    class _NoTruth:
        def truth_for_envelope(self, envelope):  # noqa: ARG002
            return None

    det = OracleDetector(_NoTruth())
    assert det.detect_envelope(frame(0)) == []
    assert det.unresolved_frames == 1  # the miss is counted, not swallowed


def test_degradation_is_perfect_flag():
    assert OracleDegradation().is_perfect is True
    assert OracleDegradation(miss_rate=0.1).is_perfect is False


def test_degradation_confidence_ramp():
    # A width-dependent confidence: tiny boxes are trusted less than full-width ones.
    deg = OracleDegradation(confidence=0.9, min_confidence=0.1, confidence_full_width_px=100)
    assert deg.confidence_for(100) == 0.9  # at/over full width -> full confidence
    assert deg.confidence_for(0) == 0.1  # zero width -> the floor
    assert deg.confidence_for(50) == pytest.approx(0.5)  # linear between
    # With no ramp configured, width is irrelevant and confidence is flat.
    assert OracleDegradation().confidence_for(5) == 0.92


def test_scripted_replays_by_emitted_frame_index():
    script = {
        0: [DetectorResult((0, 0, 10, 10), "car", 0.2)],
        2: [DetectorResult((5, 5, 15, 15), "truck", 0.9)],
    }
    det = ScriptedDetector(script)
    assert [d.class_name for d in det.detect(blank())] == ["car"]  # frame 0
    assert det.detect(blank()) == []  # frame 1 has no script entry
    assert [d.class_name for d in det.detect(blank())] == ["truck"]  # frame 2
    assert det.ships is False  # a fixture never ships


def test_scripted_reset_rewinds_the_cursor():
    det = ScriptedDetector({0: [DetectorResult((0, 0, 10, 10), "car", 0.2)]})
    det.detect(blank())
    det.detect(blank())
    det.reset()
    assert len(det.detect(blank())) == 1  # frame 0 again


def test_scripted_warmup_does_not_advance_the_cursor():
    det = ScriptedDetector({0: [DetectorResult((0, 0, 10, 10), "car", 0.2)]})
    det.warmup()  # a no-op for a table replay -- must not consume frame 0
    assert len(det.detect(blank())) == 1


# --------------------------------- the factory: a config typo fails loud, ship gates

def test_registry_names_and_ship_gate():
    assert DETECTOR_NAMES == ("rfdetr", "motion", "oracle", "scripted")
    # Exactly one backend may appear in a published accuracy claim.
    assert SHIPPABLE_DETECTORS == frozenset({"rfdetr"})
    assert detector_ships("rfdetr") is True
    for name in ("motion", "oracle", "scripted"):
        assert detector_ships(name) is False


def test_factory_rejects_malformed_config():
    for cfg in (
        {"name": "nope"},  # unknown backend
        {"name": "motion", "bogus": 1},  # unknown key -> a silent typo caught
        {},  # no name at all
        "not-a-mapping",  # not even a config block
    ):
        with pytest.raises(DetectorConfigError):
            build_detector(cfg)


def test_factory_builds_each_named_backend():
    assert type(build_detector({"name": "scripted", "script": {}})).__name__ == "ScriptedDetector"
    assert type(build_detector({"name": "motion"})).__name__ == "MotionBlobDetector"
    rf = build_detector({"name": "rfdetr", "variant": "nano", "tile": False})
    assert rf.__class__.__name__ == "RFDETRDetector" and rf.ships is True


def test_scripted_and_oracle_require_their_inputs():
    with pytest.raises(DetectorConfigError):
        build_detector({"name": "scripted"})  # no script table
    with pytest.raises(DetectorConfigError):
        build_detector({"name": "oracle"})  # no truth source passed


def test_factory_wires_oracle_to_the_source():
    det = build_detector({"name": "oracle"}, source=_TruthSource(_Truth(0, [])))
    assert det.__class__.__name__ == "OracleDetector" and det.is_oracle is True


def test_publication_gate_refuses_non_shipping_backend():
    # normalize_detector_config(..., for_publication=True) is the guard that keeps a
    # motion/oracle number out of a submitted accuracy claim.
    normalize_detector_config({"name": "rfdetr"}, for_publication=True)  # shippable: ok
    with pytest.raises(DetectorConfigError):
        normalize_detector_config({"name": "motion"}, for_publication=True)


def test_two_distinct_ship_gates_licence_vs_publishable():
    # Two different questions, deliberately not the same answer:
    #   instance.ships       -- does the LICENCE permit submitting a run using it?
    #   detector_ships(name) -- may its ACCURACY appear in a published benchmark?
    # MotionBlobDetector separates them: MIT-licensed, so a pipeline using it can ship
    # (.ships True), but its per-class accuracy must never be quoted (detector_ships
    # False). Oracle and scripted fail both gates.
    motion = build_detector({"name": "motion"})
    assert motion.ships is True  # MIT licence permits shipping
    assert detector_ships("motion") is False  # but not publishing its accuracy

    scripted = build_detector({"name": "scripted", "script": {}})
    assert scripted.ships is False and detector_ships("scripted") is False

    rf = build_detector({"name": "rfdetr", "tile": False})
    assert rf.ships is True and detector_ships("rfdetr") is True


def test_resolve_allowed_classes():
    assert resolve_allowed_classes(["car", "truck"]) == frozenset({"car", "truck"})
    assert resolve_allowed_classes([]) is None  # empty -> no restriction
    assert resolve_allowed_classes(None) is None
    with pytest.raises(DetectorConfigError):
        resolve_allowed_classes(["van"])  # not a known vehicle class


def test_describe_is_static_and_backend_free():
    described = describe_detector({"name": "rfdetr", "tile": True})
    assert described["name"] == "rfdetr"
    assert described["ships"] is True
    assert described["tiled"] is True


def test_vehicle_taxonomy_maps_coco_and_drops_non_vehicles():
    # The six classes the system counts, and the COCO names that fold into them.
    assert VEHICLE_CLASSES == frozenset(
        {"car", "motorcycle", "bus", "truck", "auto_rickshaw", "other"}
    )
    assert map_class_name("car") == "car"
    assert map_class_name("bicycle") == "motorcycle"  # COCO bicycle -> motorcycle
    assert map_class_name("Auto Rickshaw") == "auto_rickshaw"  # normalised spacing/case
    # A non-vehicle is dropped (None), never coerced to "other" -- "other" is a vehicle.
    assert map_class_name("person") is None
    assert map_class_name("traffic_light") is None


def test_is_shippable_class():
    assert is_shippable_class("car") is True
    assert is_shippable_class("van") is False


def test_latency_summary_is_none_when_empty():
    # No frames timed -> no number, not a fake 0. Percentiles are nearest-rank over the
    # sorted samples (median of four samples is the upper-middle, by that convention).
    from ai.detect.base import _latency_summary

    assert _latency_summary([]) is None
    assert _latency_summary([1.0, 2.0, 3.0, 4.0]) == {
        "count": 4,
        "median": 3.0,
        "p95": 4.0,
        "max": 4.0,
    }


def test_colourfulness_and_darkness_extremes():
    from ai.detect.blobs import colourfulness, darkness

    grey = np.full((3, 3, 3), 100, np.uint8)
    assert int(colourfulness(grey).max()) == 0  # equal channels -> zero spread
    red = np.zeros((1, 1, 3), np.uint8)
    red[0, 0] = (255, 0, 0)
    assert int(colourfulness(red)[0, 0]) == 255
    assert int(darkness(np.full((1, 1, 3), 255, np.uint8))[0, 0]) == 0
    assert int(darkness(np.zeros((1, 1, 3), np.uint8))[0, 0]) == 255
    for bad in (np.zeros((3, 3), np.uint8), np.zeros((3, 3), np.uint8)):
        with pytest.raises(ValueError):
            colourfulness(bad)
        with pytest.raises(ValueError):
            darkness(bad)
