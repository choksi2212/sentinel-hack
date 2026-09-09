"""ai/real_predictor -- the module benchmarks/run.py imports instead of the stub.

Two rules from docs/manuals/akshat/MONDAY.md decide whether the resulting report
is a measurement or a rehearsal, and both are pinned here:

  1. The prediction comes from pixels, never from row["plate_text"]. Checked
     structurally (AST, so a docstring quoting the rule does not pass for
     obeying it) and behaviourally (the label and the OCR reading are
     deliberately different strings in every fixture below, so a leak shows up
     as a wrong answer rather than a suspiciously good one).
  2. Frames are grouped for fusion by TrackKey and ordered by source_pts_ms.
     Never by obs_id order -- which happens to agree with pts order in today's
     index and is not the contract.

Nothing here loads paddle, cv2, or the dataset. The subprocess boundary is the
seam: _run_ocr_batch is replaced by a fake engine, so every branch this module
owns -- the xywh->xyxy conversion, the width-floor refusal, the fusion floor,
the cache, the eligibility decision -- is exercised on a machine with no models
installed. What cannot be tested here is whether PP-OCRv4 reads a plate; that is
what the run itself measures.
"""

import ast
import json
from pathlib import Path

import pytest

from ai import real_predictor as rp
from ai.fusion.consensus import MIN_FUSION_WEIGHT

TRUTH = "GJ01AB1234"
# Every fixture's label. The fake engine never returns this unless a test asks
# it to, so any predict() that answers TRUTH without the engine having read it
# is reading the label column.
NEAR_MISS = "GJ01AB1284"
JUNK = "XX00XX0000"


def make_row(
    obs_id,
    *,
    frame=None,
    pts=0,
    camera="cam_a",
    session="sess_a",
    track=1,
    bbox=(0, 0, 120, 30),
    eligible=True,
    source="synthetic_plates",
):
    """One TRINETRA-HARD row, with the fields schema.json marks required that
    this module actually touches. plate_text is present and is the truth,
    exactly as it is in the real index -- the point is that predict ignores it."""
    return {
        "obs_id": obs_id,
        "frame_path": frame if frame is not None else f"datasets/raw/x.png#{obs_id}",
        "source_pts_ms": pts,
        "camera_id": camera,
        "stream_session_id": session,
        "track_id": track,
        "plate_bbox": list(bbox),
        "plate_width_px": float(bbox[2]),
        "width_bucket": ">100",
        "eligible": eligible,
        "plate_text": TRUTH,
        "label_source": "synthetic_truth",
        "source_dataset": source,
        "plate_bbox_source": "rendered",
        "track_type": "fixed_distance",
        "slice": "easy",
    }


def reading(text, confidence, quality, *, variant="raw", agreement=1.0):
    return {
        "text": text,
        "confidence": confidence,
        "image_quality": quality,
        "variant": variant,
        "agreement": agreement,
    }


REFUSED = {"text": None, "confidence": None, "image_quality": 0.0}


class FakeEngine:
    """Stands in for the .venv-ocr subprocess.

    Replaces _run_ocr_batch rather than subprocess.run, because the protocol
    boundary worth testing is "which frames were asked for, with which box" --
    not JSON serialisation. Records every batch so a test can assert the corpus
    was read once, and that indian_road frames were never asked for at all.
    """

    def __init__(self):
        self.readings: dict[str, dict] = {}
        self.batches: list[list[dict]] = []

    def __call__(self, requests):
        self.batches.append(requests)
        by_file = {str(rp._frame_file(fp)): r for fp, r in self.readings.items()}
        return {req["path"]: by_file.get(req["path"], dict(REFUSED)) for req in requests}

    @property
    def frames_asked(self):
        return [req["path"] for batch in self.batches for req in batch]


@pytest.fixture
def engine(monkeypatch, tmp_path):
    """real_predictor wired to a fake engine and a throwaway cache directory.

    The cache redirect is not hygiene, it is correctness: benchmarks/cache holds
    the readings of a real run, and a test suite that overwrote them would make
    the next report a mixture of measured and invented frames.
    """
    fake = FakeEngine()
    monkeypatch.setattr(rp, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rp, "CACHE_FRAMES_DIR", tmp_path / "frames")
    monkeypatch.setattr(rp, "CACHE_FILE", tmp_path / "staged_readings.json")
    monkeypatch.setattr(rp, "_run_ocr_batch", fake)
    monkeypatch.setattr(rp, "_materialize_frames", lambda frame_paths: None)
    rp.reset()
    yield fake
    rp.reset()


# --- rule 1: the prediction comes from pixels ---------------------------------


def test_never_reads_the_label_column():
    """MONDAY.md checklist item 3, checked on the AST rather than the text.

    A grep for 'plate_text' matches this module's own docstring, where the rule
    is quoted. Walking the tree instead asks the only question that matters:
    does any expression in this file subscript or .get() the label column.
    """
    tree = ast.parse(Path(rp.__file__).read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            key = node.slice
            if isinstance(key, ast.Constant) and key.value == "plate_text":
                offenders.append(node.lineno)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "plate_text"
        ):
            offenders.append(node.lineno)
    assert not offenders, (
        f"ai/real_predictor.py reads the label column at line(s) {offenders}. "
        "The report would then measure the labels, not the pipeline."
    )


def test_prediction_is_the_engines_reading_not_the_label(engine):
    """The engine reads NEAR_MISS; the row's plate_text says TRUTH. A predictor
    that returns TRUTH here is reading the label, and the whole benchmark is
    void."""
    row = make_row("th_1")
    engine.readings[row["frame_path"]] = reading(NEAR_MISS, 0.9, 0.9)
    rp.set_rows([row])

    assert rp.predict(row, fusion_enabled=False) == NEAR_MISS
    assert rp.predict(row, fusion_enabled=True) == NEAR_MISS


def test_unread_frame_is_none_not_a_guess(engine):
    """The width floor refusing a 20 px plate and no variant finding characters
    both arrive here as text: None. Both are the pipeline's answer, and neither
    licenses falling back to the label or to the track's other frames."""
    row = make_row("th_1")
    engine.readings[row["frame_path"]] = dict(REFUSED)
    rp.set_rows([row])

    assert rp.predict(row, fusion_enabled=False) is None
    assert rp.predict(row, fusion_enabled=True) is None


def test_predictions_align_one_to_one_with_rows(engine):
    """MONDAY.md checklist item 1: one entry per row, always -- including rows
    with no pixels and rows nothing could be read from."""
    rows = [
        make_row("th_1", frame="a", pts=0),
        make_row("th_2", frame="b", pts=40),
        make_row(
            "th_3",
            frame="c.tar#clip_0001.jpg",
            pts=80,
            source="indian_road",
            camera="cam_road",
            session="sess_road",
        ),
    ]
    engine.readings["a"] = reading(NEAR_MISS, 0.9, 0.9)
    rp.set_rows(rows)

    predictions = {r["obs_id"]: rp.predict(r, True) for r in rows}
    assert len(predictions) == len(rows)
    assert predictions["th_3"] is None


# --- rule 2: group by TrackKey, order by source_pts_ms ------------------------


def test_fusion_groups_by_track_key_not_by_row_order(engine):
    """Two tracks interleaved in the index. Each row must get its own track's
    consensus; a predictor that walked rows in order and reset on a boundary it
    inferred from obs_id would cross the two."""
    rows = [
        make_row("th_1", frame="a1", pts=0, camera="cam_a", session="sess_a"),
        make_row("th_2", frame="b1", pts=0, camera="cam_b", session="sess_b"),
        make_row("th_3", frame="a2", pts=40, camera="cam_a", session="sess_a"),
        make_row("th_4", frame="b2", pts=40, camera="cam_b", session="sess_b"),
    ]
    for frame in ("a1", "a2"):
        engine.readings[frame] = reading(NEAR_MISS, 0.9, 0.9)
    for frame in ("b1", "b2"):
        engine.readings[frame] = reading(JUNK, 0.9, 0.9)
    rp.set_rows(rows)

    assert [rp.predict(r, True) for r in rows] == [NEAR_MISS, JUNK, NEAR_MISS, JUNK]


def test_same_track_id_in_two_sessions_is_two_tracks(engine):
    """The track-merge bug the TrackKey exists to prevent. track_id is reused
    across sessions -- in the real corpus every synthetic row carries track_id 1
    -- so a key missing stream_session_id merges unrelated vehicles and reports
    one confident plate for both."""
    rows = [
        make_row("th_1", frame="a", session="sess_a", track=1),
        make_row("th_2", frame="b", session="sess_b", track=1),
    ]
    engine.readings["a"] = reading(NEAR_MISS, 0.9, 0.9)
    engine.readings["b"] = reading(JUNK, 0.9, 0.9)
    rp.set_rows(rows)

    assert rp.predict(rows[0], True) == NEAR_MISS
    assert rp.predict(rows[1], True) == JUNK
    assert rp.diagnostics()["tracks"] == 2
    assert rp.diagnostics()["tracks_with_plate"] == 2


def test_fused_answer_does_not_depend_on_row_order(engine):
    """Fusion is aligned on source_pts_ms, so shuffling the index must not move
    the answer. If it does, the ordering came from somewhere else."""
    frames = {"f0": 0, "f1": 40, "f2": 80}
    engine.readings["f0"] = reading(NEAR_MISS, 0.9, 0.9)
    engine.readings["f1"] = reading(TRUTH, 0.5, 0.6)
    engine.readings["f2"] = reading(TRUTH, 0.5, 0.6)

    forward = [make_row(f"th_{i}", frame=f, pts=p) for i, (f, p) in enumerate(frames.items())]
    rp.set_rows(forward)
    answer = rp.predict(forward[0], True)

    rp.reset()
    rp.set_rows(list(reversed(forward)))
    assert rp.predict(forward[0], True) == answer


def test_frame_index_is_derived_from_pts_at_25_fps():
    """schema.json carries no frame_index, so the observation's frame identity
    comes from source_pts_ms and the generator's 25 fps -- which the corpus
    confirms: consecutive frames are 40 ms apart."""
    assert rp._frame_index(0) == 0
    assert rp._frame_index(40) == 1
    assert rp._frame_index(80) == 2
    assert rp._frame_index(1000) == 25


def test_observed_at_is_timezone_aware_and_reproducible():
    """Two runs over the same frozen corpus must produce the same observations,
    so observed_at is anchored on pts rather than on the wallclock."""
    first = rp._observed_at(40)
    assert first == rp._observed_at(40)
    assert first.endswith("+00:00")
    assert rp._observed_at(0) != rp._observed_at(40)


# --- the staged stages actually run -------------------------------------------


def test_consensus_beats_the_best_single_frame(engine):
    """Why fusion is in the pipeline at all, and proof this routes through
    ai/fusion rather than picking a maximum locally.

    One sharp frame reads NEAR_MISS at weight 0.81; three weaker frames agree on
    TRUTH at 0.30 each. Highest-confidence-single-frame -- which is what
    benchmarks/paddle_predictor.py does -- answers NEAR_MISS. Weighted evidence
    share answers TRUTH, 0.90 to 0.81.
    """
    rows = [make_row(f"th_{i}", frame=f"f{i}", pts=40 * i) for i in range(4)]
    engine.readings["f0"] = reading(NEAR_MISS, 0.9, 0.9)
    for frame in ("f1", "f2", "f3"):
        engine.readings[frame] = reading(TRUTH, 0.5, 0.6)
    rp.set_rows(rows)

    assert rp.predict(rows[0], fusion_enabled=False) == NEAR_MISS
    assert rp.predict(rows[0], fusion_enabled=True) == TRUTH
    assert rp.diagnostics()["fusion_changed_tracks"] == 1


def test_readings_below_the_fusion_floor_do_not_vote(engine):
    """MIN_FUSION_WEIGHT is the per-frame eligibility floor: a reading the OCR
    engine and the quality scorer between them rated worthless is not evidence,
    and letting a pile of them agree is how a track of unreadable frames
    manufactures a confident plate.

    Two frames at weight 0.04 agree on JUNK (0.08 total) against one real read
    at 0.11. On raw weight JUNK wins; both junk frames are below the floor, so
    they never vote.
    """
    rows = [make_row(f"th_{i}", frame=f"f{i}", pts=40 * i) for i in range(3)]
    engine.readings["f0"] = reading(TRUTH, 0.8, 0.14)
    engine.readings["f1"] = reading(JUNK, 0.2, 0.2)
    engine.readings["f2"] = reading(JUNK, 0.2, 0.2)
    assert 0.2 * 0.2 < MIN_FUSION_WEIGHT <= 0.8 * 0.14 + 0.001
    rp.set_rows(rows)

    assert rp.predict(rows[0], True) == TRUTH
    assert rp.diagnostics()["observations_below_fusion_floor"] == 2


def test_bbox_is_read_as_xywh_and_handed_over_as_xyxy(engine):
    """The row's plate_bbox is [x, y, w, h]; every stage contract is xyxy. The
    corpus's boxes all start at x=0, which is what makes the slip survivable
    long enough to reach a report: plate_width_px would equal x2 and look right,
    while ai/quality scored resolution against the wrong number."""
    assert rp._xyxy([10, 20, 30, 40]) == (10, 20, 40, 60)

    row = make_row("th_1", bbox=(10, 5, 100, 25))
    engine.readings[row["frame_path"]] = reading(TRUTH, 0.9, 0.9)
    rp.set_rows([row])
    observation = rp._observation_for(row)

    assert observation.plate_bbox_xyxy == (10, 5, 110, 30)
    assert observation.plate_width_px == 100
    assert observation.fusion_weight == pytest.approx(0.81)


def test_the_worker_is_given_the_rows_own_box(engine):
    """paddle_predictor sends box: None and OCRs the whole materialised file.
    This predictor sends the row's box, which is what puts the plate's scene
    width in front of MIN_OCR_PLATE_WIDTH_PX -- a 27 px row has to be refused
    for being 27 px, not read anyway because the file was handed over whole."""
    row = make_row("th_1", bbox=(0, 0, 27, 9))
    rp.set_rows([row])

    assert len(engine.batches) == 1
    assert engine.batches[0][0]["box"] == [0, 0, 27, 9]


# --- eligibility: the one field that decides whether fabrication is measured ---


def test_ineligible_row_is_answered_from_pixels_by_default(engine):
    """An ineligible row is one the generator rendered unreadable. The pipeline
    does not know that, so if it reads a plate there anyway, the report should
    say so: that is what fabrication_count is for, and scorer.py excludes these
    rows from n_eligible entirely, so this cannot flatter the accuracy rate.

    Suppressing here would drive fabrication_count to 0 by construction, for
    every predictor, forever.
    """
    assert rp.RETURN_NONE_ON_INELIGIBLE is False
    row = make_row("th_1", eligible=False)
    engine.readings[row["frame_path"]] = reading(NEAR_MISS, 0.9, 0.9)
    rp.set_rows([row])

    assert rp.predict(row, fusion_enabled=False) == NEAR_MISS
    assert rp.predict(row, fusion_enabled=True) == NEAR_MISS


def test_ineligible_row_is_suppressed_when_the_guard_is_on(engine, monkeypatch):
    """The other mode, kept so the suppressed number can be produced for
    comparison rather than argued about."""
    monkeypatch.setattr(rp, "RETURN_NONE_ON_INELIGIBLE", True)
    row = make_row("th_1", eligible=False)
    engine.readings[row["frame_path"]] = reading(NEAR_MISS, 0.9, 0.9)
    rp.set_rows([row])

    assert rp.predict(row, fusion_enabled=False) is None
    assert rp.predict(row, fusion_enabled=True) is None
    assert rp.diagnostics()["return_none_on_ineligible"] is True


def test_fusion_carries_consensus_onto_ineligible_frames(engine):
    """403 of the corpus's 600 synthetic tracks mix eligible and ineligible
    frames, so fusion-on necessarily fabricates more than fusion-off: the
    track's plate lands on every row of the track. That trade is the finding,
    and it has to be visible rather than smoothed away."""
    rows = [
        make_row("th_1", frame="good", pts=0, eligible=True),
        make_row("th_2", frame="dark", pts=40, eligible=False),
    ]
    engine.readings["good"] = reading(NEAR_MISS, 0.9, 0.9)
    engine.readings["dark"] = dict(REFUSED)
    rp.set_rows(rows)

    assert rp.predict(rows[1], fusion_enabled=False) is None
    assert rp.predict(rows[1], fusion_enabled=True) == NEAR_MISS


# --- rows with no pixels -------------------------------------------------------


def test_indian_road_rows_are_never_read(engine):
    """They point inside a .tar shard with no loader, and scorer.py drops
    unverified_real rows before eligibility is considered -- they contribute to
    no reported number, not accuracy and not fabrication. Reading them would
    cost the run an hour and change nothing."""
    rows = [
        make_row("th_1", frame="synth", source="synthetic_plates"),
        make_row(
            "th_2",
            frame="shard.tar#clip_0001.jpg",
            source="indian_road",
            camera="cam_road",
            session="sess_road",
            track=9,
        ),
    ]
    engine.readings["synth"] = reading(NEAR_MISS, 0.9, 0.9)
    rp.set_rows(rows)

    assert engine.frames_asked == [str(rp._frame_file("synth"))]
    assert rp.predict(rows[1], fusion_enabled=False) is None
    assert rp.predict(rows[1], fusion_enabled=True) is None


# --- set_rows: the contract around it ------------------------------------------


def test_fusion_without_set_rows_raises(engine):
    """A track's consensus is undefined until every frame of it has been read.
    Answering from one frame and labelling it fusion would report a number for
    a stage that did not run -- so this is loud, not silently degraded."""
    rp.reset()
    with pytest.raises(RuntimeError, match="set_rows"):
        rp.predict(make_row("th_1"), fusion_enabled=True)


def test_fusion_off_needs_no_preload(engine):
    """The mirror of the rule above: with no fusion there is no track, so a
    missing preload is a miss, not an error."""
    rp.reset()
    assert rp.predict(make_row("th_1"), fusion_enabled=False) is None


def test_a_filtered_row_set_cannot_reach_outside_itself(engine):
    """run.py's --track-type hands over a subset. A per-bucket report must not
    borrow evidence from frames outside its own bucket, so the index is exactly
    the rows given."""
    inside = make_row("th_1", frame="a", camera="cam_a", session="sess_a")
    outside = make_row("th_2", frame="b", camera="cam_b", session="sess_b")
    engine.readings["a"] = reading(NEAR_MISS, 0.9, 0.9)
    engine.readings["b"] = reading(JUNK, 0.9, 0.9)
    rp.set_rows([inside])

    assert rp.predict(inside, True) == NEAR_MISS
    assert rp.predict(outside, True) is None
    assert engine.frames_asked == [str(rp._frame_file("a"))]


def test_one_frame_named_by_several_rows_is_read_once(engine):
    """Distinct rows can name the same rendered frame. Re-reading identical
    pixels is the one genuinely expensive thing this module does."""
    shared = "datasets/raw/x.png#frame0"
    rows = [make_row("th_1", frame=shared, pts=0), make_row("th_2", frame=shared, pts=40)]
    engine.readings[shared] = reading(NEAR_MISS, 0.9, 0.9)
    rp.set_rows(rows)

    assert engine.frames_asked == [str(rp._frame_file(shared))]
    assert rp.predict(rows[1], fusion_enabled=False) == NEAR_MISS


def test_reset_clears_the_index(engine):
    row = make_row("th_1")
    engine.readings[row["frame_path"]] = reading(NEAR_MISS, 0.9, 0.9)
    rp.set_rows([row])
    rp.reset()

    assert rp.predict(row, fusion_enabled=False) is None
    with pytest.raises(RuntimeError):
        rp.predict(row, fusion_enabled=True)


# --- the reading cache ---------------------------------------------------------


def test_cached_readings_are_reused_across_runs(engine):
    """6,822 frames is the expensive half of a run. A second process must not
    pay for it again."""
    row = make_row("th_1")
    engine.readings[row["frame_path"]] = reading(NEAR_MISS, 0.9, 0.9)
    rp.set_rows([row])
    rp.reset()
    rp.set_rows([row])

    assert len(engine.batches) == 1
    assert rp.predict(row, fusion_enabled=False) == NEAR_MISS


def test_cache_is_invalidated_when_the_engine_changes(engine, monkeypatch):
    """The readings depend on the engine and on the worker driving it, not just
    on the frames. Without this, changing the variant list or the width floor and
    re-running would silently re-report yesterday's numbers."""
    row = make_row("th_1")
    engine.readings[row["frame_path"]] = reading(NEAR_MISS, 0.9, 0.9)
    rp.set_rows([row])
    rp.reset()

    monkeypatch.setattr(rp, "OCR_VERSION", "PP-OCRv5 (hypothetical)")
    rp.set_rows([row])
    assert len(engine.batches) == 2


def test_cache_miss_on_a_frame_absent_from_the_cache(engine):
    """A cache built for one --track-type run must not be treated as complete
    for a wider one."""
    first = make_row("th_1", frame="a")
    second = make_row("th_2", frame="b")
    engine.readings["a"] = reading(NEAR_MISS, 0.9, 0.9)
    engine.readings["b"] = reading(JUNK, 0.9, 0.9)
    rp.set_rows([first])
    rp.reset()
    rp.set_rows([first, second])

    assert len(engine.batches) == 2
    assert rp.predict(second, fusion_enabled=False) == JUNK
    # the first run's reading survived the second batch rather than being dropped
    assert rp.predict(first, fusion_enabled=False) == NEAR_MISS


def test_use_cache_false_forces_a_reread(engine):
    row = make_row("th_1")
    engine.readings[row["frame_path"]] = reading(NEAR_MISS, 0.9, 0.9)
    rp.set_rows([row])
    rp.reset()
    rp.set_rows([row], use_cache=False)

    assert len(engine.batches) == 2


def test_a_corrupt_cache_file_is_a_reread_not_a_crash(engine, tmp_path):
    """A half-written cache from an interrupted run must cost a re-read, not the
    whole benchmark."""
    rp.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    rp.CACHE_FILE.write_text("{not json", encoding="utf-8")
    row = make_row("th_1")
    engine.readings[row["frame_path"]] = reading(NEAR_MISS, 0.9, 0.9)
    rp.set_rows([row])

    assert rp.predict(row, fusion_enabled=False) == NEAR_MISS


# --- what the report records ---------------------------------------------------


def test_diagnostics_reports_what_the_stages_did(engine):
    """predict returns a bare string by contract, so this is the only place a
    run can record confidence, quality, which preprocessing variant won, and how
    much fusion changed. MONDAY.md's `diagnostics` block."""
    rows = [make_row(f"th_{i}", frame=f"f{i}", pts=40 * i) for i in range(3)]
    engine.readings["f0"] = reading(TRUTH, 0.8, 0.5, variant="upscale_2x")
    engine.readings["f1"] = reading(TRUTH, 0.6, 0.5, variant="upscale_2x")
    engine.readings["f2"] = dict(REFUSED)
    rp.set_rows(rows)

    diagnostics = rp.diagnostics()
    assert diagnostics["frames_read"] == 3
    assert diagnostics["frames_with_text"] == 2
    assert diagnostics["frames_unread"] == 1
    assert diagnostics["variant_wins"] == {"upscale_2x": 2}
    assert diagnostics["mean_ocr_confidence"] == pytest.approx(0.7)
    assert diagnostics["mean_image_quality"] == pytest.approx(0.5)
    assert diagnostics["tracks"] == 1
    assert diagnostics["tracks_with_plate"] == 1
    assert diagnostics["min_fusion_weight"] == MIN_FUSION_WEIGHT


def test_diagnostics_on_an_empty_run_is_null_not_zero(engine):
    """A mean over nothing is unknown, not 0.0 -- the same rule the width buckets
    follow in scorer.py. Reporting 0.0 would read as "the engine is broken"."""
    rp.set_rows([])
    diagnostics = rp.diagnostics()

    assert diagnostics["frames_read"] == 0
    assert diagnostics["mean_ocr_confidence"] is None
    assert diagnostics["mean_image_quality"] is None


def test_diagnostics_counts_worker_errors_separately(engine):
    """A frame that raised inside the worker is not the same as a frame the
    engine declined to read, and a run where thousands raised should not look
    like a run where the plates were unreadable."""
    row = make_row("th_1")
    engine.readings[row["frame_path"]] = {
        "text": None, "confidence": None, "image_quality": 0.0,
        "error": "cannot read frame",
    }
    rp.set_rows([row])

    assert rp.diagnostics()["frame_errors"] == 1
    assert rp.diagnostics()["frames_with_text"] == 0


def test_weights_info_declines_to_hash_absent_weights(monkeypatch, tmp_path):
    """No hash is the honest answer when the weights are not on disk. The note
    has to say which model would have been hashed and why only that one."""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    sha, note = rp.weights_info()

    assert sha is None
    assert rp.REC_MODEL_DIR in note
    assert "rec_only" in note


def test_weights_info_hashes_the_recogniser_only(monkeypatch, tmp_path):
    """The staged engine never loads PP-OCRv4's detector, so a hash covering it
    would claim provenance over a model that did not run -- and would compare
    equal across two runs that differed in the only model either used."""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    rec = tmp_path / ".paddlex" / "official_models" / rp.REC_MODEL_DIR
    rec.mkdir(parents=True)
    (rec / "inference.pdmodel").write_bytes(b"rec-weights")
    det = tmp_path / ".paddlex" / "official_models" / "PP-OCRv4_mobile_det"
    det.mkdir(parents=True)
    (det / "inference.pdmodel").write_bytes(b"det-weights")

    first, note = rp.weights_info()
    assert first is not None
    assert "1 recogniser file" in note

    # changing the detector must not move the hash; changing the recogniser must
    (det / "inference.pdmodel").write_bytes(b"det-weights-v2")
    assert rp.weights_info()[0] == first
    (rec / "inference.pdmodel").write_bytes(b"rec-weights-v2")
    assert rp.weights_info()[0] != first


def test_reading_for_exposes_the_per_frame_detail(engine):
    """The failure taxonomy needs per-frame confidence and quality to tell
    ocr_wrong from plate_too_small, and predict() cannot carry them."""
    row = make_row("th_1")
    engine.readings[row["frame_path"]] = reading(NEAR_MISS, 0.75, 0.5)
    rp.set_rows([row])

    detail = rp.reading_for(row["frame_path"])
    assert detail["confidence"] == 0.75
    assert detail["image_quality"] == 0.5
    assert rp.reading_for("never/materialised") is None


def test_missing_venv_is_a_named_manual_step(monkeypatch, tmp_path):
    """The isolated venv is a manual step on a fresh machine. A FileNotFoundError
    naming it beats a subprocess failure whose message is a Windows error code."""
    monkeypatch.setattr(rp, "VENV_PYTHON", tmp_path / "nonexistent" / "python.exe")
    with pytest.raises(FileNotFoundError, match="MANUAL STEP"):
        rp._run_ocr_batch([{"path": "x.png", "box": [0, 0, 100, 20]}])


def test_no_requests_means_no_subprocess(monkeypatch, tmp_path):
    """An empty batch must not spawn a process, or a --track-type with no rows
    would fail on a machine that has no .venv-ocr."""
    monkeypatch.setattr(rp, "VENV_PYTHON", tmp_path / "nonexistent" / "python.exe")
    assert rp._run_ocr_batch([]) == {}


def test_sanitize_never_produces_a_windows_illegal_name():
    """Fixed-distance frame_paths embed the width_bucket names '>100' and '<30',
    and cv2.imwrite fails on those characters by returning False with no
    exception -- which is how those two buckets once lost every frame silently.
    The mapping matches paddle_predictor's on purpose, so both predictors read
    the one materialised frame directory."""
    illegal = set('<>:"|?*')
    for frame_path in (
        "datasets/raw/synthetic_plates/generated/x.png#fixed_>100_0000_frame0",
        "datasets/raw/synthetic_plates/generated/x.png#fixed_<30_0000_frame0",
        "datasets/raw/synthetic_plates/generated/Kerala/private/KL61AVY6032.png#frame0",
    ):
        assert not (illegal & set(rp._sanitize(frame_path)))
    assert rp._sanitize("a/b.png#frame0") == "a_b.png__frame0"


def test_cache_file_is_not_shared_with_the_paddle_baseline():
    """Same frames, different engine, different fields. One file for both would
    mean whichever predictor ran last decided what the other one measured."""
    assert rp.CACHE_FILE.name != "ocr_readings.json"
    assert rp.CACHE_FILE.parent == rp.CACHE_DIR


def test_cache_file_is_written_with_its_provenance(engine):
    """A cache with no record of what produced it cannot be invalidated, and an
    un-invalidatable cache is how a stale number survives a code change."""
    row = make_row("th_1")
    engine.readings[row["frame_path"]] = reading(NEAR_MISS, 0.9, 0.9)
    rp.set_rows([row])

    blob = json.loads(rp.CACHE_FILE.read_text(encoding="utf-8"))
    assert blob["meta"]["ocr_version"] == rp.OCR_VERSION
    assert blob["meta"]["min_fusion_weight"] == MIN_FUSION_WEIGHT
    assert row["frame_path"] in blob["readings"]
