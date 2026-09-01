"""Turn a finished track's evidence into one EventEnvelope. Stage 13 of the 14.

This is the join point. Everything upstream produces per-frame observations; everything
downstream -- ingest, search, GIS, watchlist -- consumes exactly one row per vehicle
sighting. That narrowing happens here and nowhere else, which is why the rules about what
may and may not be asserted in an event are enforced in this file rather than trusted to
callers.

Three of those rules are worth stating at the top because they are the ones a reasonable
person gets wrong:

**plate: null is a correct event.** A vehicle passed and could not be identified. That is
real information -- it belongs in a count of vehicles seen, it proves the camera was
working, and it is honest. The alternative is emitting the pipeline's best guess, which
attaches a real registration number, belonging to a real person, to a place and time they
were never at. Unlike a null that error is invisible: it looks exactly like a correct
event. Contracts section 3.2.

**Confidences are never multiplied.** Detector confidence, fused-plate evidence share and
image quality are three uncalibrated numbers on three different scales. Their product is
not a probability of anything, and presenting one invites a question that cannot be
answered. Each travels in its own field. Contracts section 4.4.

**event_id is minted once, here.** It is Mihir's idempotency key, so it has to survive
every POST retry unchanged -- if a retry mints a new one the same sighting lands twice and
every count in the system is wrong by an amount nobody can measure. Minting it at build
time rather than at send time is what makes the retry in http_sink.py safe.
"""

from typing import Any, Optional, Sequence

from ai.contracts.enums import MATCH_STATES, SOURCE_MODES
from ai.contracts.event import (
    EventEnvelope,
    EvidenceBlock,
    ModelProvenance,
    PlateBlock,
    VehicleBlock,
)
from ai.contracts.ids import new_event_id
from ai.contracts.stages import FusedPlate, PlateObservation
from ai.fusion.accumulator import CropBuffer, TrackCrop
from ai.fusion.consensus import fuse_observations
from ai.normalize.matching import match_state_for

# Fallback when a track produced no vehicle box at all. Should not happen -- note_track
# records a box on the first frame -- but a zero box is a visibly wrong value that shows
# up in the UI as a degenerate rectangle, whereas a guessed box looks plausible and is
# not. Preferring the obviously-broken output over the plausibly-broken one is the same
# choice plate: null makes.
_EMPTY_BBOX = (0, 0, 0, 0)


class EventBuildError(ValueError):
    """The evidence cannot produce a valid event.

    Raised rather than returning None, because every caller in the pipeline treats a
    finished track as something that must be accounted for. Silently dropping one turns
    a bug into a slightly-lower vehicle count, which is the hardest class of error to
    notice: the number is still a number.
    """


def observations_from_buffer(
    buffer: CropBuffer,
    reads: Sequence[tuple[TrackCrop, str, float]],
) -> list[PlateObservation]:
    """Build the PlateObservation list fusion consumes, from (crop, text, confidence).

    Kept separate from build_event so the audit trail is constructible on its own: these
    rows are what land in plate_observations, and they are the evidence that proves the
    consensus. An event whose plate cannot be traced back to the frames that produced it
    is an assertion, not a finding.

    **The filter is on empty text, not on unnormalizable text**, and the distinction is
    the one the schema draws. A read of "!!" is an observation: OCR looked at the plate
    and returned characters that are not a registration number, which is what the
    "unreadable" match_state means and why the contract requires a non-empty plate.raw
    alongside it. A read of "" is not an observation -- nothing was returned, so there is
    nothing to audit and nothing to report.

    Getting that line wrong in the other direction is what dropping unnormalizable reads
    did on the first pass here: it discarded the only text an unreadable event could put
    in raw, and left build_event with a choice between an invalid event and a fabricated
    string. fuse() skips non-normalizing reads on its own, so keeping them costs nothing
    and they belong in total_observations -- evidence_count over total_observations is
    read as "how unanimous was this", and a garbage read genuinely is dissent.
    """
    observations: list[PlateObservation] = []
    for crop, text, confidence in reads:
        if not text:
            continue
        observations.append(
            PlateObservation(
                camera_id=buffer.track_key.camera_id,
                stream_session_id=buffer.track_key.stream_session_id,
                track_id=buffer.track_key.track_id,
                plate_bbox_xyxy=crop.candidate.plate_bbox_xyxy,
                plate_width_px=crop.plate_width_px,
                plate_raw=text,
                ocr_confidence=float(confidence),
                # The plate crop's own quality, not the frame's. A sharp plate on a
                # blurry frame is good evidence and the reverse is not, so weighting by
                # frame quality would let a clean background vouch for an unreadable
                # plate.
                image_quality=float(crop.quality),
                frame_index=crop.frame_index,
                pts_ms=crop.pts_ms,
                observed_at=crop.observed_at,
            )
        )
    return observations


def build_event(
    buffer: CropBuffer,
    *,
    source_mode: str,
    model: ModelProvenance,
    observations: Sequence[PlateObservation] = (),
    fused: Optional[FusedPlate] = None,
    exact_watchlist_hit: bool = False,
    snapshot_uri: Optional[str] = None,
    plate_crop_uri: Optional[str] = None,
    event_id: Optional[str] = None,
) -> EventEnvelope:
    """One finished track -> one EventEnvelope.

    fused may be passed in when the caller has already run consensus (the worker does,
    because it needs the FusedPlate for logging before it decides to emit). Otherwise it
    is computed here from observations. Passing both is fine and the passed-in one wins;
    passing a fused plate that disagrees with the observations is not detectable here and
    is the caller's problem.

    observed_at is the track's LAST observation, not its first. A sighting is reported at
    the moment the evidence was complete: reporting the first frame would timestamp the
    event before the plate that identifies it was ever read, and a journey reconstructed
    from first-frame timestamps has vehicles arriving before they were identified.
    """
    if source_mode not in SOURCE_MODES:
        raise EventBuildError(
            f"source_mode {source_mode!r} is not one of {list(SOURCE_MODES)}; "
            f"ingest has a CHECK constraint on this and would return 422"
        )

    track_key = buffer.track_key
    if fused is None and observations:
        fused = fuse_observations(observations)

    observed_at = buffer.last_observed_at or buffer.first_observed_at
    if not observed_at:
        raise EventBuildError(
            f"{track_key} has no observed_at; note_track was never called for it, so "
            f"this buffer holds crops for a track that was never seen"
        )
    pts_ms = buffer.last_pts_ms if buffer.last_pts_ms is not None else 0

    vehicle = VehicleBlock(
        type=buffer.vehicle_class,
        confidence=float(buffer.best_vehicle_confidence),
        bbox_xyxy=buffer.best_vehicle_bbox or _EMPTY_BBOX,
    )

    plate_block = _plate_block(buffer, observations, fused, exact_watchlist_hit)

    envelope = EventEnvelope(
        event_id=event_id or new_event_id(),
        camera_id=track_key.camera_id,
        stream_session_id=track_key.stream_session_id,
        track_id=track_key.track_id,
        observed_at=observed_at,
        source_pts_ms=int(pts_ms),
        source_mode=source_mode,
        vehicle=vehicle,
        # The best plate crop's quality when there is one, else the best frame's. These
        # are different measurements and the field carries whichever was actually used
        # to weight the plate decision, so that a reader comparing image_quality against
        # plate confidence is comparing the two numbers that were multiplied inside
        # fuse() -- and not one of them against an unrelated third.
        image_quality=float(buffer.best_frame_quality),
        model=model,
        plate=plate_block,
        evidence=EvidenceBlock(
            snapshot_uri=snapshot_uri,
            plate_crop_uri=plate_crop_uri,
        ),
    )

    errors = envelope.validate()
    if errors:
        # Validating before returning rather than trusting the assembly above. The
        # backend would reject this with a 422 either way; failing here names the field,
        # in the process that produced it, instead of surfacing as an HTTP status in a
        # retry loop two modules away.
        raise EventBuildError(
            f"built an invalid event for {track_key}: " + "; ".join(errors)
        )
    return envelope


def _plate_block(
    buffer: CropBuffer,
    observations: Sequence[PlateObservation],
    fused: Optional[FusedPlate],
    exact_watchlist_hit: bool,
) -> Optional[PlateBlock]:
    """The plate block, or None for a genuinely unidentified vehicle.

    Four outcomes, and the schema draws the line between them in a specific place:

        no plate was ever located          -> None. Nothing to report.
        located, OCR returned nothing      -> None. Nothing to put in raw.
        located, characters but no plate    -> a block with match_state "unreadable".
        located and read                   -> a block with the fused answer.

    The third case is the one worth having. "A plate was visible, OCR returned GJ0 and
    that is not a registration number" is a different statement from "no plate was
    visible", and only the first is a measure of OCR performance.

    The second case collapses into the first, because plate.raw is required to be a
    non-empty string whenever a plate block is present, and raw is defined as exactly
    what OCR returned. When OCR returned nothing there is no raw, and inventing one to
    preserve the distinction would be the precise failure this module exists to prevent
    -- a fabricated string in the field whose whole purpose is to be unfabricated.

    Nothing is actually lost there. "Located but no read at all" is a pipeline statistic,
    not a claim about a vehicle, so it belongs in ai/metrics.py's counters where the
    worker can see it against the number of crops that were offered. And it does not
    affect the primary metric: eligible vehicle events are eligible by ground truth, not
    by what the pipeline managed to emit, so the denominator never depended on this.
    """
    if not buffer.crops:
        return None

    best_crop = buffer.crops[0]

    if fused is None or not fused.normalized:
        if not observations:
            # Located, but OCR returned nothing at all. See the docstring: there is no
            # raw to report, so this is plate: null.
            return None
        # Located, and OCR returned characters that are not a registration number. raw
        # carries the highest-weight attempt, so the failure is inspectable rather than
        # merely counted -- knowing OCR returned "GJ0" on a plate it could not resolve is
        # a diagnostic; knowing only that it failed is not.
        best_attempt = max(observations, key=lambda o: o.fusion_weight)
        return PlateBlock(
            raw=best_attempt.plate_raw,
            normalized=None,
            # Zero, not the OCR confidence of the discarded attempt. A confidence on a
            # plate this event is declaring unreadable would be a number about nothing,
            # and the one place it could be read as meaningful -- ranking unreadable
            # events by how nearly they were read -- is not a thing anyone should do.
            confidence=0.0,
            match_state="unreadable",
            plate_width_px=best_attempt.plate_width_px,
            # How many reads back the claim that this plate could not be read. Not zero:
            # the schema requires at least one, and it is right to require it, because a
            # single blurry attempt and eleven consistent failures are different amounts
            # of evidence for the same conclusion.
            evidence_count=len(observations),
            bbox_xyxy=best_attempt.plate_bbox_xyxy,
        )

    best = fused.best_observation
    match_state = match_state_for(
        fused.normalized,
        fused.evidence_count,
        fused.confidence,
        exact_watchlist_hit,
    )
    if match_state not in MATCH_STATES:  # pragma: no cover - defensive
        raise EventBuildError(f"derived match_state {match_state!r} is not a valid value")

    return PlateBlock(
        # raw is exactly what OCR returned on the winning frame, unmodified. Keeping it
        # alongside the normalized form is what makes it possible to report how often OCR
        # was right against how often the grammar rules rescued or ruined it -- if only
        # the normalized string survived, every I-to-1 correction would be invisible and
        # the OCR stage would take credit for normalization's work.
        raw=best.plate_raw if best is not None else fused.normalized,
        normalized=fused.normalized,
        confidence=float(fused.confidence),
        match_state=match_state,
        # From the winning observation, not the best crop. The width bucket has to
        # describe the frame the answer came from, or the per-bucket accuracy table is
        # measuring the wrong thing -- a track whose best crop was 110 px but whose
        # winning read came from a 60 px frame would credit the >100 bucket for a read it
        # did not produce.
        plate_width_px=best.plate_width_px if best is not None else best_crop.plate_width_px,
        evidence_count=fused.evidence_count,
        bbox_xyxy=best.plate_bbox_xyxy if best is not None else best_crop.candidate.plate_bbox_xyxy,
    )


def winning_crop(
    buffer: CropBuffer, fused: Optional[FusedPlate]
) -> Optional[TrackCrop]:
    """The crop that produced the asserted plate, or the best one if none did.

    Matched back by frame_index, because the operator confirming a hit should be looking at
    the image the answer came from. Showing the highest-*quality* crop instead would put a
    crisp photograph next to a plate string that a different, blurrier frame produced --
    which reads as corroboration and is the opposite: it invites someone to confirm a read
    against evidence that did not make it.

    Falls back to the best crop when there is no fused plate (nothing was read, so there is
    no winning frame and the sharpest view is the most useful thing to show) or when the
    match fails.
    """
    if not buffer.crops:
        return None
    if fused is not None and fused.best_observation is not None:
        target = fused.best_observation.frame_index
        for crop in buffer.crops:
            if crop.frame_index == target:
                return crop
    return buffer.crops[0]


def build_event_with_evidence(
    buffer: CropBuffer,
    *,
    source_mode: str,
    model: ModelProvenance,
    snapshots: Any,
    observations: Sequence[PlateObservation] = (),
    fused: Optional[FusedPlate] = None,
    exact_watchlist_hit: bool = False,
    event_id: Optional[str] = None,
) -> EventEnvelope:
    """build_event, plus the two evidence stills, in the one order that works.

    The stills are named after the event, and the event carries their URIs, so there is a
    dependency in both directions. It resolves in exactly one sequence -- mint the event_id,
    commit the stills under it, then build the event with both the id and the URIs -- and
    getting it wrong is not loudly wrong: minting a second id for the event leaves the
    files orphaned under the first, and the event points at nothing while two JPEGs sit on
    disk belonging to a sighting that does not exist. This function exists so no caller has
    to remember that.

    `snapshots` is a SnapshotWriter or a NullSnapshotWriter; the null one returns
    (None, None) and both URIs end up null, which is a valid event.

    A commit failure is not an event failure. Both URIs go null and the sighting is emitted
    regardless: an unwritable JPEG is a storage problem, and dropping the vehicle over it
    would turn a missing photograph into a missing observation.
    """
    if fused is None and observations:
        fused = fuse_observations(observations)

    minted = event_id or new_event_id()
    crop = winning_crop(buffer, fused)
    snapshot_uri, plate_crop_uri = snapshots.commit(
        buffer.track_key,
        event_id=minted,
        observed_at=buffer.last_observed_at or buffer.first_observed_at,
        plate_crop_bgr=crop.crop_bgr if crop is not None else None,
    )
    return build_event(
        buffer,
        source_mode=source_mode,
        model=model,
        observations=observations,
        fused=fused,
        exact_watchlist_hit=exact_watchlist_hit,
        snapshot_uri=snapshot_uri,
        plate_crop_uri=plate_crop_uri,
        event_id=minted,
    )


def build_events(
    buffers: Sequence[CropBuffer],
    *,
    source_mode: str,
    model: ModelProvenance,
    reads_for: Any,
    exact_watchlist_hit_for: Any = None,
) -> list[EventEnvelope]:
    """Build events for several finished tracks, skipping none silently.

    reads_for(buffer) -> sequence of (crop, text, confidence). Injected rather than run
    here because OCR is the expensive stage and the worker batches it differently in live
    and offline modes; this function's job is the bookkeeping, not the scheduling.

    A buffer that raises is re-raised, not skipped. See EventBuildError.
    """
    events: list[EventEnvelope] = []
    for buffer in buffers:
        observations = observations_from_buffer(buffer, reads_for(buffer))
        hit = bool(exact_watchlist_hit_for(buffer)) if exact_watchlist_hit_for else False
        events.append(
            build_event(
                buffer,
                source_mode=source_mode,
                model=model,
                observations=observations,
                exact_watchlist_hit=hit,
            )
        )
    return events
