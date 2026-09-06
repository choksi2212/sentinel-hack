import type { AdaptedJourney, AdaptedJourneySegment } from '../api/adapters'
import type { VehicleSighting } from '../types/api'

// Pure. No React, no Leaflet, no rendering, no copy strings -- the copy for a
// gap lives with the component that draws it, because this module has no
// business knowing what a gap looks like or what it says.

// Three distinct facts, not two. All three render the same string on screen
// per D-018, because the renderer cannot tell an operator anything useful by
// distinguishing them. The distinction is for System Status and for debugging:
// a counter labelled "ambiguous" that fires when nothing was ambiguous is the
// same defect as the required_speed_kmh counter that fired on absence -- a
// counter reporting one thing under the name of another.
export type ConnectorSlot =
  | { kind: 'connector'; segment: AdaptedJourneySegment }
  | { kind: 'gap'; reason: 'no_segment' | 'ambiguous' | 'no_time_match' }

export interface JourneyTimeline {
  // journey.sightings, already sorted by readJourney, passed through unchanged.
  nodes: VehicleSighting[]
  // EXACTLY nodes.length - 1 entries, or 0 when nodes.length <= 1.
  // connectors[i] sits between nodes[i] and nodes[i + 1].
  connectors: ConnectorSlot[]
  // Segments no pair consumed. Surfaced, never dropped: silence about a
  // segment the server sent is an assertion by omission.
  unplacedSegments: AdaptedJourneySegment[]
  // Several candidates survived the time window and we refused to pick one.
  ambiguousPairCount: number
  // Candidates existed by camera identity but none fell inside the window.
  noTimeMatchPairCount: number
}

// Matched by CAMERA IDENTITY, never by array position. Position is what makes
// sorted nodes disagree with unsorted connectors, and that disagreement draws
// a journey that never happened. readJourney also discards unusable segments,
// so indices stop lining up after the first discard anyway.
export function buildJourneyTimeline(journey: AdaptedJourney): JourneyTimeline {
  const nodes = journey.sightings

  // A segment is consumed by at most one pair. Without this, a single A->B
  // segment would serve every repetition of A->B in the same journey and the
  // screen would claim the backend assessed traversals it never saw.
  const consumed = new Set<number>()
  const connectors: ConnectorSlot[] = []
  let ambiguousPairCount = 0
  let noTimeMatchPairCount = 0

  for (let i = 0; i + 1 < nodes.length; i += 1) {
    const from = nodes[i]
    const to = nodes[i + 1]
    // noUncheckedIndexedAccess: both are T | undefined. The loop bound makes
    // this unreachable, but it is checked rather than asserted.
    if (from === undefined || to === undefined) {
      connectors.push({ kind: 'gap', reason: 'no_segment' })
      continue
    }

    const candidates: number[] = []
    journey.segments.forEach((segment, index) => {
      if (consumed.has(index)) return
      if (
        segment.from_camera_id === from.camera_id &&
        segment.to_camera_id === to.camera_id
      ) {
        candidates.push(index)
      }
    })

    if (candidates.length === 0) {
      connectors.push({ kind: 'gap', reason: 'no_segment' })
      continue
    }

    if (candidates.length === 1) {
      const index = candidates[0]
      const segment = index === undefined ? undefined : journey.segments[index]
      if (index === undefined || segment === undefined) {
        connectors.push({ kind: 'gap', reason: 'no_segment' })
        continue
      }
      consumed.add(index)
      connectors.push({ kind: 'connector', segment })
      continue
    }

    // More than one candidate. Narrow by time: keep those whose from_time
    // falls inside the inclusive interval spanned by this pair.
    //
    // Lower bound is the FIRST node's first_seen_at, not its last_seen_at,
    // because a traversal can begin the moment the vehicle is first observed
    // at the departure camera -- the sighting is an interval, and departure
    // is not required to wait for the end of it. Upper bound is the second
    // node's last_seen_at for the mirror-image reason: arrival is anywhere
    // within the arrival sighting's interval.
    //
    // Parsed times, not string comparison, so an offset-bearing timestamp
    // still orders correctly.
    const lower = Date.parse(from.first_seen_at)
    const upper = Date.parse(to.last_seen_at)

    const inWindow = candidates.filter((index) => {
      const segment = journey.segments[index]
      if (segment === undefined) return false
      const at = Date.parse(segment.from_time)
      if (Number.isNaN(at) || Number.isNaN(lower) || Number.isNaN(upper)) {
        return false
      }
      return at >= lower && at <= upper
    })

    if (inWindow.length === 1) {
      const index = inWindow[0]
      const segment = index === undefined ? undefined : journey.segments[index]
      if (index === undefined || segment === undefined) {
        connectors.push({ kind: 'gap', reason: 'no_segment' })
        continue
      }
      consumed.add(index)
      connectors.push({ kind: 'connector', segment })
      continue
    }

    // Nothing is attached either way, and nothing is consumed, so every
    // candidate still surfaces as unplaced. But the two outcomes are
    // different facts and are counted separately.
    if (inWindow.length === 0) {
      // Candidates matched by camera identity, none by time. Not ambiguity --
      // there was nothing to be ambiguous between.
      noTimeMatchPairCount += 1
      connectors.push({ kind: 'gap', reason: 'no_time_match' })
    } else {
      // Several survived. Attaching any of them would be a guess about which
      // traversal the backend actually assessed. A wrong connector is a false
      // assertion; a gap is an honest one.
      ambiguousPairCount += 1
      connectors.push({ kind: 'gap', reason: 'ambiguous' })
    }
  }

  const unplacedSegments = journey.segments.filter(
    (_segment, index) => !consumed.has(index),
  )

  return {
    nodes,
    connectors,
    unplacedSegments,
    ambiguousPairCount,
    noTimeMatchPairCount,
  }
}
