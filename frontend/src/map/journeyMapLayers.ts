import type { AdaptedJourneySegment } from '../api/adapters'
import { hasCoordinates } from '../lib/geo'
import type { JourneyTimeline } from '../lib/journeyTimeline'
import type { VehicleSighting } from '../types/api'
import type { FeasibilityState } from './journeyLineStyle'
import { feasibilityState } from './journeyLineStyle'

// PURE. No React, no Leaflet instance, no DOM. Coordinates and counts only,
// so the projection is verifiable in the smoke console rather than by looking
// at a screenshot and believing it.

// A sighting that is known to carry both coordinates. Structurally still a
// VehicleSighting, so consumers expecting VehicleSighting[] are satisfied,
// but MapCanvas does not have to re-narrow before calling fitBounds.
export type PlacedSighting = VehicleSighting & { lat: number; lon: number }

export interface DrawableConnector {
  from: { lat: number; lon: number }
  to: { lat: number; lon: number }
  state: FeasibilityState
  segment: AdaptedJourneySegment
}

export interface JourneyMapLayers {
  plottedNodes: PlacedSighting[]
  unplacedNodeCount: number
  drawableConnectors: DrawableConnector[]
  // A connector whose endpoint sighting has no coordinates. It is NOT drawn
  // partially, and the camera's position is NOT substituted for the
  // sighting's -- a line to a guessed endpoint asserts a route between two
  // places, one of which we do not know.
  undrawableConnectorCount: number
  gapCount: number
}

export function buildJourneyMapLayers(timeline: JourneyTimeline): JourneyMapLayers {
  // The shared predicate from src/lib/geo.ts. Not a fourth copy.
  const plottedNodes = timeline.nodes.filter(hasCoordinates)
  const unplacedNodeCount = timeline.nodes.length - plottedNodes.length

  const drawableConnectors: DrawableConnector[] = []
  let undrawableConnectorCount = 0
  let gapCount = 0

  timeline.connectors.forEach((slot, index) => {
    if (slot.kind === 'gap') {
      gapCount += 1
      return
    }

    // A JourneySegment carries no coordinates at all. A connector's endpoints
    // come from the two SIGHTINGS it joins, never from the segment, and
    // connectors[i] joins nodes[i] to nodes[i + 1].
    const from = timeline.nodes[index]
    const to = timeline.nodes[index + 1]

    if (
      from === undefined ||
      to === undefined ||
      !hasCoordinates(from) ||
      !hasCoordinates(to)
    ) {
      undrawableConnectorCount += 1
      return
    }

    drawableConnectors.push({
      from: { lat: from.lat, lon: from.lon },
      to: { lat: to.lat, lon: to.lon },
      state: feasibilityState(slot.segment.feasible),
      segment: slot.segment,
    })
  })

  return {
    plottedNodes,
    unplacedNodeCount,
    drawableConnectors,
    undrawableConnectorCount,
    gapCount,
  }
}
