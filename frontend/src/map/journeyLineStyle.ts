import type { PolylineOptions } from 'leaflet'

// The single source of truth for connector appearance, with two consumers:
// Leaflet on the map and SVG in the timeline. Both derive from the one table
// below, so the two halves of the Journey screen cannot drift apart.
export type FeasibilityState = 'plausible' | 'not_plausible' | 'unassessed'

interface ConnectorStyle {
  readonly color: string
  readonly weight: number
  readonly dashArray: string
}

// EVERY state carries a dashArray, and there is no branch, parameter or
// override anywhere in this module that produces a solid line or an empty
// dash. Cameras cover a fraction of a percent of road-km, so everything
// between two of them is inference. A solid line asserts a route nobody
// observed, and in a police tool that invites an operational decision taken
// on a line we drew.
//
// The three states differ in colour and weight only. Frozen so a caller
// cannot reach in and delete the dash after the fact.
const CONNECTOR: Readonly<Record<FeasibilityState, ConnectorStyle>> = Object.freeze({
  // --color-ink-2. The ordinary case, quiet but legible.
  plausible: Object.freeze({ color: '#45565D', weight: 2, dashArray: '6 6' }),
  // --color-lowconf, rust. Heavier dash and heavier weight: this is the one
  // an analyst must look at.
  not_plausible: Object.freeze({ color: '#8C3A2B', weight: 4, dashArray: '10 5' }),
  // --color-unknown, neutral grey. Sparser dash so it reads as quieter than
  // plausible rather than as a second kind of warning.
  unassessed: Object.freeze({ color: '#94A3A8', weight: 2, dashArray: '3 6' }),
})

// Leaflet consumer. No polyline is constructed here -- this module returns
// options and nothing else.
export function journeyLineStyle(state: FeasibilityState): PolylineOptions {
  const style = CONNECTOR[state]
  return {
    color: style.color,
    weight: style.weight,
    dashArray: style.dashArray,
  }
}

// SVG consumer, for the timeline. Same table, different attribute names.
export function journeyLineSvgProps(state: FeasibilityState): {
  stroke: string
  strokeWidth: number
  strokeDasharray: string
} {
  const style = CONNECTOR[state]
  return {
    stroke: style.color,
    strokeWidth: style.weight,
    strokeDasharray: style.dashArray,
  }
}

// Maps the adapter's three-state feasibility onto the style table. `undefined`
// means the backend never assessed the segment, which is a distinct state from
// assessed-and-passed and must never collapse into it.
export function feasibilityState(feasible: boolean | undefined): FeasibilityState {
  if (feasible === undefined) return 'unassessed'
  return feasible ? 'plausible' : 'not_plausible'
}
