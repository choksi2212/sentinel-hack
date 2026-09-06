import L from 'leaflet'
import { useEffect } from 'react'
import { hasCoordinates } from '../lib/geo'
import type { MatchState, VehicleSighting } from '../types/api'
// Path ends at journeyLineStyle with no extension, which is what the guard's
// IMPORTS_LINE_STYLE regex requires. Adding ".ts" here fails the build.
import { journeyLineStyle } from './journeyLineStyle'
import type { DrawableConnector, JourneyMapLayers } from './journeyMapLayers'
import { useMapInstance } from './useMapInstance'

// Colour by match_state, never by raw confidence. Canonical 4.4: these
// confidences are relative evidence, not probabilities, so a gradient keyed on
// one would imply a precision we do not have. Same four values as the design
// tokens in src/styles/theme.css.
const MATCH_STATE_COLOR: Record<MatchState, string> = {
  exact: '#0B6E4F',
  probable: '#A15C00',
  low_confidence: '#8C3A2B',
  unreadable: '#55656B',
}

export function MapCanvas({
  sightings,
  journey,
  connectorPopup,
}: {
  sightings: VehicleSighting[]
  // Absent means no journey layer at all, and LiveMap behaves exactly as it
  // did before this prop existed.
  journey?: JourneyMapLayers
  // The page supplies the popup, so no copy string lives in src/map.
  connectorPopup?: (connector: DrawableConnector) => HTMLElement
}) {
  const { containerRef, mapRef, connectorLayerRef, layerRef } = useMapInstance()

  // Imperative update. The map is never recreated for new data; the marker
  // group is emptied and refilled.
  useEffect(() => {
    const layer = layerRef.current
    if (layer === null) return

    layer.clearLayers()

    // Sightings with a null lat or lon are excluded outright, never styled
    // differently and never nudged onto the map. Technical Master Plan E3:
    // "do not invent coordinates." The count of what was excluded is shown
    // beside the map by the caller, so the omission is visible.
    for (const sighting of sightings.filter(hasCoordinates)) {
      L.circleMarker([sighting.lat, sighting.lon], {
        radius: 6,
        color: '#FFFFFF',
        weight: 1.5,
        fillColor: MATCH_STATE_COLOR[sighting.match_state],
        fillOpacity: 1,
      })
        .bindTooltip(
          // Never a guess and never a blank: a plate we could not read reads
          // "Unreadable".
          `${sighting.plate ?? 'Unreadable'} — ${sighting.camera_name}`,
        )
        .addTo(layer)
    }
  }, [sightings, layerRef])

  // Journey connectors, updated imperatively in their own effect. The map is
  // still created exactly once, in useMapInstance's empty-deps effect; this
  // only refills a layer group that already exists, and clears it on cleanup.
  useEffect(() => {
    const connectorLayer = connectorLayerRef.current
    const map = mapRef.current
    if (connectorLayer === null) return

    connectorLayer.clearLayers()
    if (journey === undefined) return

    for (const connector of journey.drawableConnectors) {
      // Options come from journeyLineStyle and nothing is merged over them.
      // Every state it returns is dashed; a solid line would assert a route
      // nobody observed.
      const line = L.polyline(
        [
          [connector.from.lat, connector.from.lon],
          [connector.to.lat, connector.to.lon],
        ],
        journeyLineStyle(connector.state),
      )
      if (connectorPopup !== undefined) {
        line.bindPopup(connectorPopup(connector))
      }
      line.addTo(connectorLayer)
    }

    // Only when there is something to fit. fitBounds on an empty set throws.
    if (map !== null && journey.plottedNodes.length > 0) {
      map.fitBounds(
        L.latLngBounds(
          journey.plottedNodes.map((node) => [node.lat, node.lon] as [number, number]),
        ),
        { padding: [32, 32], maxZoom: 15 },
      )
    }

    return () => {
      connectorLayer.clearLayers()
    }
  }, [journey, connectorPopup, connectorLayerRef, mapRef])

  return <div ref={containerRef} className="h-full w-full" />
}
