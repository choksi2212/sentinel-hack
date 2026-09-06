import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { useEffect, useRef } from 'react'
import { basemapConfig } from './basemap'

// The URL and attribution moved to basemap.ts when the offline source landed.
// Online OSM is still the default; VITE_BASEMAP=offline selects the local one.

// [lat, lon] for Leaflet -- note this is the opposite order from MapLibre's
// [lon, lat], and the opposite of GeoJSON. Ahmedabad.
const CENTRE: L.LatLngExpression = [23.02, 72.57]
const ZOOM = 11

export interface MapInstance {
  containerRef: React.RefObject<HTMLDivElement | null>
  mapRef: React.RefObject<L.Map | null>
  // Journey connectors. Added to the map BEFORE the marker group so it sits
  // underneath: a connector must never cover the sighting it joins.
  connectorLayerRef: React.RefObject<L.LayerGroup | null>
  layerRef: React.RefObject<L.LayerGroup | null>
}

// Leaflet owns the DOM inside the container and React must never render into
// it. The map is created once, held in a ref, and everything after creation is
// imperative -- the same ownership model the MapLibre version used.
export function useMapInstance(): MapInstance {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<L.Map | null>(null)
  const connectorLayerRef = useRef<L.LayerGroup | null>(null)
  const layerRef = useRef<L.LayerGroup | null>(null)

  useEffect(() => {
    const container = containerRef.current
    if (container === null) return

    const map = L.map(container, {
      center: CENTRE,
      zoom: ZOOM,
      // The audience watches a projector; a scroll wheel nudging the map
      // mid-demo is a hazard, and the keyboard handler steals arrow keys.
      scrollWheelZoom: true,
      attributionControl: true,
    })

    // Read once at creation. Switching source at runtime would mean tearing
    // down and rebuilding the map, and nothing asks for that -- the source is a
    // deployment decision, fixed for the life of the page.
    const basemap = basemapConfig()
    L.tileLayer(basemap.url, {
      attribution: basemap.attribution,
      maxZoom: basemap.maxZoom,
      // Undefined on OSM, set on OFFLINE. Leaflet treats undefined as "leave
      // the tile transparent", which is the right answer for a network failure
      // and the wrong one for a tile that was never seeded.
      errorTileUrl: basemap.errorTileUrl,
    }).addTo(map)

    // Order is fixed at creation, bottom to top: basemap, journey connectors,
    // then markers. Camera markers, when they arrive, belong between these two
    // so a busy junction cannot hide its own camera under a sighting.
    const connectorLayer = L.layerGroup().addTo(map)
    // Sighting markers, cleared and refilled on data change so the map itself
    // is never rebuilt.
    const layer = L.layerGroup().addTo(map)

    mapRef.current = map
    connectorLayerRef.current = connectorLayer
    layerRef.current = layer

    return () => {
      layerRef.current = null
      connectorLayerRef.current = null
      mapRef.current = null
      map.remove()
    }
  }, [])

  return { containerRef, mapRef, connectorLayerRef, layerRef }
}
