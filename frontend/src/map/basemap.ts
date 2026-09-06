// Basemap source selection.
//
// The online OSM raster source stays and stays DEFAULT. The offline source is
// selectable, never a replacement: a change that breaks the working online map
// to gain an offline one is a net loss, and the demo machine may well have
// network on the day.
//
// Selected by VITE_BASEMAP, which is not a secret -- VITE_ variables compile
// into the public bundle, and this one is a two-value switch, not a key.

export type BasemapSource = 'osm' | 'offline'

export interface BasemapConfig {
  url: string
  attribution: string
  maxZoom: number
  // Leaflet renders nothing for a 404 and leaves the tile transparent, so the
  // page background shows through and the map reads as half-rendered rather
  // than as missing data -- on a projector that looks like a bug in the app.
  //
  // Set on OFFLINE only, and it substitutes a flat neutral tile so the gap
  // reads as deliberately empty. It does NOT label the tile: a viewport is
  // thirty to forty tiles across, and thirty copies of the words "no tile" is
  // worse noise than the blank it replaces. OSM leaves this undefined, because
  // a missing tile there means the network died mid-demo, which drill 4 in
  // docs/demo/RECOVERY_DRILLS.md handles by saying so out loud.
  errorTileUrl?: string
}

// The OSM public tile servers. Fine for interactive use, which is what this is;
// see basemap.README in WORKLOG for why they are NOT a source of bulk tiles.
const OSM: BasemapConfig = {
  url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  attribution:
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  maxZoom: 19,
}

// Served from public/, which Vite copies verbatim into dist/ and does NOT
// bundle, so this costs the JS bundle nothing at all.
//
// maxZoom 16 is measured, not assumed: the map opens at z11 and fitBounds caps
// at z15, so 16 is one level of headroom for a manual scroll-zoom. z17 upward
// is deliberately absent -- the demo bbox needs 2,982 tiles at z17 and 47,094
// at z19, against 1,086 for everything up to z16.
//
// ATTRIBUTION IS NOT OPTIONAL and must match whatever was actually seeded.
// The default below assumes an OSM-derived source, which every practical
// option is. If tiles come from somewhere else, this string changes with them.
// A flat tile in the sunken paper tone with a faint diagonal hatch. Inline as a
// data URI rather than a file in public/, so it cannot itself 404 -- a fallback
// that can fail to load is not a fallback. '#' is percent-encoded because an
// unescaped one ends the URI.
const MISSING_TILE =
  'data:image/svg+xml;utf8,' +
  '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256">' +
  '<rect width="256" height="256" fill="%23eef1f2"/>' +
  '<path d="M-64 192L192 -64M0 256L256 0M64 320L320 64" ' +
  'stroke="%23dfe4e6" stroke-width="6"/></svg>'

const OFFLINE: BasemapConfig = {
  url: '/tiles/{z}/{x}/{y}.png',
  attribution:
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &middot; offline tiles',
  maxZoom: 16,
  errorTileUrl: MISSING_TILE,
}

export function basemapSource(): BasemapSource {
  return import.meta.env.VITE_BASEMAP === 'offline' ? 'offline' : 'osm'
}

export function basemapConfig(source: BasemapSource = basemapSource()): BasemapConfig {
  return source === 'offline' ? OFFLINE : OSM
}
