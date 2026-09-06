// One definition of "can this be plotted", used by the map, the map sidebar
// and the camera grid. There were three copies of this predicate; they agreed
// by luck, and the failure mode if one drifted was silent -- the map would
// count four unplaced cameras while the grid showed three, and neither screen
// would look wrong.
export interface MaybePlaced {
  lat: number | null
  lon: number | null
}

// Both coordinates or neither. A record with one of the two is not half
// placeable, it is unplaceable: Technical Master Plan E3, "do not invent
// coordinates."
//
// Tests for a finite NUMBER rather than "not null", which is what this used to
// do. Canonical 6.4 says lat/lon are `number | null`, and against that a null
// check is exactly right -- but the execution manual OMITS the keys instead,
// and `undefined !== null` is true, so absent coordinates sailed through and
// reached Leaflet as `L.latLng(undefined, undefined)`. Measured: that throws
// "Invalid LatLng object" during render and WHITE-SCREENS the whole app, on
// both the live map and search.
//
// NaN and Infinity are excluded for the same reason: a coordinate that is not
// a finite number cannot be plotted, however it arrived.
export function hasCoordinates<T extends MaybePlaced>(
  item: T,
): item is T & { lat: number; lon: number } {
  return Number.isFinite(item.lat) && Number.isFinite(item.lon)
}
