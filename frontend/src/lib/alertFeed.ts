import type { AdaptedAlert } from '../api/adapters'

// PURE. No React, no zustand, no copy strings. Both sources arrive as
// arguments so the whole thing is exercisable from a console.

// Same reason as the live store's caps: the demo laptop runs Postgres, Redis,
// FastAPI and a GPU worker at once, and an unbounded list becomes projector
// stutter that an audience attributes to the whole system rather than to one
// careless array.
export const ALERT_FEED_CAP = 100

export interface AlertFeed {
  alerts: AdaptedAlert[]
  duplicatesDropped: number
  droppedByCap: number
  // The ids still present in the merged feed. The caller prunes the sticky set
  // to these, so it cannot grow without bound across a long shift. Returned
  // rather than pruned here, because pruning is a side effect and this stays
  // pure.
  presentIds: ReadonlySet<string>
}

// locallyAcknowledged: ids the SERVER confirmed acknowledged this session.
// An alert whose id is in that set renders acknowledged regardless of what the
// socket said, because a frame carrying acknowledged:false that arrives after a
// confirmed 2xx predates the write -- it is stale, not a correction, and
// letting it win asserts the operator's action did not happen. D-038.
export function mergeAlertFeed(
  fetched: AdaptedAlert[],
  live: AdaptedAlert[],
  locallyAcknowledged: ReadonlySet<string> = new Set(),
): AlertFeed {
  const byId = new Map<string, AdaptedAlert>()

  // Live first, then fetched, so that on an alert_id collision the FETCHED
  // copy overwrites the socket copy. The server's version is authoritative:
  // the socket's may predate an update (an acknowledgement, a re-scored
  // match), and showing the stale one would contradict what the backend
  // currently holds.
  for (const alert of live) byId.set(alert.alert_id, alert)
  for (const alert of fetched) byId.set(alert.alert_id, alert)

  const unique = [...byId.values()]
  const duplicatesDropped = fetched.length + live.length - unique.length

  // Newest first by created_at, the only timestamp the Alert contract carries.
  // Parsed times, not string comparison, so an offset-bearing timestamp still
  // orders correctly.
  unique.sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))

  // Cap AFTER dedup and sort, so what survives is the newest hundred rather
  // than the first hundred that happened to arrive.
  const capped = unique.slice(0, ALERT_FEED_CAP)
  const droppedByCap = unique.length - capped.length

  // Applied last, so it survives dedup, sort and cap alike. Only ever forces
  // acknowledged TRUE: this set records confirmed writes, and it must never be
  // able to un-acknowledge something the server reported as acknowledged.
  const alerts = capped.map((alert) =>
    !alert.acknowledged && locallyAcknowledged.has(alert.alert_id)
      ? { ...alert, acknowledged: true }
      : alert,
  )

  // Pruning targets are the ids that SURVIVED the cap. An id dropped by the cap
  // is no longer renderable, so keeping its sticky flag buys nothing.
  const presentIds = new Set(alerts.map((alert) => alert.alert_id))

  return { alerts, duplicatesDropped, droppedByCap, presentIds }
}
