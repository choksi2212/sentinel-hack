import { create } from 'zustand'
import type { AdaptedAlert } from '../api/adapters'
import type { VehicleSighting } from '../types/api'
import type { WsStatus } from '../types/ui'

// Memory only. Nothing here is persisted: the guard bans localStorage and
// sessionStorage outright, and a demo that resurrects yesterday's alerts on
// load is worse than one that honestly starts empty.

// Caps are enforced HERE, not at the render layer. The demo laptop runs
// Postgres, Redis, FastAPI and a GPU worker at the same time; an unbounded
// array becomes projector stutter, and an audience attributes that stutter to
// the whole system rather than to one careless array.
export const ALERT_CAP = 100
export const SIGHTING_CAP = 50

export interface LiveState {
  wsStatus: WsStatus
  alerts: AdaptedAlert[]
  sightings: VehicleSighting[]
  lastHeartbeatAt: number | null
  // Local clock minus the heartbeat's ts. Recomputed on every beat. Arrival
  // time is never stamped as event time.
  clockSkewMs: number | null
  unknownMessageTypes: string[]
  malformedMessageCount: number
  reconnectCount: number

  // Ids the SERVER confirmed acknowledged, this session. Written only on a 2xx
  // from the acknowledge mutation, removed on rollback.
  //
  // Why it has to exist: a socket frame carrying acknowledged:false that arrives
  // AFTER a confirmed write predates that write. It is stale, not a correction,
  // and letting it win asserts the operator's action did not happen. D-038.
  //
  // Memory only. The storage rule bans localStorage and sessionStorage, and it
  // should: an acknowledgement that outlives the tab would be a claim about
  // server state made from a cache nobody reconciled.
  locallyAcknowledged: ReadonlySet<string>

  setWsStatus: (status: WsStatus) => void
  addAlert: (alert: AdaptedAlert) => void
  // Confirmed by the server. Sticky from here on.
  confirmAcknowledged: (alertId: string) => void
  // Rolled back. The write did not land, so the claim must not persist.
  clearAcknowledged: (alertId: string) => void
  // Drop ids no longer present in the merged feed, so the set cannot grow
  // without bound across a long shift.
  pruneAcknowledged: (presentIds: ReadonlySet<string>) => void
  // An alert that arrived only over the socket has no row in the REST list, so
  // an optimistic acknowledge written into the query cache alone would not show
  // on it. Takes the value explicitly rather than toggling, so the rollback
  // path restores a known state instead of assuming it can flip back.
  setAlertAcknowledged: (alertId: string, acknowledged: boolean) => void
  addSighting: (sighting: VehicleSighting) => void
  recordHeartbeat: (serverIso: string, receivedAt: number) => void
  recordUnknownType: (type: string) => void
  recordMalformed: () => void
  recordReconnect: () => void
  reset: () => void
}

export const useLiveStore = create<LiveState>()((set) => ({
  wsStatus: 'offline',
  alerts: [],
  sightings: [],
  lastHeartbeatAt: null,
  clockSkewMs: null,
  unknownMessageTypes: [],
  malformedMessageCount: 0,
  reconnectCount: 0,
  locallyAcknowledged: new Set<string>(),

  setWsStatus: (status) => set({ wsStatus: status }),

  confirmAcknowledged: (alertId) =>
    set((state) => {
      const next = new Set(state.locallyAcknowledged)
      next.add(alertId)
      return { locallyAcknowledged: next }
    }),

  clearAcknowledged: (alertId) =>
    set((state) => {
      if (!state.locallyAcknowledged.has(alertId)) return {}
      const next = new Set(state.locallyAcknowledged)
      next.delete(alertId)
      return { locallyAcknowledged: next }
    }),

  pruneAcknowledged: (presentIds) =>
    set((state) => {
      const next = new Set<string>()
      for (const id of state.locallyAcknowledged) {
        if (presentIds.has(id)) next.add(id)
      }
      // Returning the SAME set when nothing changed matters: this runs on every
      // merge, and a fresh Set each time would retrigger every subscriber and
      // re-run the merge that called it.
      if (next.size === state.locallyAcknowledged.size) return {}
      return { locallyAcknowledged: next }
    }),

  // Newest first, deduped by alert_id. A reconnect triggers a REST refetch and
  // the socket may also redeliver, so the same alert arrives twice by design.
  addAlert: (alert) =>
    set((state) => {
      const withoutDuplicate = state.alerts.filter(
        (existing) => existing.alert_id !== alert.alert_id,
      )
      return { alerts: [alert, ...withoutDuplicate].slice(0, ALERT_CAP) }
    }),

  setAlertAcknowledged: (alertId, acknowledged) =>
    set((state) => ({
      alerts: state.alerts.map((alert) =>
        alert.alert_id === alertId ? { ...alert, acknowledged } : alert,
      ),
    })),

  addSighting: (sighting) =>
    set((state) => {
      const withoutDuplicate = state.sightings.filter(
        (existing) => existing.sighting_id !== sighting.sighting_id,
      )
      return { sightings: [sighting, ...withoutDuplicate].slice(0, SIGHTING_CAP) }
    }),

  recordHeartbeat: (serverIso, receivedAt) =>
    set(() => {
      const serverMs = Date.parse(serverIso)
      return {
        lastHeartbeatAt: receivedAt,
        clockSkewMs: Number.isNaN(serverMs) ? null : receivedAt - serverMs,
      }
    }),

  // Once each. The list is for System Status to render later, so a schema
  // addition is visible rather than merely survived.
  recordUnknownType: (type) =>
    set((state) =>
      state.unknownMessageTypes.includes(type)
        ? state
        : { unknownMessageTypes: [...state.unknownMessageTypes, type] },
    ),

  recordMalformed: () =>
    set((state) => ({ malformedMessageCount: state.malformedMessageCount + 1 })),

  recordReconnect: () =>
    set((state) => ({ reconnectCount: state.reconnectCount + 1 })),

  reset: () =>
    set({
      wsStatus: 'offline',
      alerts: [],
      sightings: [],
      lastHeartbeatAt: null,
      clockSkewMs: null,
      unknownMessageTypes: [],
      malformedMessageCount: 0,
      reconnectCount: 0,
      locallyAcknowledged: new Set<string>(),
    }),
}))

// Slice selectors. A component reading wsStatus must not re-render when a
// sighting arrives, so each of these returns a primitive or a stable
// reference rather than a fresh object.
export const selectWsStatus = (state: LiveState): WsStatus => state.wsStatus
export const selectAlerts = (state: LiveState): AdaptedAlert[] => state.alerts
export const selectLocallyAcknowledged = (state: LiveState): ReadonlySet<string> =>
  state.locallyAcknowledged
export const selectSightings = (state: LiveState): VehicleSighting[] =>
  state.sightings
export const selectClockSkewMs = (state: LiveState): number | null =>
  state.clockSkewMs
export const selectLastHeartbeatAt = (state: LiveState): number | null =>
  state.lastHeartbeatAt
export const selectReconnectCount = (state: LiveState): number =>
  state.reconnectCount
export const selectMalformedCount = (state: LiveState): number =>
  state.malformedMessageCount
export const selectUnknownTypes = (state: LiveState): string[] =>
  state.unknownMessageTypes
