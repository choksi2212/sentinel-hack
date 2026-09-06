import type { QueryClient } from '@tanstack/react-query'
import { wsAlerts } from '../api/endpoints'
import { readAlert } from '../api/adapters'
import type { VehicleSighting } from '../types/api'
import { useLiveStore } from './liveStore'

// 30 seconds of silence force-closes the socket. Two missed 15s heartbeats.
// Never set this below 25s or a single late beat starts a reconnect storm.
//
// Why a watchdog at all: a socket held open by a NAT with nothing flowing is
// indistinguishable, from the readyState alone, from a quiet night. A UI
// showing "connected" while zero data arrives is the same category of lie as
// labelling replay ONLINE.
const WATCHDOG_MS = 30_000
const MAX_BACKOFF_MS = 30_000
// After this many failures we stop claiming to be reconnecting and say
// offline. We keep retrying regardless -- the demo laptop's wifi drops.
const ATTEMPTS_BEFORE_OFFLINE = 6

interface Envelope {
  type?: unknown
  data?: unknown
  payload?: unknown
  ts?: unknown
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function socketUrl(): string {
  const base = import.meta.env.VITE_WS_BASE_URL
  // Path comes from endpoints.ts. Never hand-written here.
  return `${base}${wsAlerts()}`
}

// Module-level, so a second call while one connection is live is a no-op.
// StrictMode runs effects twice in development; a naive connect opens two
// sockets, doubles every message, and makes the caps and the dedup
// untestable -- the bug would only show as "the numbers are wrong".
let activeSocket: WebSocket | null = null
let connectionCount = 0
let disposed = false

// Exposed for the smoke check in Part 6c. Reading it is how "exactly one
// socket under StrictMode" gets proven rather than asserted.
export function liveSocketCount(): number {
  return connectionCount
}

export function connectLiveSocket(queryClient: QueryClient): () => void {
  if (activeSocket !== null) {
    // Already connected or connecting. Hand back a disconnect that does
    // nothing, so the second StrictMode cleanup cannot tear down the first
    // effect's live socket.
    return () => {}
  }

  disposed = false
  let attempt = 0
  let socket: WebSocket | null = null
  let watchdog: number | null = null
  let retry: number | null = null
  // The first open is a connection, not a reconnection.
  let hasConnectedOnce = false
  // Reset on every open. Until a frame arrives we do not claim to be
  // connected, and the backoff keeps escalating.
  let sawTrafficSinceOpen = false

  function clearWatchdog(): void {
    if (watchdog !== null) {
      window.clearTimeout(watchdog)
      watchdog = null
    }
  }

  // Reset on ANY traffic, not only heartbeats. A busy socket delivering
  // sightings every 3s is demonstrably alive even if a beat is dropped.
  function armWatchdog(): void {
    clearWatchdog()
    const armedAt = Date.now()
    watchdog = window.setTimeout(() => {
      // readyState is logged BEFORE the close, because the whole claim under
      // test is that this socket was still OPEN and simply silent. If it reads
      // CLOSED/CLOSING here then onclose already ran and did the reconnecting;
      // the watchdog would be taking credit for someone else's work.
      const dead = socket
      const state = dead === null ? 'no socket' : String(dead.readyState)
      console.warn(
        `[ws] WATCHDOG fired: no traffic for ${Date.now() - armedAt}ms ` +
          `(threshold ${WATCHDOG_MS}ms), readyState before close = ${state} ` +
          `(0 CONNECTING, 1 OPEN, 2 CLOSING, 3 CLOSED), forcing reconnect`,
      )
      if (dead === null) return

      // Abandon the socket HERE rather than waiting for onclose to call
      // scheduleRetry. Measured: close() to onclose took 13 seconds against an
      // MSW-intercepted socket, and for all 13 the UI still read "Live" over a
      // connection this client had already declared dead -- precisely the lie
      // the watchdog exists to prevent. Close delivery is not ours to control,
      // so the status change must not depend on it.
      //
      // Handlers are detached BEFORE close() so the late onclose cannot
      // schedule a second, competing retry.
      dead.onopen = null
      dead.onmessage = null
      dead.onerror = null
      dead.onclose = null
      if (activeSocket === dead) activeSocket = null
      socket = null
      try {
        dead.close(4000, 'watchdog')
      } catch (error) {
        console.warn('[ws] close after watchdog threw, continuing', error)
      }
      scheduleRetry()
    }, WATCHDOG_MS)
  }

  function scheduleRetry(): void {
    if (disposed) return
    attempt += 1
    useLiveStore
      .getState()
      .setWsStatus(attempt >= ATTEMPTS_BEFORE_OFFLINE ? 'offline' : 'reconnecting')

    const base = Math.min(MAX_BACKOFF_MS, 1000 * 2 ** (attempt - 1))
    // +/-20% jitter so a fleet of clients does not retry in lockstep and
    // hammer the backend the instant it comes back.
    const jitter = base * 0.2 * (Math.random() * 2 - 1)
    const delay = Math.max(0, Math.round(base + jitter))
    console.warn(`[ws] retry ${attempt} in ${delay}ms (base ${base}ms)`)
    retry = window.setTimeout(open, delay)
  }

  function handleMessage(raw: string): void {
    // The whole body is wrapped. A malformed frame or an unexpected shape
    // must never escape as an unhandled rejection: that is how a screen goes
    // white mid-demo over one bad packet.
    try {
      const parsed: unknown = JSON.parse(raw)
      if (!isRecord(parsed)) {
        useLiveStore.getState().recordMalformed()
        console.warn('[ws] frame was not an object, ignored')
        return
      }

      const envelope: Envelope = parsed
      const type = typeof envelope.type === 'string' ? envelope.type : null
      if (type === null) {
        useLiveStore.getState().recordMalformed()
        console.warn('[ws] frame had no string type, ignored')
        return
      }

      // heartbeat carries ts at the TOP LEVEL with no wrapper. Reading
      // msg.data.ts here would throw on every beat.
      if (type === 'heartbeat') {
        if (typeof envelope.ts === 'string') {
          useLiveStore.getState().recordHeartbeat(envelope.ts, Date.now())
        } else {
          useLiveStore.getState().recordMalformed()
          console.warn('[ws] heartbeat had no top-level ts')
        }
        return
      }

      // Canonical says `data`, the backend manual says `payload`, and the
      // disagreement is unresolved. Read both and say which arrived, so the
      // answer comes from traffic rather than from a meeting.
      const usedPayloadKey = envelope.data === undefined && envelope.payload !== undefined
      const body = envelope.data ?? envelope.payload
      if (usedPayloadKey) {
        console.warn(`[ws] envelope key "payload" received for type "${type}" (canonical is "data")`)
      }

      if (!isRecord(body)) {
        useLiveStore.getState().recordMalformed()
        console.warn(`[ws] type "${type}" had no usable data/payload body`)
        return
      }

      if (type === 'alert') {
        // The socket path goes through readAlert too. A raw cast here would have
        // left the WS feed with exactly the drift the REST feed just stopped
        // having, and the two lists merge into one -- so a rename would have
        // been absorbed or not depending on which way the alert arrived.
        const alert = readAlert(body)
        if (alert !== null) useLiveStore.getState().addAlert(alert)
        return
      }
      if (type === 'sighting') {
        useLiveStore.getState().addSighting(body as unknown as VehicleSighting)
        return
      }
      if (type === 'system') {
        void queryClient.invalidateQueries({ queryKey: ['systemStatus'] })
        return
      }

      // Unknown type: logged once, recorded, ignored. Never thrown. A schema
      // addition from the backend must not take the screen down.
      const known = useLiveStore.getState().unknownMessageTypes
      if (!known.includes(type)) {
        console.warn(`[ws] unknown message type "${type}", ignoring (logged once)`)
      }
      useLiveStore.getState().recordUnknownType(type)
    } catch (error) {
      useLiveStore.getState().recordMalformed()
      console.warn('[ws] malformed frame caught, app continues', error)
    }
  }

  function open(): void {
    if (disposed) return

    const url = socketUrl()
    sawTrafficSinceOpen = false
    useLiveStore.getState().setWsStatus(hasConnectedOnce ? 'reconnecting' : 'connecting')

    try {
      socket = new WebSocket(url)
    } catch (error) {
      console.warn('[ws] could not construct socket', error)
      scheduleRetry()
      return
    }

    activeSocket = socket
    connectionCount += 1

    // onopen is deliberately NOT treated as "connected", and deliberately does
    // NOT reset the backoff.
    //
    // Measured, not assumed: with MSW running, `new WebSocket()` fires onopen
    // even for a port nothing has ever listened on -- the interceptor opens
    // the client side first and attempts passthrough afterwards. A probe
    // against ws://localhost:59999/nowhere fired onopen. Resetting the backoff
    // here produced a permanent ~1s reconnect loop against a dead server while
    // the UI cheerfully read "connected".
    //
    // The same principle already justifies the watchdog: traffic is evidence
    // of a working connection, an open handle is not. So the promotion to
    // connected happens on the first frame instead.
    socket.onopen = () => {
      armWatchdog()
      console.warn('[ws] socket open, waiting for first frame before reporting connected')
    }

    socket.onmessage = (event: MessageEvent<unknown>) => {
      armWatchdog()

      if (!sawTrafficSinceOpen) {
        sawTrafficSinceOpen = true
        attempt = 0
        useLiveStore.getState().setWsStatus('connected')

        if (hasConnectedOnce) {
          useLiveStore.getState().recordReconnect()
          // A socket-only client silently loses every event from the outage
          // gap. Refetch the REST surfaces that could have changed while we
          // were away. Doing this on first traffic rather than on open also
          // stops a flapping socket from firing a refetch storm.
          void queryClient.invalidateQueries({ queryKey: ['alerts'] })
          void queryClient.invalidateQueries({ queryKey: ['systemStatus'] })
          void queryClient.invalidateQueries({ queryKey: ['cameras'] })
          console.warn('[ws] reconnected, invalidated alerts/systemStatus/cameras')
        } else {
          console.warn('[ws] connected (first frame received)')
        }
        hasConnectedOnce = true
      }

      if (typeof event.data === 'string') handleMessage(event.data)
    }

    socket.onerror = () => {
      // onclose always follows; retry is scheduled there so it happens once.
      console.warn('[ws] socket error')
    }

    socket.onclose = () => {
      clearWatchdog()
      if (activeSocket === socket) activeSocket = null
      socket = null
      if (disposed) return
      scheduleRetry()
    }
  }

  open()

  return () => {
    disposed = true
    clearWatchdog()
    if (retry !== null) window.clearTimeout(retry)
    if (socket !== null) {
      socket.onclose = null
      socket.close(1000, 'client disconnect')
    }
    activeSocket = null
    useLiveStore.getState().setWsStatus('offline')
  }
}
