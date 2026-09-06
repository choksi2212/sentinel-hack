import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import type { AdaptedAlert } from '../api/adapters'
import { readAlerts } from '../api/adapters'
import { apiDelete, apiGet, apiPost } from '../api/client'
import { endpoints } from '../api/endpoints'
import { ErrorPanel } from '../components/ErrorPanel'
import { MatchStateChip } from '../components/MatchStateChip'
import { mergeAlertFeed } from '../lib/alertFeed'
import {
  selectAlerts,
  selectLocallyAcknowledged,
  selectWsStatus,
  useLiveStore,
} from '../lib/liveStore'
import { formatAbsoluteTime, formatRelativeTime } from '../lib/time'
import { isOptimistic, optimisticEntry, validateDraft } from '../lib/watchlist'
import type { Alert } from '../types/api'
import type { WatchlistDraft, WatchlistEntry, WatchlistPriority, WsStatus } from '../types/ui'

// Priority is a LEFT BAR plus a TEXT LABEL, never a full-width colour fill.
// A screen of filled red rows reads as a system-wide alarm rather than as a
// list of individual matches, and an operator stops reading it.
//
// Every state carries colour AND a label AND a shape: projector gamma
// flattens hue, and roughly one man in twelve has some colour deficiency.
// Class names are written out so Tailwind's source scan can see them.
const PRIORITY: Record<
  Alert['priority'],
  { label: string; bar: string; text: string; ticks: number }
> = {
  critical: { label: 'Critical', bar: 'bg-prio-critical', text: 'text-prio-critical', ticks: 4 },
  high: { label: 'High', bar: 'bg-prio-high', text: 'text-prio-high', ticks: 3 },
  medium: { label: 'Medium', bar: 'bg-prio-medium', text: 'text-prio-medium', ticks: 2 },
  low: { label: 'Low', bar: 'bg-prio-low', text: 'text-prio-low', ticks: 1 },
}

const PRIORITY_ORDER: WatchlistPriority[] = ['low', 'medium', 'high', 'critical']

// The wire is not the type. Serving the execution manual's `severity` instead
// of Canonical 6.5's `priority` makes alert.priority undefined, and
// PRIORITY[undefined].bar threw during render and white-screened the entire
// app -- every screen, not just this row.
//
// "Unknown" with no ticks is the honest degradation: we have an alert and we do
// not know how urgent it is. Inventing "medium" would be worse than useless on
// a police screen, because it reads as an assessment nobody made.
const PRIORITY_UNKNOWN = {
  label: 'Unknown priority',
  bar: 'bg-rule',
  text: 'text-ink-2',
  ticks: 0,
} as const

// Exact copy contract. "Candidate match — requires review" carries an em-dash
// and the wsStatus strings carry an ellipsis; those are precisely the glyphs a
// corrupted UTF-8 read destroys silently, which is why the mojibake gate runs
// every session.
//
// Only 'exact' gets unqualified language, and the word "confirmed" appears
// nowhere: a match is evidence, not a verdict.
const MATCH_NOTE: Record<Alert['match_state'], string | null> = {
  exact: null,
  probable: 'Candidate match — requires review',
}

const WS_STATUS_COPY: Record<WsStatus, string> = {
  connecting: 'Connecting…',
  connected: 'Live',
  reconnecting: 'Reconnecting…',
  offline: 'Disconnected — showing last known data',
}

const WS_STATUS_STYLE: Record<WsStatus, string> = {
  connecting: 'text-ink-3',
  connected: 'text-live',
  reconnecting: 'text-degraded',
  offline: 'text-offline',
}

// Shape as well as colour: a filled dot for live, a hollow one for offline,
// dashed while it is trying.
function WsShape({ status }: { status: WsStatus }) {
  return (
    <svg viewBox="0 0 20 20" className="size-3 shrink-0" aria-hidden="true">
      {status === 'connected' && <circle cx="10" cy="10" r="6" fill="currentColor" />}
      {(status === 'connecting' || status === 'reconnecting') && (
        <circle cx="10" cy="10" r="6" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="3 2.5" />
      )}
      {status === 'offline' && (
        <circle cx="10" cy="10" r="6" fill="none" stroke="currentColor" strokeWidth="2" />
      )}
    </svg>
  )
}

function PriorityTicks({ ticks, className }: { ticks: number; className: string }) {
  return (
    <span className="inline-flex items-center gap-0.5" aria-hidden="true">
      {[1, 2, 3, 4].map((n) => (
        <span key={n} className={`inline-block h-2.5 w-1 ${n <= ticks ? className : 'bg-rule'}`} />
      ))}
    </span>
  )
}

function AlertRow({
  alert,
  onAcknowledge,
  pending,
}: {
  alert: AdaptedAlert
  onAcknowledge: (alertId: string) => void
  pending: boolean
}) {
  // priority is OPTIONAL on AdaptedAlert: absent means the API sent a value
  // outside the union, or none at all, and no assessment is invented for it.
  const priority =
    alert.priority === undefined
      ? PRIORITY_UNKNOWN
      : (PRIORITY[alert.priority] ?? PRIORITY_UNKNOWN)
  const note = MATCH_NOTE[alert.match_state] ?? null
  // Never blank. A plate we could not read says so, exactly as the search rows
  // and the map popups do -- CONVENTIONS.md rule 6, applied to the third surface.
  const plateDisplay = alert.plate ?? 'Unreadable'
  const plateUnreadable = alert.plate === null

  return (
    // An acknowledged alert STAYS in the list and is dimmed, never removed.
    // Disappearing would assert the alert is resolved; acknowledgement only
    // means somebody has seen it.
    <li
      className={`flex items-stretch border-b border-rule ${
        alert.acknowledged ? 'bg-sunken' : 'bg-panel'
      }`}
    >
      <span className={`w-1.5 shrink-0 ${priority.bar}`} aria-hidden="true" />

      <div className={`min-w-0 flex-1 px-3 py-2 ${alert.acknowledged ? 'opacity-70' : ''}`}>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span
            className={
              plateUnreadable ? 'font-mono text-ink-2' : 'font-mono font-medium text-ink'
            }
          >
            {plateDisplay}
          </span>
          {/* No confidence prop exists on this chip and must not be added. */}
          <MatchStateChip state={alert.match_state} observations={1} />
          <span className={`inline-flex items-center gap-1.5 text-sm ${priority.text}`}>
            <PriorityTicks ticks={priority.ticks} className={priority.bar} />
            {priority.label}
          </span>

          {alert.acknowledged ? (
            <span className="inline-flex items-center gap-1 border border-rule px-1.5 py-0.5 text-sm text-ink-2">
              <svg viewBox="0 0 16 16" className="size-3" aria-hidden="true">
                <path
                  d="M3 8.5l3.2 3.2L13 5"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="square"
                />
              </svg>
              Acknowledged
            </span>
          ) : (
            <button
              type="button"
              onClick={() => onAcknowledge(alert.alert_id)}
              disabled={pending}
              className="border border-rule-2 px-2 py-0.5 text-sm text-ink hover:bg-sunken disabled:opacity-50"
            >
              {pending ? 'Acknowledging…' : 'Acknowledge'}
            </button>
          )}
        </div>

        {note !== null && <p className="mt-1 text-sm text-probable">{note}</p>}

        <p className="mt-1 text-sm text-ink-2" title={formatAbsoluteTime(alert.created_at)}>
          {alert.camera_name} · {formatRelativeTime(alert.created_at)}
        </p>
      </div>
    </li>
  )
}

function WatchlistPanel() {
  const queryClient = useQueryClient()
  const [plate, setPlate] = useState('')
  const [reason, setReason] = useState('')
  const [priority, setPriority] = useState<WatchlistPriority>('medium')
  const [submitted, setSubmitted] = useState(false)
  const [writeError, setWriteError] = useState<unknown>(null)

  const watchlistQuery = useQuery({
    queryKey: ['watchlist'],
    queryFn: () => apiGet<WatchlistEntry[]>(endpoints.watchlist()),
    staleTime: 5000,
  })

  const draft: WatchlistDraft = { plate, reason, priority }
  const { problems, normalised } = validateDraft(draft)

  const addMutation = useMutation({
    mutationFn: (entry: WatchlistDraft) =>
      apiPost<WatchlistDraft, WatchlistEntry>(endpoints.watchlist(), entry),
    onMutate: async (entry) => {
      await queryClient.cancelQueries({ queryKey: ['watchlist'] })
      const previous = queryClient.getQueryData<WatchlistEntry[]>(['watchlist'])
      queryClient.setQueryData<WatchlistEntry[]>(['watchlist'], (old) => [
        ...(old ?? []),
        optimisticEntry(entry),
      ])
      return { previous }
    },
    onError: (error, _entry, context) => {
      if (context?.previous !== undefined) {
        queryClient.setQueryData(['watchlist'], context.previous)
      }
      setWriteError(error)
    },
    onSuccess: () => {
      setWriteError(null)
      setPlate('')
      setReason('')
      setSubmitted(false)
    },
    onSettled: () => void queryClient.invalidateQueries({ queryKey: ['watchlist'] }),
  })

  const removeMutation = useMutation({
    mutationFn: (id: string) => apiDelete<null>(endpoints.watchlistItem(id)),
    onMutate: async (id) => {
      await queryClient.cancelQueries({ queryKey: ['watchlist'] })
      const previous = queryClient.getQueryData<WatchlistEntry[]>(['watchlist'])
      queryClient.setQueryData<WatchlistEntry[]>(['watchlist'], (old) =>
        (old ?? []).filter((e) => e.id !== id),
      )
      return { previous }
    },
    onError: (error, _id, context) => {
      if (context?.previous !== undefined) {
        queryClient.setQueryData(['watchlist'], context.previous)
      }
      setWriteError(error)
    },
    onSuccess: () => setWriteError(null),
    onSettled: () => void queryClient.invalidateQueries({ queryKey: ['watchlist'] }),
  })

  const entries = watchlistQuery.data ?? []

  return (
    <section className="mt-6 border border-rule bg-panel p-4">
      <h2 className="text-base font-semibold text-ink">Watchlist</h2>

      <form
        className="mt-3 flex flex-wrap items-start gap-3"
        onSubmit={(event) => {
          event.preventDefault()
          setSubmitted(true)
          if (problems.length > 0) return
          addMutation.mutate(normalised)
        }}
      >
        <label className="flex flex-col gap-1">
          <span className="text-sm text-ink-2">Plate</span>
          <input
            value={plate}
            onChange={(e) => setPlate(e.target.value)}
            className="border border-rule bg-sunken px-2 py-1 font-mono text-ink"
            placeholder="GJ01AB1234"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-sm text-ink-2">Reason</span>
          <input
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            className="w-64 border border-rule bg-sunken px-2 py-1 text-ink"
            placeholder="Why this plate is watched"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-sm text-ink-2">Priority</span>
          <select
            value={priority}
            onChange={(e) => setPriority(e.target.value as WatchlistPriority)}
            className="border border-rule bg-sunken px-2 py-1 text-ink"
          >
            {PRIORITY_ORDER.map((p) => (
              <option key={p} value={p}>
                {PRIORITY[p].label}
              </option>
            ))}
          </select>
        </label>

        <button
          type="submit"
          disabled={addMutation.isPending}
          className="mt-6 border border-rule-2 px-3 py-1 text-ink hover:bg-sunken disabled:opacity-50"
        >
          {addMutation.isPending ? 'Adding…' : 'Add'}
        </button>
      </form>

      {/* Show what will actually be stored, before it is stored. Telling the
          user that "gj 01 ab 1234" becomes "GJ01AB1234" is free honesty about
          what is being watched, and catches a typo while it is still cheap. */}
      {plate.trim() !== '' && (
        <p className="mt-2 text-sm text-ink-2">
          Will be watched as <span className="font-mono text-ink">{normalised.plate}</span>
        </p>
      )}

      {submitted &&
        problems.map((problem) => (
          <p key={problem.field} className="mt-2 text-sm text-offline">
            {problem.message}
          </p>
        ))}

      {writeError !== null && (
        <div className="mt-3">
          {/* A watchlist add or delete. A 404 here is a real failure, not an
              empty result, so it must not render as a calm no-results panel. */}
          <ErrorPanel error={writeError} operation="write" />
        </div>
      )}

      {watchlistQuery.isError && (
        <div className="mt-3">
          <ErrorPanel
            error={watchlistQuery.error}
            onRetry={() => void watchlistQuery.refetch()}
          />
        </div>
      )}

      {watchlistQuery.isPending && (
        <p className="mt-3 text-sm text-ink-2">Loading watchlist…</p>
      )}

      {!watchlistQuery.isPending && !watchlistQuery.isError && (
        <>
          {entries.length === 0 ? (
            <p className="mt-3 text-sm text-ink-2">No plates on the watchlist.</p>
          ) : (
            <ul className="mt-3 border-t border-rule">
              {entries.map((entry) => (
                <li
                  key={entry.id}
                  className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-rule py-2"
                >
                  <span className="font-mono text-ink">{entry.plate}</span>
                  {/* Same guard as the alert row: a priority the contract does
                      not list must degrade, not throw. */}
                  <span className={`text-sm ${(PRIORITY[entry.priority] ?? PRIORITY_UNKNOWN).text}`}>
                    {(PRIORITY[entry.priority] ?? PRIORITY_UNKNOWN).label}
                  </span>
                  <span className="min-w-0 flex-1 truncate text-sm text-ink-2">
                    {entry.reason}
                  </span>
                  {isOptimistic(entry) ? (
                    <span className="text-sm text-ink-3">Adding…</span>
                  ) : (
                    <button
                      type="button"
                      onClick={() => removeMutation.mutate(entry.id)}
                      disabled={removeMutation.isPending}
                      className="border border-rule px-2 py-0.5 text-sm text-ink-2 hover:bg-sunken disabled:opacity-50"
                    >
                      Remove
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  )
}

export function Alerts() {
  const queryClient = useQueryClient()
  const wsStatus = useLiveStore(selectWsStatus)
  const liveAlerts = useLiveStore(selectAlerts)
  const locallyAcknowledged = useLiveStore(selectLocallyAcknowledged)
  const [ackError, setAckError] = useState<unknown>(null)

  const alertsQuery = useQuery({
    queryKey: ['alerts'],
    queryFn: async () => readAlerts(await apiGet<unknown>(endpoints.alerts())),
    staleTime: 5000,
  })

  // Read only for the empty-state count. It must never fabricate a number:
  // while this is loading or errored the empty state says "No alerts." alone.
  const watchlistQuery = useQuery({
    queryKey: ['watchlist'],
    queryFn: () => apiGet<WatchlistEntry[]>(endpoints.watchlist()),
    staleTime: 5000,
  })

  const ackMutation = useMutation({
    // The path carries the id and the verb carries the intent, so there is
    // nothing to send. Canonical 6.5 does not specify a body for this endpoint.
    mutationFn: (alertId: string) =>
      apiPost<Record<string, never>, AdaptedAlert>(endpoints.alertAcknowledge(alertId), {}),
    onMutate: async (alertId) => {
      await queryClient.cancelQueries({ queryKey: ['alerts'] })
      const previous = queryClient.getQueryData<AdaptedAlert[]>(['alerts'])
      // Snapshot the LIVE copy's value too, so a rollback restores a known
      // state rather than assuming the flip can simply be reversed.
      const previousLive =
        useLiveStore.getState().alerts.find((a) => a.alert_id === alertId)?.acknowledged ?? null

      // acknowledged is a BOOLEAN. The optimistic write flips a boolean; it
      // does not invent a timestamp the server has not confirmed.
      queryClient.setQueryData<AdaptedAlert[]>(['alerts'], (old) =>
        (old ?? []).map((a) => (a.alert_id === alertId ? { ...a, acknowledged: true } : a)),
      )
      useLiveStore.getState().setAlertAcknowledged(alertId, true)
      return { previous, previousLive }
    },
    onError: (error, alertId, context) => {
      if (context?.previous !== undefined) {
        queryClient.setQueryData(['alerts'], context.previous)
      }
      if (context?.previousLive !== null && context?.previousLive !== undefined) {
        useLiveStore.getState().setAlertAcknowledged(alertId, context.previousLive)
      }
      // The write did not land, so the sticky claim must not survive it.
      useLiveStore.getState().clearAcknowledged(alertId)
      // A silently reverted checkbox is a lie: the operator watched it change.
      // The rollback must be accompanied by a visible reason and a request_id.
      setAckError(error)
    },
    // Written ONLY here, on a confirmed 2xx. Not in onMutate: an optimistic
    // flip is a prediction, and making a prediction sticky would outrank the
    // server on nothing but hope.
    onSuccess: (_data, alertId) => {
      useLiveStore.getState().confirmAcknowledged(alertId)
      setAckError(null)
    },
    onSettled: () => void queryClient.invalidateQueries({ queryKey: ['alerts'] }),
  })

  const feed = useMemo(
    () => mergeAlertFeed(alertsQuery.data ?? [], liveAlerts, locallyAcknowledged),
    [alertsQuery.data, liveAlerts, locallyAcknowledged],
  )

  // Prune in an effect, not during the merge: pruning is a store write, and a
  // store write during render is exactly the loop the memo exists to avoid.
  // pruneAcknowledged returns the same set when nothing changed, so this
  // settles after one pass rather than oscillating.
  useEffect(() => {
    useLiveStore.getState().pruneAcknowledged(feed.presentIds)
  }, [feed.presentIds])

  // Only a settled, successful watchlist query may contribute a number.
  const watchlistCount =
    !watchlistQuery.isPending && !watchlistQuery.isError && watchlistQuery.data
      ? watchlistQuery.data.length
      : null

  return (
    <section>
      <h1 className="text-[1.35rem] font-semibold text-ink">Alerts</h1>

      {/* Always visible, not only when degraded. A UI that speaks up only when
          something is broken cannot be trusted when it is silent. */}
      <p className={`mt-1 inline-flex items-center gap-2 text-sm ${WS_STATUS_STYLE[wsStatus]}`}>
        <WsShape status={wsStatus} />
        {WS_STATUS_COPY[wsStatus]}
      </p>

      {ackError !== null && (
        <div className="mt-4">
          <ErrorPanel error={ackError} operation="write" />
        </div>
      )}

      {alertsQuery.isError && (
        <div className="mt-4">
          <ErrorPanel error={alertsQuery.error} onRetry={() => void alertsQuery.refetch()} />
        </div>
      )}

      {alertsQuery.isPending && (
        <div className="mt-4 border border-rule">
          {Array.from({ length: 6 }, (_, index) => (
            <div
              key={index}
              className="flex items-center gap-3 border-b border-rule bg-panel px-3 py-3"
            >
              <div className="h-4 w-32 bg-sunken" />
              <div className="h-4 w-24 bg-sunken" />
            </div>
          ))}
        </div>
      )}

      {!alertsQuery.isPending && !alertsQuery.isError && (
        <>
          {feed.alerts.length === 0 ? (
            <p className="mt-4 border border-rule bg-sunken p-4 text-ink">
              {watchlistCount === null
                ? 'No alerts.'
                : `No alerts. Watchlist is monitoring ${watchlistCount} plates.`}
            </p>
          ) : (
            <>
              <p className="mt-4 text-sm text-ink-2">
                {feed.alerts.length} alerts
                {feed.droppedByCap > 0 && <> · {feed.droppedByCap} older alerts not shown</>}
              </p>
              <ul className="mt-2 border-x border-t border-rule">
                {feed.alerts.map((alert) => (
                  <AlertRow
                    key={alert.alert_id}
                    alert={alert}
                    onAcknowledge={(id) => ackMutation.mutate(id)}
                    pending={ackMutation.isPending && ackMutation.variables === alert.alert_id}
                  />
                ))}
              </ul>
            </>
          )}
        </>
      )}

      <WatchlistPanel />
    </section>
  )
}
