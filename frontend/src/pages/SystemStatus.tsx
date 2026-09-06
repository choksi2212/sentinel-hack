import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { driftSummary, readBenchmark, readCameras } from '../api/adapters'
import { apiBaseUrl, apiGet } from '../api/client'
import { endpoints } from '../api/endpoints'
import { ErrorPanel } from '../components/ErrorPanel'
import { hasCoordinates } from '../lib/geo'
import {
  selectClockSkewMs,
  selectLastHeartbeatAt,
  selectMalformedCount,
  selectReconnectCount,
  selectUnknownTypes,
  selectWsStatus,
  useLiveStore,
} from '../lib/liveStore'
import type { SourceMode, SystemStatus as SystemStatusBody } from '../types/api'
import type { BenchmarkReport, CameraStatus, WsStatus } from '../types/ui'
import { BENCHMARK_BUCKET_ORDER } from '../types/ui'

// Colour AND text AND shape for every state. Projector gamma flattens hue and
// roughly one man in twelve has some colour deficiency, so no state may be
// distinguishable by colour alone.
type Tone = 'ok' | 'warn' | 'bad' | 'unknown'

const TONE_TEXT: Record<Tone, string> = {
  ok: 'text-live',
  warn: 'text-degraded',
  bad: 'text-offline',
  unknown: 'text-ink-3',
}

function ToneMark({ tone }: { tone: Tone }) {
  return (
    <svg viewBox="0 0 16 16" className="size-3.5 shrink-0" aria-hidden="true">
      {tone === 'ok' && (
        <path d="M3 8.5l3.2 3.2L13 5" fill="none" stroke="currentColor" strokeWidth="2.4" />
      )}
      {tone === 'warn' && (
        <path d="M8 2 L15 14 H1 Z" fill="none" stroke="currentColor" strokeWidth="1.8" />
      )}
      {tone === 'bad' && (
        <path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" strokeWidth="2.4" />
      )}
      {tone === 'unknown' && (
        <circle cx="8" cy="8" r="5.5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeDasharray="3 2.5" />
      )}
    </svg>
  )
}

function StateLine({ label, tone, value }: { label: string; tone: Tone; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-rule py-1.5">
      <span className="text-ink-2">{label}</span>
      <span className={`inline-flex items-center gap-2 font-medium ${TONE_TEXT[tone]}`}>
        <ToneMark tone={tone} />
        {value}
      </span>
    </div>
  )
}

function Section({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <section className="mt-6 border border-rule bg-panel p-4">
      <h2 className="text-base font-semibold text-ink">{title}</h2>
      {note !== undefined && <p className="mt-0.5 text-sm text-ink-3">{note}</p>}
      <div className="mt-3">{children}</div>
    </section>
  )
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-rule py-1.5">
      <span className="text-ink-2">{label}</span>
      <span className="font-mono text-ink">{value}</span>
    </div>
  )
}

// source_mode spelled out. "ONLINE" is never a mode label: replay is labelled
// replay, and a live feed is named by its transport.
const SOURCE_MODE_LABEL: Record<SourceMode, string> = {
  live_rtsp: 'Live RTSP',
  live_hls: 'Live HLS',
  file: 'File replay',
  frames: 'Frame replay',
  synthetic: 'Synthetic replay',
}

const CAMERA_STATUS_LABEL: Record<CameraStatus, string> = {
  online: 'Online',
  offline: 'Offline',
  degraded: 'Degraded',
  unknown: 'Unknown',
}

const CAMERA_STATUS_TONE: Record<CameraStatus, Tone> = {
  online: 'ok',
  offline: 'bad',
  degraded: 'warn',
  unknown: 'unknown',
}

const WS_LABEL: Record<WsStatus, string> = {
  connecting: 'Connecting',
  connected: 'Live',
  reconnecting: 'Reconnecting',
  offline: 'Disconnected',
}

const WS_TONE: Record<WsStatus, Tone> = {
  connecting: 'unknown',
  connected: 'ok',
  reconnecting: 'warn',
  offline: 'bad',
}

// The failure-bucket list that used to live here is gone. Canonical 7.3
// defines no failure-reason field, so every count it rendered came from a
// shape the contract does not describe. Its one irreplaceable line -- that a
// plate below ~30px was never resolved by the sensor and no model change
// recovers it -- now sits beside the <30 bucket it explains, as a standing
// engineering finding rather than as data we pretend the API sent.

function HealthLine({ label, path, note }: { label: string; path: string; note: string }) {
  // Polled, because a socket says nothing about whether HTTP is answering.
  // refetchInterval lives HERE and nowhere else: every other query is either
  // event-driven or covered by the WS reconnect invalidation.
  const query = useQuery({
    queryKey: ['health', path],
    queryFn: async () => {
      await apiGet<unknown>(path)
      return true
    },
    refetchInterval: 10_000,
    // The ONLY background-refetching queries in the app.
    //
    // Without this, TanStack pauses the interval whenever the document is
    // hidden, so a minimised or projector-backgrounded tab stops polling and
    // leaves the last "Passing" frozen on screen. That is the same lie as the
    // dead-port onopen from D-026: a UI asserting health it is no longer
    // measuring. Liveness is the one thing that must keep checking when nobody
    // is looking at it.
    //
    // refetchOnWindowFocus stays OFF everywhere, here included: alt-tabbing back
    // must not fire a refetch storm across every screen mid-demo.
    refetchIntervalInBackground: true,
    retry: false,
  })

  const tone: Tone = query.isPending ? 'unknown' : query.isError ? 'bad' : 'ok'
  const value = query.isPending ? 'Checking' : query.isError ? 'Failing' : 'Passing'

  return (
    <div className="border-b border-rule py-1.5">
      <div className="flex items-center justify-between gap-4">
        <span className="text-ink-2">
          {label} <span className="font-mono text-ink-3">{path}</span>
        </span>
        <span className={`inline-flex items-center gap-2 font-medium ${TONE_TEXT[tone]}`}>
          <ToneMark tone={tone} />
          {value}
        </span>
      </div>
      <p className="mt-0.5 text-sm text-ink-3">{note}</p>
    </div>
  )
}

function BenchmarkPanel() {
  const query = useQuery({
    queryKey: ['benchmark'],
    queryFn: async () => {
      const raw = await apiGet<unknown>(endpoints.metricsBenchmark())
      return readBenchmark(raw)
    },
    staleTime: 5000,
  })

  if (query.isPending) {
    return <p className="text-sm text-ink-2">Loading benchmark…</p>
  }
  if (query.isError) {
    return <ErrorPanel error={query.error} onRetry={() => void query.refetch()} />
  }

  const report: BenchmarkReport | null = query.data ?? null
  if (report === null) {
    return (
      <p className="text-ink">
        The benchmark report could not be read. Required fields were missing, so
        nothing is shown rather than a partial result.
      </p>
    )
  }

  return (
    <>
      {/* THE BUCKETS COME FIRST AND CARRY THE WEIGHT. Canonical 7.2: "No
          accuracy number may be reported as a single average." 7.3 none the
          less ships one headline scalar, so conforming to 7.3 by rendering
          that scalar prominently would violate 7.2. The resolution is
          ordering: the six buckets are the finding and are read first; the
          average is a footnote below them, where it cannot be quoted alone.

          Rates render as the decimals 7.3 sends -- 0.38, not "38%". No
          percent sign appears on this screen. */}
      <h3 className="text-sm font-semibold text-ink-2">
        Correct-plate event rate by plate width
      </h3>
      <p className="mt-0.5 text-sm text-ink-3">
        Canonical 7.2: accuracy is never one number. Read down the column.
      </p>

      <ul className="mt-2">
        {BENCHMARK_BUCKET_ORDER.map((key) => {
          const rate = report.by_plate_width[key]
          const measured = typeof rate === 'number'
          return (
            <li key={key} className="border-b border-rule py-2">
              <div className="flex items-baseline gap-3">
                <span className="w-20 shrink-0 font-mono text-ink">{key}</span>
                {/* Colour AND width AND the number. The bar is a second
                    encoding of the same value, not decoration. */}
                <span className="h-2.5 min-w-0 flex-1 bg-sunken" aria-hidden="true">
                  <span
                    className={`block h-full ${measured && rate < 0.5 ? 'bg-offline' : 'bg-live'}`}
                    style={{ width: measured ? `${rate * 100}%` : '0%' }}
                  />
                </span>
                <span
                  className={`w-16 shrink-0 text-right font-mono ${
                    measured ? 'text-ink' : 'text-ink-3'
                  }`}
                >
                  {measured ? rate.toFixed(2) : key in report.by_plate_width ? 'not run' : '—'}
                </span>
              </div>
              {key === '<30' && (
                /* Not from the API. Canonical 7.3 has no failure-reason field,
                   and this is a standing engineering finding rather than
                   backend data: below ~30px the plate was never resolved by
                   the sensor, so no model change recovers it. It is the
                   strongest sentence on this screen and it belongs beside the
                   bucket it explains. */
                <p className="mt-1 pl-[5.75rem] text-sm text-ink-2">
                  No software fix at this width — recommend camera placement.
                </p>
              )}
            </li>
          )
        })}
      </ul>

      {/* The headline scalar, deliberately AFTER the buckets and deliberately
          small. Labelled as an average so it cannot be lifted out of context
          as "the accuracy number", which is precisely what 7.2 forbids. */}
      <p className="mt-3 text-sm text-ink-2">
        Mean across all widths:{' '}
        <span className="font-mono text-ink">
          {report.e2e_correct_plate_event_rate === null
            ? 'not reported'
            : report.e2e_correct_plate_event_rate.toFixed(2)}
        </span>{' '}
        — an average over the six buckets above, not a system accuracy figure.
      </p>

      <div className="mt-4">
      <Row label="Run id" value={report.run_id} />
      </div>
      {report.dataset_manifest_sha256 === null ? (
        // Stated on screen, not hidden. Without the manifest hash the run
        // cannot be reproduced, and a number nobody can reproduce is not
        // evidence. The panel still renders: hiding it would lose the finding
        // along with the caveat.
        <div className="flex items-baseline justify-between gap-4 border-b border-rule py-1.5">
          <span className="text-ink-2">Dataset manifest sha256</span>
          <span className="text-offline">
            Absent — this run cannot be reproduced and is not evidence
          </span>
        </div>
      ) : (
        <Row
          label="Dataset manifest sha256"
          value={
            <span className="break-all text-sm">{report.dataset_manifest_sha256}</span>
          }
        />
      )}

    </>
  )
}

export function SystemStatus() {
  const wsStatus = useLiveStore(selectWsStatus)
  const lastHeartbeatAt = useLiveStore(selectLastHeartbeatAt)
  const clockSkewMs = useLiveStore(selectClockSkewMs)
  const reconnectCount = useLiveStore(selectReconnectCount)
  const malformedCount = useLiveStore(selectMalformedCount)
  const unknownTypes = useLiveStore(selectUnknownTypes)

  // Ticks so "seconds since last heartbeat" actually climbs. One interval for
  // the whole screen, cleared on unmount.
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [])

  const statusQuery = useQuery({
    queryKey: ['systemStatus'],
    queryFn: () => apiGet<SystemStatusBody>(endpoints.systemStatus()),
    staleTime: 5000,
  })

  const camerasQuery = useQuery({
    queryKey: ['cameras'],
    queryFn: async () => readCameras(await apiGet<unknown>(endpoints.cameras())),
    staleTime: 5000,
  })

  // Read on every render rather than held in state: these are module-level
  // counters and a snapshot would go stale the moment an adapter ran.
  const drift = driftSummary()

  const status = statusQuery.data ?? null
  const cameras = camerasQuery.data ?? []

  const cameraCounts = cameras.reduce<Record<CameraStatus, number>>(
    (acc, camera) => {
      acc[camera.status] += 1
      return acc
    },
    { online: 0, offline: 0, degraded: 0, unknown: 0 },
  )
  const unplaced = cameras.filter((camera) => !hasCoordinates(camera)).length

  const secondsSinceHeartbeat =
    lastHeartbeatAt === null ? null : Math.max(0, Math.round((now - lastHeartbeatAt) / 1000))

  return (
    <section>
      <h1 className="text-[1.35rem] font-semibold text-ink">System status</h1>

      {/* 1. SERVICE HEALTH */}
      <Section title="Service health">
        {statusQuery.isError ? (
          // Scoped to this section. Health, realtime, drift and build do not
          // depend on this query and keep rendering.
          <ErrorPanel error={statusQuery.error} onRetry={() => void statusQuery.refetch()} />
        ) : statusQuery.isPending ? (
          <p className="text-sm text-ink-2">Loading service health…</p>
        ) : status === null ? (
          <p className="text-ink">The status endpoint returned no body.</p>
        ) : (
          <>
            <StateLine
              label="API"
              tone={status.api === 'ok' ? 'ok' : 'warn'}
              value={status.api === 'ok' ? 'OK' : 'Degraded'}
            />
            <StateLine
              label="PostgreSQL"
              tone={status.postgres ? 'ok' : 'bad'}
              value={status.postgres ? 'Reachable' : 'Unreachable'}
            />
            <StateLine
              label="Redis"
              tone={status.redis ? 'ok' : 'bad'}
              value={status.redis ? 'Reachable' : 'Unreachable'}
            />
          </>
        )}
      </Section>

      {/* 2. SOURCE. No throughput figure: SystemStatus carries none by design,
          and computing one client-side from replay would be an inflated number
          needing a caveat nobody reads. */}
      <Section title="Source">
        {status === null ? (
          <p className="text-sm text-ink-2">
            {statusQuery.isError ? 'Unavailable while the status endpoint is failing.' : 'Loading…'}
          </p>
        ) : (
          <>
            <StateLine
              label="Mode"
              tone={status.is_live ? 'ok' : 'warn'}
              value={SOURCE_MODE_LABEL[status.source_mode]}
            />
            <StateLine
              label="Feed"
              tone={status.is_live ? 'ok' : 'warn'}
              value={status.is_live ? 'Live feed' : 'Replay, not live'}
            />
            <Row label="Cameras total" value={status.cameras_total} />
            <Row label="Cameras live" value={status.cameras_live} />
            <Row label="Cameras replay" value={status.cameras_replay} />
          </>
        )}
      </Section>

      {/* 3. CAMERA BREAKDOWN, counted from the existing cameras query. */}
      <Section title="Camera breakdown" note="Counted from the camera registry, by status.">
        {camerasQuery.isError ? (
          <ErrorPanel error={camerasQuery.error} onRetry={() => void camerasQuery.refetch()} />
        ) : camerasQuery.isPending ? (
          <p className="text-sm text-ink-2">Loading cameras…</p>
        ) : (
          <>
            {(Object.keys(CAMERA_STATUS_LABEL) as CameraStatus[]).map((key) => (
              <StateLine
                key={key}
                label={CAMERA_STATUS_LABEL[key]}
                tone={CAMERA_STATUS_TONE[key]}
                value={String(cameraCounts[key])}
              />
            ))}
            <Row label="Registered" value={cameras.length} />
            {/* Listed, never plotted. A camera without coordinates is not a
                camera at 0,0. */}
            <Row label="Without coordinates" value={unplaced} />
          </>
        )}
      </Section>

      {/* 4. LIVENESS. Root-level paths, not under /api/v1. */}
      <Section
        title="Liveness"
        note="Polled every 10 seconds. Running and usable are different states."
      >
        <HealthLine
          label="Live"
          path={endpoints.healthLive()}
          note="The process is running and answering."
        />
        <HealthLine
          label="Ready"
          path={endpoints.healthReady()}
          note="Dependencies are up, so the process can actually serve requests."
        />
      </Section>

      {/* 5. REALTIME */}
      <Section title="Realtime">
        <StateLine label="WebSocket" tone={WS_TONE[wsStatus]} value={WS_LABEL[wsStatus]} />
        <Row
          label="Since last heartbeat"
          value={secondsSinceHeartbeat === null ? 'No heartbeat yet' : `${secondsSinceHeartbeat}s`}
        />
        {/* Named as a client-side measurement, not as system truth. This is one
            browser's clock against one heartbeat's ts, including network delay;
            it is a smell test, not an NTP reading. */}
        <Row
          label="Clock skew, measured by this client against the heartbeat ts"
          value={clockSkewMs === null ? 'Not measured yet' : `${clockSkewMs} ms`}
        />
        <Row label="Reconnects" value={reconnectCount} />
        <Row label="Malformed frames" value={malformedCount} />
        <Row
          label="Unknown message types"
          value={unknownTypes.length === 0 ? 'None' : unknownTypes.join(', ')}
        />
      </Section>

      {/* 6. CONTRACT DRIFT */}
      <Section
        title="Contract drift"
        note="Counted since page load, in memory, reset on reload — not cumulative system figures. Counts adapter reads, not distinct records: re-opening the same journey re-reads the same payload and increments again."
      >
        <h3 className="text-sm font-semibold text-ink-2">
          Fallbacks — a name or default moved, the record survived
        </h3>
        {drift.fallbacks.length === 0 ? (
          <p className="mt-1 text-ink-2">No fallbacks since page load.</p>
        ) : (
          <ul className="mt-1">
            {drift.fallbacks.map((entry) => (
              <li
                key={entry.name}
                className="flex items-baseline justify-between gap-4 border-b border-rule py-1.5"
              >
                <span className="font-mono text-ink">{entry.name}</span>
                <span className="font-mono text-ink">{entry.count}</span>
              </li>
            ))}
          </ul>
        )}

        {/* Deliberately a SECOND list. Collapsed into one total, "3 records were
            discarded" hides inside "the API spells it differently" -- and only
            one of those means someone needs to know today. */}
        {/* Wording measured, not assumed. The counter increments once per
            adapter read, so three visits to one journey with one bad segment
            reads 6, not 1. "A record was dropped entirely" would then be a
            false claim about how much evidence was lost -- the exact kind of
            inflated number this screen exists to avoid. */}
        <h3 className="mt-4 text-sm font-semibold text-ink-2">
          Discards — a record was unusable and dropped on read
        </h3>
        {drift.discards.length === 0 ? (
          <p className="mt-1 text-ink-2">No discards since page load.</p>
        ) : (
          <ul className="mt-1">
            {drift.discards.map((entry) => (
              <li
                key={entry.name}
                className="flex items-baseline justify-between gap-4 border-b border-rule py-1.5"
              >
                <span className="font-mono text-ink">{entry.name}</span>
                <span className="font-mono text-offline">{entry.count}</span>
              </li>
            ))}
          </ul>
        )}

        {drift.fallbacks.length === 0 && drift.discards.length === 0 && (
          <p className="mt-3 text-ink">No drift detected since page load.</p>
        )}
      </Section>

      {/* 7. BENCHMARK */}
      <Section title="Benchmark" note="Read-accuracy run against a fixed manifest.">
        <BenchmarkPanel />
      </Section>

      {/* 8. BUILD */}
      <Section
        title="Build"
        note="Build-time configuration. These values are inlined when the bundle is built, so the same artifact cannot be repointed without a rebuild."
      >
        <Row label="API base URL" value={apiBaseUrl} />
        <Row label="App mode" value={import.meta.env.VITE_APP_MODE} />
      </Section>
    </section>
  )
}
