import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { apiGet } from '../api/client'
import { endpoints } from '../api/endpoints'
import { appMode, isDemo } from '../lib/appMode'
import { selectWsStatus, useLiveStore } from '../lib/liveStore'
import type { SourceMode, SystemStatus } from '../types/api'
import type { WsStatus } from '../types/ui'

// source_mode spelled out beside the REPLAY badge. The raw enum value is not
// a sentence, and the audience has to know what they are looking at.
const SOURCE_MODE_LABEL: Record<SourceMode, string> = {
  live_rtsp: 'live RTSP',
  live_hls: 'live HLS',
  file: 'file replay',
  frames: 'frame replay',
  synthetic: 'synthetic',
}

function Clock() {
  const [now, setNow] = useState(() => new Date())

  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000)
    return () => window.clearInterval(id)
  }, [])

  return (
    <span className="font-mono text-sm tabular-nums text-ink-2">
      {now.toLocaleTimeString('en-GB', { hour12: false })}
    </span>
  )
}

// The single most important control in the app. If this reads LIVE while the
// backend is replaying recorded footage, everything on screen is a claim
// about the present that is not true.
function ModeBadge() {
  const { data, isPending, isError } = useQuery({
    queryKey: ['systemStatus'],
    queryFn: () => apiGet<SystemStatus>(endpoints.systemStatus()),
  })

  // Never guess a mode. Until the answer arrives, say nothing rather than
  // defaulting to either badge -- a default here is a claim.
  if (isPending) {
    return (
      <span className="border border-rule px-2 py-0.5 font-mono text-sm text-ink-3">
        MODE &mdash;
      </span>
    )
  }

  if (isError || data === null) {
    return (
      <span className="border border-rule-2 px-2 py-0.5 font-mono text-sm text-ink-2">
        MODE UNAVAILABLE
      </span>
    )
  }

  // Demo mode is served entirely from fixtures, so REPLAY is the only honest
  // label no matter what is_live claims in the payload. Checked BEFORE is_live
  // so a fixture edit can never put LIVE on a rehearsal.
  //
  // The badge never reads "ONLINE" in any branch. That word is reserved for
  // camera reachability and must not be reused as a mode: labelling replay
  // ONLINE is the original sin this status bar exists to prevent.
  if (data.is_live && !isDemo(appMode())) {
    return (
      <span className="bg-live px-2 py-0.5 font-mono text-sm font-medium text-white">
        LIVE
      </span>
    )
  }

  return (
    <span className="flex items-center gap-2">
      <span className="bg-replay px-2 py-0.5 font-mono text-sm font-medium text-white">
        REPLAY
      </span>
      <span className="text-sm text-ink-2">
        {SOURCE_MODE_LABEL[data.source_mode]}
      </span>
    </span>
  )
}

// Was hardcoded to "offline" with the note that this was the truth until the
// socket was wired. The socket is wired now, so the hardcoded string had
// become a lie: it read "offline" on every screen while alerts were arriving,
// and directly contradicted the Alerts status line on the same page. A global
// indicator that disagrees with the screen it sits above costs more trust
// than no indicator at all.
const SOCKET_CHIP: Record<WsStatus, { label: string; dot: string }> = {
  connecting: { label: 'Live updates connecting', dot: 'bg-ink-3' },
  connected: { label: 'Live updates on', dot: 'bg-live' },
  reconnecting: { label: 'Live updates reconnecting', dot: 'bg-degraded' },
  offline: { label: 'Live updates offline', dot: 'bg-offline' },
}

function SocketChip() {
  const status = useLiveStore(selectWsStatus)
  const chip = SOCKET_CHIP[status]
  return (
    <span className="flex items-center gap-1.5 text-sm text-ink-2">
      <span className={`inline-block size-2 rounded-full ${chip.dot}`} aria-hidden="true" />
      {chip.label}
    </span>
  )
}

// Rendered by AppLayout, never by a page, so no screen can omit it.
export function StatusBar() {
  return (
    // 3.7333rem is exactly 56px at the 15px root, and still scales when
    // .projector raises the root to 19px. A hard 56px would not.
    <header className="fixed inset-x-0 top-0 z-20 flex h-[3.7333rem] items-center gap-4 border-b border-rule bg-panel px-4">
      <span className="font-mono text-sm font-medium tracking-wide text-ink">
        TRINETRA
      </span>
      <ModeBadge />
      <div className="flex-1" />
      <SocketChip />
      <Clock />
    </header>
  )
}
