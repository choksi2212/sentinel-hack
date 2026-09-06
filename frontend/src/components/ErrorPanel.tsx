import { toDisplayError } from '../lib/errors'

// Every failure surface in the app goes through here, so request_id can never
// be forgotten on one screen and remembered on another. The same value is in
// the backend log line, and during a live demo it is the fastest way to find
// out what actually happened.
export function ErrorPanel({
  error,
  onRetry,
  operation,
}: {
  error: unknown
  onRetry?: () => void
  // What the failed call was doing. Defaults to a read, so every existing
  // caller is unchanged. Writes pass "write" so a 404 renders as the failure it
  // is rather than as a no-results state.
  operation?: 'read' | 'write'
}) {
  const display = toDisplayError(error, { operation: operation ?? 'read' })
  const isEmpty = display.kind === 'empty'

  return (
    <div
      className={[
        'border p-4',
        isEmpty ? 'border-rule bg-sunken' : 'border-rule-2 bg-panel',
      ].join(' ')}
      role={isEmpty ? undefined : 'alert'}
    >
      <p className="font-medium text-ink">{display.title}</p>
      <p className="mt-1 text-sm text-ink-2">{display.detail}</p>

      {display.requestId !== null && (
        <p className="mt-2 font-mono text-sm text-ink-3">
          Request id {display.requestId}
        </p>
      )}

      {display.retry && onRetry !== undefined && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 rounded-[4px] border border-rule-2 bg-panel px-3 py-1.5 text-sm text-ink hover:bg-sunken"
        >
          Retry
        </button>
      )}
    </div>
  )
}
