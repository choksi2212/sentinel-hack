import type { CameraStatus } from '../types/ui'

// Every state carries three signals: colour, a text label, and a shape.
// Never colour alone. Projector gamma flattens hue distinctions and roughly
// one man in twelve has some colour deficiency, so a chip that only differs
// by hue is a chip that some of the room cannot read.
//
// Class names are written out in full rather than built from the status
// string, because Tailwind scans source text and would not see a name
// assembled at runtime.
const STATUS: Record<CameraStatus, { label: string; text: string; shape: 'filled' | 'ringed' | 'hollow' | 'dashed' }> = {
  // Sentence case, not "ONLINE". The uppercase word is reserved against the
  // live/replay badge in the status bar -- see DECISIONS.md D-009.
  online: { label: 'Online', text: 'text-online', shape: 'filled' },
  degraded: { label: 'Degraded', text: 'text-degraded', shape: 'ringed' },
  offline: { label: 'Offline', text: 'text-offline', shape: 'hollow' },
  unknown: { label: 'Unknown', text: 'text-unknown', shape: 'dashed' },
}

function StatusShape({ shape }: { shape: 'filled' | 'ringed' | 'hollow' | 'dashed' }) {
  return (
    <svg viewBox="0 0 20 20" className="size-3.5 shrink-0" aria-hidden="true">
      {shape === 'filled' && <circle cx="10" cy="10" r="6" fill="currentColor" />}
      {shape === 'ringed' && (
        <>
          <circle cx="10" cy="10" r="4" fill="currentColor" />
          <circle cx="10" cy="10" r="8" fill="none" stroke="currentColor" strokeWidth="1.5" />
        </>
      )}
      {shape === 'hollow' && (
        <circle cx="10" cy="10" r="6" fill="none" stroke="currentColor" strokeWidth="2" />
      )}
      {shape === 'dashed' && (
        <circle
          cx="10"
          cy="10"
          r="6"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeDasharray="3 2.5"
        />
      )}
    </svg>
  )
}

export function CameraStatusChip({ status }: { status: CameraStatus }) {
  // The type says this is always a CameraStatus. The wire does not.
  //
  // Measured, not hypothetical: serving the execution manual's `health_state`
  // instead of Canonical 6.4's `status` makes camera.status undefined, and
  // STATUS[undefined].text threw here and WHITE-SCREENED THE WHOLE APP -- not
  // just this chip, every screen, because an unguarded render error unmounts
  // the tree. One renamed field in one endpoint took the entire dashboard down.
  //
  // Falling back to "unknown" is the honest reading: we genuinely do not know
  // this camera's state, which is exactly what that value means. It is not a
  // guess dressed as data, and the drift is visible because a wall of Unknown
  // chips is unmissable.
  const style = STATUS[status] ?? STATUS.unknown
  return (
    <span className={`inline-flex items-center gap-1.5 text-sm ${style.text}`}>
      <StatusShape shape={style.shape} />
      <span className="font-medium">{style.label}</span>
    </span>
  )
}
