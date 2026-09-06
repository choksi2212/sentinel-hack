import type { MatchState } from '../types/api'

// This component takes a match_state and an observation count. It does NOT
// take a confidence, and that is the point: the wrong thing is impossible
// here rather than merely discouraged. Canonical 4.4 -- confidences are
// relative evidence, not probabilities, and a bare 0.87 on screen reads as
// "87% certain" to every officer who sees it.
//
// If you find yourself wanting to pass a number in, the answer is no.
export function MatchStateChip({
  state,
  observations,
}: {
  state: MatchState
  observations: number
}) {
  const style = MATCH_STATE[state]

  return (
    <span className="inline-flex items-baseline gap-2">
      <span
        className={`inline-block border px-1.5 py-0.5 text-sm font-medium ${style.border} ${style.text}`}
      >
        {style.label}
      </span>
      <span className="text-sm text-ink-3">
        {observations === 1 ? '1 observation' : `${observations} observations`}
      </span>
    </span>
  )
}

// Written out in full so Tailwind's source scan can see every class.
const MATCH_STATE: Record<MatchState, { label: string; text: string; border: string }> = {
  exact: { label: 'Exact', text: 'text-exact', border: 'border-exact' },
  probable: { label: 'Probable', text: 'text-probable', border: 'border-probable' },
  low_confidence: {
    label: 'Low confidence',
    text: 'text-lowconf',
    border: 'border-lowconf',
  },
  unreadable: { label: 'Unreadable', text: 'text-unread', border: 'border-unread' },
}
