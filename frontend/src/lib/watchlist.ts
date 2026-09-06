import type { WatchlistDraft, WatchlistEntry } from '../types/ui'
import { normalisePlate } from './plate'

// PURE. Validation and the optimistic shapes live here so they can be
// exercised without a browser, and so the form and the mutation cannot
// disagree about what "valid" means.

export interface DraftProblem {
  field: 'plate' | 'reason'
  message: string
}

// Honest validation: reject empty, and do not silently transform beyond
// normalisation. The normalised plate is returned so the form can SHOW it
// before submitting -- telling the user that "gj 01 ab 1234" will be watched as
// "GJ01AB1234" is free honesty about what is actually being stored, and it is
// the difference between a typo caught now and a plate that never alerts.
export function validateDraft(raw: WatchlistDraft): {
  problems: DraftProblem[]
  normalised: WatchlistDraft
} {
  const plate = normalisePlate(raw.plate)
  const reason = raw.reason.trim()
  const problems: DraftProblem[] = []

  if (plate === '') {
    problems.push({
      field: 'plate',
      message: 'Enter a plate. Letters and digits only are kept.',
    })
  }
  if (reason === '') {
    problems.push({
      field: 'reason',
      message: 'Enter a reason. It is shown to whoever acts on the alert.',
    })
  }

  return { problems, normalised: { plate, reason, priority: raw.priority } }
}

// The optimistic row. Its id is a marker, not a guess at what the server will
// assign: nothing may key off it beyond removing it when the real list lands.
export const OPTIMISTIC_ID_PREFIX = 'wl-pending-'

export function optimisticEntry(draft: WatchlistDraft): WatchlistEntry {
  return {
    id: `${OPTIMISTIC_ID_PREFIX}${draft.plate}`,
    plate: draft.plate,
    reason: draft.reason,
    priority: draft.priority,
  }
}

export function isOptimistic(entry: WatchlistEntry): boolean {
  return entry.id.startsWith(OPTIMISTIC_ID_PREFIX)
}
