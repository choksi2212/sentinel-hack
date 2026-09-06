const RELATIVE = new Intl.RelativeTimeFormat('en', { numeric: 'auto' })

const UNITS: Array<[Intl.RelativeTimeFormatUnit, number]> = [
  ['year', 31_536_000],
  ['month', 2_592_000],
  ['week', 604_800],
  ['day', 86_400],
  ['hour', 3_600],
  ['minute', 60],
  ['second', 1],
]

// Relative time is easier to scan on a projector than a timestamp. The
// absolute value always accompanies it in a title attribute, because
// "2 days ago" is useless in a report and the exact instant is what gets
// quoted back to a court.
// Accepts undefined as well as null. Canonical 6.4 types last_seen_at as
// `string | null`, but the execution manual OMITS the key, and a signature
// honest about null is silent about absent. Before this, an absent timestamp
// fell past the null check into `new Date(undefined)` and rendered
// "Time unreadable" -- which says the clock is broken when the truth is that
// the camera has never been seen. Two different facts, one wrong label.
export function formatRelativeTime(
  iso: string | null | undefined,
  now: Date = new Date(),
): string {
  if (iso === null || iso === undefined) return 'Never seen'

  const then = new Date(iso)
  if (Number.isNaN(then.getTime())) return 'Time unreadable'

  const seconds = Math.round((then.getTime() - now.getTime()) / 1000)
  const magnitude = Math.abs(seconds)

  if (magnitude < 45) return 'just now'

  for (const [unit, size] of UNITS) {
    if (magnitude >= size) {
      return RELATIVE.format(Math.round(seconds / size), unit)
    }
  }
  return 'just now'
}

// Time of day only. A sighting is an interval, and the two ends of it are
// usually seconds apart, so repeating the date on both would bury the part
// that differs.
export function formatClock(iso: string): string {
  const value = new Date(iso)
  if (Number.isNaN(value.getTime())) return '--:--:--'
  return value.toLocaleTimeString('en-GB', { hour12: false })
}

// Full precision, for the title attribute and anywhere a value is quoted.
export function formatAbsoluteTime(iso: string | null | undefined): string {
  // Same absent-vs-null reasoning as formatRelativeTime above.
  if (iso === null || iso === undefined) return 'Never seen'
  const value = new Date(iso)
  if (Number.isNaN(value.getTime())) return 'Time unreadable'
  return value.toLocaleString('en-GB', { hour12: false, timeZoneName: 'short' })
}
