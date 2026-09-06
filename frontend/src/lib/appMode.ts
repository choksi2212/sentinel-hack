// The four VITE_APP_MODE values, in one place, so "does this mode start MSW?"
// is a question with a written answer rather than a condition buried in a
// startup function.
//
// Nothing here is secret. VITE_ variables compile into the public bundle, and
// this is a four-value switch.

export type AppMode = 'mock' | 'api' | 'demo' | 'auto'

const MODES: Record<AppMode, true> = { mock: true, api: true, demo: true, auto: true }

// `auto` is DECLARED in the contract and has no resolution rule anywhere in
// this repository or in Canonical 8.1. It is not invented here: resolving it to
// mock would silently fake data against a real backend, and resolving it to api
// would silently break an offline demo. Both failures are quiet, which is the
// worst property a mode switch can have.
//
// So it resolves to `mock` ONLY as an explicit, logged fallback, and the log
// says the rule is undefined. When Mihir settles it, this is the one place that
// changes.
export const AUTO_RESOLVES_TO: AppMode = 'mock'

export function appMode(): AppMode {
  const raw = import.meta.env.VITE_APP_MODE
  if (typeof raw === 'string' && Object.hasOwn(MODES, raw)) return raw as AppMode
  // An unset or misspelled mode is not a reason to guess. Say so and take the
  // safe branch: mock cannot corrupt a real backend.
  console.warn(
    `[trinetra] VITE_APP_MODE is "${String(raw)}", which is not one of ` +
      `mock | api | demo | auto. Falling back to "mock".`,
  )
  return 'mock'
}

// The single question the startup path asks. `api` is the only mode that talks
// to a real backend; everything else is served by MSW.
export function usesMockLayer(mode: AppMode): boolean {
  if (mode === 'api') return false
  if (mode === 'auto') return usesMockLayer(AUTO_RESOLVES_TO)
  return true
}

// Demo mode pins the fixtures so a rehearsal is identical every run. A demo
// that differs between rehearsal and stage is not a rehearsal.
export function isDemo(mode: AppMode): boolean {
  return mode === 'demo'
}
