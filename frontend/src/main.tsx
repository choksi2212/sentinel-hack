import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.tsx'
import { apiBaseUrl } from './api/client'
import { AUTO_RESOLVES_TO, appMode, usesMockLayer } from './lib/appMode'
import { connectLiveSocket } from './lib/wsClient'
import './styles/theme.css'

// TanStack Query owns all data fetching. Nothing else in the app may fetch,
// because two systems fetching means two caches that disagree on a
// projector. refetchOnWindowFocus is off: alt-tabbing during a demo must not
// silently change what is on screen.
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      refetchOnWindowFocus: false,

      // networkMode "always" instead of the default "online". The default
      // model is wrong for this app twice over: in mock mode MSW answers
      // from a service worker and no network is involved at all, and in api
      // mode the backend is on a LAN, where navigator.onLine says nothing
      // about whether it is reachable. Let the request run and fail honestly
      // rather than be withheld on a signal that does not apply.
      //
      // What this does NOT do, measured rather than assumed: it does not stop
      // a query pausing between retries. TanStack pauses a retry whenever the
      // document is unfocused, and that check does not consult networkMode.
      // A backgrounded tab therefore sits on skeletons until it is focused
      // again, at which point the retry resumes and the error renders. That
      // is acceptable -- nobody is reading a screen they are not looking at
      // -- but it is not fixed here, and no comment should imply otherwise.
      networkMode: 'always',

      // Three retries with exponential backoff is roughly seven seconds
      // before an error appears, which on a projector reads as a hang. One
      // retry still absorbs a dropped packet, surfaces a real outage in about
      // a second, and shortens the window in which an unfocused tab can be
      // parked mid-retry.
      retry: 1,
    },
  },
})

function render(): void {
  const rootElement = document.getElementById('root')

  // Was `document.getElementById('root')!`. The assertion turned a missing
  // element into a throw inside an async function, which became an unhandled
  // rejection and a white screen.
  if (rootElement === null) {
    document.body.textContent =
      'The application could not start. The page is missing its root element. Reload, and report this if it continues.'
    return
  }

  createRoot(rootElement).render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </QueryClientProvider>
    </StrictMode>,
  )
}

// The worker is dynamically imported so MSW never reaches the bundle in any
// mode but "mock". Rendering waits for it, otherwise the first requests race
// the service worker and fall through to the network.
async function startMocks(): Promise<void> {
  const { worker, onUnhandledRequest } = await import('./mocks/browser')
  await worker.start({ onUnhandledRequest })

  // The drift smoke test used to run here. Its own comment said to remove it
  // "once System Status renders driftSummary() for real", and that screen now
  // does.
  //
  // It had to go, not merely could: it fed drifted fixtures through the
  // adapters at BOOT, so the drift counters were already at is_feasible 2,
  // plate_normalized 1, disclaimer 1, source_mode 1, to_time 1 on a fresh load
  // before any journey was opened. A status screen reporting that as observed
  // API drift would have been reporting its own diagnostic back to itself --
  // a fabricated system claim, and the zero state could never be seen.
  //
  // src/mocks/smoke.ts is now DELETED. Leaving it unimported on disk was worse
  // than it looked: a file that once wrote into a production drift counter is
  // exactly the file someone re-imports during integration to "check the
  // adapters quickly", and the counter it pollutes is the one the status screen
  // presents as observed API behaviour.
}

// Rendering is unconditional. Everything that can reject is inside the try:
// the dynamic chunk import, and worker.start(), which rejects whenever
// service workers are unavailable -- an insecure context is the one that will
// actually happen, because a laptop serving this demo over http://192.168.x.x
// on the venue LAN is not a secure origin and localhost's exemption does not
// apply. Before this, that combination rendered nothing at all.
async function start(): Promise<void> {
  const mode = appMode()
  if (mode === 'auto') {
    console.warn(
      `[trinetra] VITE_APP_MODE=auto has no defined resolution rule in Canonical 8.1 ` +
        `or in this repo. Falling back to "${AUTO_RESOLVES_TO}" and saying so, rather ` +
        `than guessing silently.`,
    )
  }
  console.log(
    `[trinetra] mode=${mode}, mock layer ${usesMockLayer(mode) ? 'ON' : 'OFF'}, ` +
      `api base ${apiBaseUrl}`,
  )

  if (usesMockLayer(mode)) {
    try {
      await startMocks()
    } catch (error) {
      console.error(
        '[trinetra] the mock layer failed to start; rendering anyway, requests will go to the real API',
        error,
      )
    }
  }

  render()

  // AFTER render, never before, and never awaited. The render call above is
  // unconditional and must stay that way: a rejection on the startup path
  // renders nothing at all, shell included, and that has already happened
  // here once. connectLiveSocket is idempotent, so StrictMode's double effect
  // cannot open two sockets.
  try {
    connectLiveSocket(queryClient)
  } catch (error) {
    console.error('[trinetra] live socket failed to start; the app continues', error)
  }
}

// Nothing above should reject: the mock path is wrapped and render() is
// synchronous. This exists so that a future edit which adds an await cannot
// silently blank the app the way the smoke test once did.
void start().catch((error) => {
  console.error('[trinetra] startup failed unexpectedly', error)
})
