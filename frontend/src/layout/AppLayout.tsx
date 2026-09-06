import { useEffect } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { ScreenErrorBoundary } from '../components/ScreenErrorBoundary'
import { LeftRail } from './LeftRail'
import { StatusBar } from './StatusBar'

// Ctrl+Alt+P toggles html.projector, which raises the root font from 15px to
// 19px (src/styles/theme.css). Every size in the app is rem-based, so one class
// enlarges the whole interface for the back of the room.
//
// The class already existed and NOTHING turned it on, which made a working
// feature unreachable -- and legibility at five metres is a submission
// checklist item, not a nicety.
//
// WHY THIS CHORD. Two modifiers, so it cannot be produced by ordinary typing,
// and P for projector so it is recallable while standing in front of a room.
// Ctrl+Shift+P is deliberately avoided: Firefox binds it to a private window.
//
// The editable-target guard is not redundant. On layouts where AltGr is sent as
// Ctrl+Alt, AltGr+P is a printable character, so on those keyboards this chord
// IS reachable by typing. The guard costs one branch and removes the whole
// class of problem. The consequence, stated plainly: with focus in the plate
// filter or the camera dropdown the shortcut does nothing -- click the page
// background first.
//
// Not persisted. CONVENTIONS forbids localStorage and sessionStorage, so this
// resets on reload. That is also the honest default: projector mode is for a
// projector, and the presenter turns it on once one is connected.
function useProjectorToggle() {
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (!event.ctrlKey || !event.altKey) return
      if (event.key.toLowerCase() !== 'p') return

      const target = event.target
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        (target instanceof HTMLElement && target.isContentEditable)
      ) {
        return
      }

      event.preventDefault()
      const on = document.documentElement.classList.toggle('projector')
      console.info(
        `[trinetra] projector mode ${on ? 'ON, root 19px' : 'OFF, root 15px'}`,
      )
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])
}

// The status bar and the rail live here, not in any page, so no screen can
// ship without them.
export function AppLayout() {
  // The ROUTER's location, not window.location. The global does not change on
  // a client-side navigation, so keying the boundary on it would leave a
  // crashed screen stuck after navigating away.
  const routerLocation = useLocation()

  useProjectorToggle()

  return (
    <div className="min-h-screen bg-paper text-ink">
      <StatusBar />
      {/* Clears the fixed status bar. Same 3.7333rem (56px) it uses. */}
      <div className="flex min-h-screen pt-[3.7333rem]">
        <LeftRail />
        {/* min-w-0 keeps a wide child from forcing the page to scroll
            sideways at 1024px. */}
        <main className="min-w-0 flex-1 p-5">
          {/* INSIDE the layout, wrapping only the routed content. A boundary
              around the whole app would take the StatusBar and the left rail
              down with the screen, and a blank projector is the worst possible
              outcome -- the badge, the clock and the navigation are exactly
              what a presenter needs when one screen has failed.
              Keyed on the pathname so navigating away clears the error. */}
          <ScreenErrorBoundary resetKey={routerLocation.pathname}>
            <Outlet />
          </ScreenErrorBoundary>
        </main>
      </div>
    </div>
  )
}
