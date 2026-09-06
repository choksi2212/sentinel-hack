import { Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'

// The last line of defence, and deliberately only that.
//
// It exists because an unguarded error during render unmounts the WHOLE React
// tree: three separate field renames in the hostile harness each turned one bad
// lookup into a blank projector. A boundary turns that into a broken panel
// under a working shell, which a presenter can talk over and click past.
//
// It is IN ADDITION to the guarded lookups, never instead of them. Catching a
// crash here means something already read a wire value as if it were a type;
// the honest fix is still the guard at the read. Anything landing in this
// boundary is a bug to go and find, not a state to design around.
//
// A class because React has no hook equivalent: componentDidCatch and
// getDerivedStateFromError are class-only. No dependency added.

interface Props {
  children: ReactNode
  // Changes when the route changes, so navigating away from a broken screen
  // clears the error instead of trapping the operator on it.
  resetKey: string
}

interface State {
  error: Error | null
}

export class ScreenErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Full stack to the console, always. The operator gets copy they can act
    // on; whoever debugs it afterwards needs the component stack, and a
    // boundary that swallows it trades one blank screen for a silent one.
    console.error(
      '[trinetra] a screen crashed and was caught by the error boundary',
      error,
      info.componentStack,
    )
  }

  componentDidUpdate(previous: Props): void {
    if (previous.resetKey !== this.props.resetKey && this.state.error !== null) {
      this.setState({ error: null })
    }
  }

  render(): ReactNode {
    const { error } = this.state
    if (error === null) return this.props.children

    return (
      <section role="alert" className="border border-rule-2 bg-panel p-4">
        <h1 className="text-[1.35rem] font-semibold text-ink">This screen failed to render</h1>
        <p className="mt-2 text-ink">
          The rest of the dashboard is still working. Use the left rail to open
          another screen, or reload to try this one again.
        </p>
        <p className="mt-3 text-sm text-ink-2">
          The full error and component stack are in the browser console.
        </p>
        {/* The message verbatim, monospaced, so it can be read off a projector
            and quoted into a bug report without a screenshot. */}
        <p className="mt-3 break-words border border-rule bg-sunken p-3 font-mono text-sm text-ink-2">
          {error.message === '' ? error.name : `${error.name}: ${error.message}`}
        </p>
        <button
          type="button"
          onClick={() => this.setState({ error: null })}
          className="mt-3 border border-rule-2 px-3 py-1 text-ink hover:bg-sunken"
        >
          Try this screen again
        </button>
      </section>
    )
  }
}
