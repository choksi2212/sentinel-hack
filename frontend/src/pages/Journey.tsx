import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { readCameras, readJourney } from '../api/adapters'
import { apiGet } from '../api/client'
import { endpoints } from '../api/endpoints'
import { ErrorPanel } from '../components/ErrorPanel'
import { buildJourneyTimeline } from '../lib/journeyTimeline'
import { normalisePlate } from '../lib/plate'
import { formatClock } from '../lib/time'
import { MapCanvas } from '../map/MapCanvas'
import type { DrawableConnector } from '../map/journeyMapLayers'
import { buildJourneyMapLayers } from '../map/journeyMapLayers'

// The plate lives in the query string, not a route param: the route is
// `/journey` and the left rail links to it bare, so a `:plate` segment would
// break that link. D-015 settled the query-string shape.
//
// Normalised client-side before it reaches the path parameter, which is named
// plate_normalized. Moved to src/lib/plate.ts when the watchlist add form
// needed the identical rule; the reasoning it carried now lives there.

function CounterLine({ children }: { children: React.ReactNode }) {
  return <li className="text-sm text-ink-2">{children}</li>
}

export function Journey() {
  const [searchParams] = useSearchParams()
  const plateParam = searchParams.get('plate') ?? ''
  const fromParam = searchParams.get('from') ?? ''
  const toParam = searchParams.get('to') ?? ''
  const plate = normalisePlate(plateParam)

  const camerasQuery = useQuery({
    queryKey: ['cameras'],
    queryFn: async () => readCameras(await apiGet<unknown>(endpoints.cameras())),
  })

  const journeyQuery = useQuery({
    queryKey: ['journey', plate, fromParam, toParam],
    queryFn: () =>
      apiGet<unknown>(endpoints.journey(plate), {
        from: fromParam,
        to: toParam,
      }),
    enabled: plate !== '',
    staleTime: 5000,
  })

  // A segment naming a camera outside the catalogue falls back to the raw id,
  // never a blank. Canonical 8.2 says a camera is marked offline rather than
  // deleted, so a miss should be rare -- but rare is not never.
  const cameraNames = useMemo(() => {
    const map = new Map<string, string>()
    for (const camera of camerasQuery.data ?? []) {
      map.set(camera.camera_id, camera.name)
    }
    return map
  }, [camerasQuery.data])

  const journey = useMemo(
    () => (journeyQuery.data === undefined ? null : readJourney(journeyQuery.data)),
    [journeyQuery.data],
  )
  const timeline = useMemo(
    () => (journey === null ? null : buildJourneyTimeline(journey)),
    [journey],
  )
  const layers = useMemo(
    () => (timeline === null ? null : buildJourneyMapLayers(timeline)),
    [timeline],
  )

  // Built as DOM rather than an HTML string so a camera name or a note from
  // the wire cannot inject markup.
  const connectorPopup = useMemo(
    () => (connector: DrawableConnector) => {
      const { segment, state } = connector
      const root = document.createElement('div')

      const route = document.createElement('p')
      route.style.fontWeight = '600'
      route.textContent = `${cameraNames.get(segment.from_camera_id) ?? segment.from_camera_id} → ${
        cameraNames.get(segment.to_camera_id) ?? segment.to_camera_id
      }`
      root.append(route)

      // Three words separate a measurement from a claim.
      const distance = document.createElement('p')
      distance.textContent = `${segment.straight_line_km} km straight line`
      root.append(distance)

      const verdict = document.createElement('p')
      if (state === 'unassessed') {
        verdict.textContent = 'Plausibility not assessed'
      } else if (segment.required_speed_kmh === undefined) {
        // Never computed from straight_line_km and elapsed_seconds, even
        // though both are present. A client-derived number would be shown as
        // if it were the backend's assessment.
        verdict.textContent = 'Speed unavailable'
      } else if (state === 'not_plausible') {
        verdict.textContent = `⚠ Not plausible — ${segment.required_speed_kmh} km/h required`
      } else {
        verdict.textContent = `${segment.required_speed_kmh} km/h required`
      }
      root.append(verdict)

      // Verbatim, and only when the server sent one. A null note on an
      // infeasible segment renders nothing; the styling and the speed carry it.
      //
      // Absent counts as null. `!== null` alone let an omitted key through, and
      // `textContent = undefined` writes the literal string "undefined" into
      // the timeline -- a word an officer would read as data.
      if (segment.note !== null && segment.note !== undefined) {
        const note = document.createElement('p')
        note.textContent = segment.note
        root.append(note)
      }

      return root
    },
    [cameraNames],
  )

  const disclaimerFooter =
    journey === null ? null : (
      <footer className="mt-4 border-t border-rule-2 pt-3 text-sm text-ink-2">
        {journey.disclaimer}
      </footer>
    )

  return (
    <section>
      <h1 className="text-[1.35rem] font-semibold text-ink">
        Journey{plate === '' ? '' : ` — ${plate}`}
      </h1>

      {plate === '' && (
        <p className="mt-2 text-ink-2">
          No journey selected. Search for a plate, then open its journey.
        </p>
      )}

      {plate !== '' && journeyQuery.isPending && (
        <div className="mt-4 h-[60vh] border border-rule bg-panel">
          <div className="flex h-full items-center justify-center">
            <p className="text-sm text-ink-2">Loading journey&hellip;</p>
          </div>
        </div>
      )}

      {journeyQuery.isError && (
        <div className="mt-4">
          <ErrorPanel
            error={journeyQuery.error}
            onRetry={() => void journeyQuery.refetch()}
          />
        </div>
      )}

      {/* readJourney returned null. D-012: a response missing its disclaimer
          makes the whole screen an error state, because the disclaimer is
          mandatory precisely so a journey cannot render without one. */}
      {!journeyQuery.isPending && !journeyQuery.isError && journeyQuery.data !== undefined && journey === null && (
        <div className="mt-4 border border-rule-2 bg-panel p-4" role="alert">
          <p className="font-medium text-ink">Journey data incomplete</p>
          <p className="mt-1 text-sm text-ink-2">
            The response was missing required fields, so nothing is shown rather
            than a partial route.
          </p>
        </div>
      )}

      {journey !== null && timeline !== null && layers !== null && (
        <>
          {timeline.nodes.length === 0 ? (
            <div className="mt-4 border border-rule bg-sunken p-4">
              <p className="text-ink">No sightings in this journey.</p>
              <p className="mt-1 text-sm text-ink-2">
                Widen the time range, or check the plate.
              </p>
            </div>
          ) : (
            <>
              <div className="mt-3 flex gap-4">
                <div className="min-w-0 flex-1 border border-rule bg-panel">
                  <div className="h-[60vh]">
                    <MapCanvas
                      sightings={layers.plottedNodes}
                      journey={layers}
                      connectorPopup={connectorPopup}
                    />
                  </div>
                </div>

                <aside className="w-[260px] shrink-0 border border-rule bg-panel p-3">
                  <h2 className="text-sm font-semibold text-ink">This journey</h2>

                  <ul className="mt-2 space-y-1">
                    <CounterLine>
                      {layers.plottedNodes.length} sightings on the map
                    </CounterLine>

                    {/* The server's claim, shown as the server's claim and
                        never reconciled against what we could plot. */}
                    {journey.sighting_count !== undefined &&
                      journey.sighting_count !== layers.plottedNodes.length && (
                        <CounterLine>
                          Server reports {journey.sighting_count} sightings for
                          this plate
                        </CounterLine>
                      )}

                    {layers.unplacedNodeCount > 0 && (
                      <CounterLine>
                        {layers.unplacedNodeCount} sightings not on the map,
                        location not surveyed
                      </CounterLine>
                    )}

                    {layers.undrawableConnectorCount > 0 && (
                      <CounterLine>
                        {layers.undrawableConnectorCount} connections not drawn,
                        camera location not surveyed
                      </CounterLine>
                    )}

                    {layers.gapCount > 0 && (
                      <CounterLine>
                        Segment data unusable — connection not shown (
                        {layers.gapCount})
                      </CounterLine>
                    )}

                    {timeline.unplacedSegments.length > 0 && (
                      <CounterLine>
                        {timeline.unplacedSegments.length} segments could not be
                        placed on this timeline
                      </CounterLine>
                    )}

                    {journey.discardedSightingCount > 0 && (
                      <CounterLine>
                        {journey.discardedSightingCount} sightings discarded as
                        unusable
                      </CounterLine>
                    )}

                    {journey.discardedSegmentCount > 0 && (
                      <CounterLine>
                        {journey.discardedSegmentCount} segments discarded as
                        unusable
                      </CounterLine>
                    )}
                  </ul>

                  <h3 className="mt-4 text-sm font-semibold text-ink">Sightings</h3>
                  <ol className="mt-1 space-y-1">
                    {timeline.nodes.map((node) => (
                      <li key={node.sighting_id} className="text-sm text-ink-2">
                        <span className="font-mono text-ink">
                          {node.plate ?? 'Unreadable'}
                        </span>{' '}
                        — {cameraNames.get(node.camera_id) ?? node.camera_id}{' '}
                        <span className="font-mono text-ink-3">
                          {formatClock(node.first_seen_at)}
                        </span>
                      </li>
                    ))}
                  </ol>
                </aside>
              </div>
            </>
          )}

          {disclaimerFooter}
        </>
      )}
    </section>
  )
}
