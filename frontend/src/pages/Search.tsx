import { useQuery } from '@tanstack/react-query'
import { useVirtualizer } from '@tanstack/react-virtual'
import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { readCameras, readSearchResponse } from '../api/adapters'
import { apiGet } from '../api/client'
import { endpoints } from '../api/endpoints'
import { ErrorPanel } from '../components/ErrorPanel'
import { MatchStateChip } from '../components/MatchStateChip'
import { formatAbsoluteTime, formatClock } from '../lib/time'
import type { VehicleSighting } from '../types/api'


const ROW_HEIGHT = 76
const LIST_MAX_HEIGHT = 520

type Candidate = VehicleSighting & { distance: number }

function NoSnapshot() {
  // Never a broken image. No endpoint serves snapshots yet, so every one of
  // these is null today -- which means this placeholder is on EVERY search row,
  // and its legibility is not a detail.
  //
  // text-sm, not text-[0.7rem]. theme.css sets the floor at 13px, and 0.7rem
  // was 10.5px at a 15px root -- smaller than the text-xs that rule bans by
  // name. "image" rather than "snapshot" because the longer word does not fit
  // a 48px box at the larger size.
  return (
    <div className="flex size-12 shrink-0 items-center justify-center border border-rule bg-sunken text-center text-sm leading-tight text-ink-3">
      No
      <br />
      image
    </div>
  )
}

function ResultRow({
  sighting,
  candidate,
}: {
  sighting: VehicleSighting
  candidate?: boolean
}) {
  // Absent takes the same path as null. `=== null` alone let an omitted plate
  // through as neither unreadable nor readable, and the cell rendered BLANK --
  // which CONVENTIONS.md forbids outright: never a guess, never a blank cell.
  const unreadable = sighting.plate === null || sighting.plate === undefined

  // Precedence matters and is not obvious. `plate` null means we could not
  // read it, and CONVENTIONS.md rule 6 says that renders "Unreadable" -- showing
  // plate_raw there would put a half-guessed string like "GJ0?A?12??" in
  // front of an officer as if it were a plate. So unreadable wins first.
  // Only when we DID read it do we show plate_raw, which is what the reader
  // actually saw, falling back to the resolved plate because plate_raw is
  // independently nullable and a blank cell is forbidden too.
  // `??` already covers absent as well as null on plate_raw. The final
  // 'Unreadable' is unreachable given the guard above and is there so this
  // expression cannot evaluate to undefined under any future edit -- a blank
  // cell must be impossible by construction, not by argument.
  const plateDisplay = unreadable
    ? 'Unreadable'
    : (sighting.plate_raw ?? sighting.plate ?? 'Unreadable')

  return (
    <div
      className={[
        'flex items-center gap-3 border-b border-rule px-3 py-2',
        candidate ? 'border-l-4 border-l-lowconf bg-sunken pl-5' : 'bg-panel',
      ].join(' ')}
      style={{ height: ROW_HEIGHT }}
    >
      {/* Every snapshot_uri is null today: no endpoint serves them yet. When
          one does, the image goes here and this stays as the fallback. It is
          never a broken image. */}
      <NoSnapshot />

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          {/* plate_raw is what the reader actually saw. A plate we could not
              resolve reads "Unreadable" -- never blank, never a guess. The
              row still renders: a vehicle that passed and could not be
              identified is information (Canonical 3.2). */}
          <span
            className={
              unreadable
                ? 'font-mono text-ink-2'
                : 'font-mono font-medium text-ink'
            }
          >
            {plateDisplay}
          </span>
          {candidate && (
            <span className="border border-lowconf px-1.5 py-0.5 text-sm font-medium text-lowconf">
              Candidate
            </span>
          )}
        </div>
        <div className="mt-0.5 flex flex-wrap items-baseline gap-x-3 text-sm text-ink-2">
          <span>{sighting.camera_name}</span>
          <span
            className="font-mono"
            title={`${formatAbsoluteTime(sighting.first_seen_at)} to ${formatAbsoluteTime(sighting.last_seen_at)}`}
          >
            {formatClock(sighting.first_seen_at)} &rarr;{' '}
            {formatClock(sighting.last_seen_at)}
          </span>
          {/* Explains uncertainty: a narrow plate is a hard read. */}
          <span className="text-ink-3">{sighting.plate_width_px ?? '--'} px wide</span>
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-3">
        {candidate && 'distance' in sighting && (
          <span className="font-mono text-sm text-lowconf">
            distance {(sighting as Candidate).distance}
          </span>
        )}
        <MatchStateChip
          state={sighting.match_state}
          observations={sighting.evidence_count}
        />
      </div>
    </div>
  )
}

// One rendering path, always virtualized, so the virtualized path is the one
// that gets exercised in development rather than only appearing past 100
// rows in production. The container collapses to fit short lists, so a
// three-row result does not sit in a tall empty scrollbox.
function VirtualRows({
  sightings,
  candidate,
}: {
  sightings: VehicleSighting[]
  candidate?: boolean
}) {
  const parentRef = useRef<HTMLDivElement | null>(null)
  const virtualizer = useVirtualizer({
    count: sightings.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 8,
  })

  const total = virtualizer.getTotalSize()

  return (
    <div
      ref={parentRef}
      className="overflow-y-auto border-x border-t border-rule"
      style={{ height: Math.min(total, LIST_MAX_HEIGHT) }}
    >
      <div className="relative w-full" style={{ height: total }}>
        {virtualizer.getVirtualItems().map((item) => {
          const sighting = sightings[item.index]
          if (sighting === undefined) return null
          return (
            <div
              key={sighting.sighting_id}
              className="absolute inset-x-0 top-0"
              style={{ transform: `translateY(${item.start}px)` }}
            >
              <ResultRow sighting={sighting} candidate={candidate} />
            </div>
          )
        })}
      </div>
    </div>
  )
}

function FilterChip({ label, onRemove }: { label: string; onRemove: () => void }) {
  return (
    <span className="inline-flex items-center gap-1.5 border border-rule bg-panel px-2 py-1 text-sm text-ink-2">
      {label}
      <button
        type="button"
        onClick={onRemove}
        className="rounded-[4px] px-1 text-ink-3 hover:bg-sunken hover:text-ink"
        aria-label={`Remove filter ${label}`}
      >
        &times;
      </button>
    </span>
  )
}

export function Search() {
  const [searchParams, setSearchParams] = useSearchParams()

  const plateParam = searchParams.get('plate') ?? ''
  const fromParam = searchParams.get('from') ?? ''
  const toParam = searchParams.get('to') ?? ''
  const cameraParam = searchParams.get('camera_id') ?? ''
  const fuzzy = searchParams.get('fuzzy') === 'true'

  // Local input state so typing stays responsive; the URL is the source of
  // truth for the query, updated 300ms after the last keystroke. Filters live
  // in the query string so a demo state is bookmarkable and a mid-demo reload
  // lands where it was.
  const [plateInput, setPlateInput] = useState(plateParam)

  function setParam(key: string, value: string) {
    setSearchParams(
      (previous) => {
        const next = new URLSearchParams(previous)
        if (value === '') next.delete(key)
        else next.set(key, value)
        return next
      },
      { replace: true },
    )
  }

  useEffect(() => {
    const id = window.setTimeout(() => setParam('plate', plateInput), 300)
    return () => window.clearTimeout(id)
    // setParam is stable enough for this: it only closes over setSearchParams.

  }, [plateInput])

  const camerasQuery = useQuery({
    queryKey: ['cameras'],
    queryFn: async () => readCameras(await apiGet<unknown>(endpoints.cameras())),
  })

  const searchQuery = useQuery({
    queryKey: ['search', plateParam, fromParam, toParam, cameraParam, fuzzy],
    queryFn: async () =>
      readSearchResponse(
        await apiGet<unknown>(endpoints.search(), {
        plate: plateParam,
        from: fromParam,
        to: toParam,
          camera_id: cameraParam,
          fuzzy: fuzzy ? 'true' : '',
        }),
      ),
    staleTime: 5000,
  })

  const response = searchQuery.data ?? null
  const results = response?.results ?? []
  const candidates = response?.candidates ?? []

  const activeFilters: Array<{ key: string; label: string }> = []
  if (plateParam) activeFilters.push({ key: 'plate', label: `Plate: ${plateParam}` })
  if (fromParam) activeFilters.push({ key: 'from', label: `From: ${fromParam}` })
  if (toParam) activeFilters.push({ key: 'to', label: `To: ${toParam}` })
  if (cameraParam)
    activeFilters.push({ key: 'camera_id', label: `Camera: ${cameraParam}` })
  if (fuzzy) activeFilters.push({ key: 'fuzzy', label: 'Fuzzy matching on' })

  function clearAll() {
    setPlateInput('')
    setSearchParams(new URLSearchParams(), { replace: true })
  }

  return (
    <section>
      <h1 className="text-[1.35rem] font-semibold text-ink">Search</h1>

      {/* Filters stay interactive while results load. */}
      <div className="mt-3 flex flex-wrap items-end gap-3 border border-rule bg-panel p-3">
        <label className="flex flex-col gap-1 text-sm text-ink-2">
          Plate
          <input
            type="text"
            value={plateInput}
            onChange={(event) => setPlateInput(event.target.value)}
            placeholder="GJ 01 ab 1234"
            className="w-52 rounded-[4px] border border-rule bg-panel px-2 py-1.5 font-mono text-ink"
          />
        </label>

        <label className="flex flex-col gap-1 text-sm text-ink-2">
          From
          <input
            type="datetime-local"
            value={fromParam}
            onChange={(event) => setParam('from', event.target.value)}
            className="rounded-[4px] border border-rule bg-panel px-2 py-1.5 text-ink"
          />
        </label>

        <label className="flex flex-col gap-1 text-sm text-ink-2">
          To
          <input
            type="datetime-local"
            value={toParam}
            onChange={(event) => setParam('to', event.target.value)}
            className="rounded-[4px] border border-rule bg-panel px-2 py-1.5 text-ink"
          />
        </label>

        <label className="flex flex-col gap-1 text-sm text-ink-2">
          Camera
          <select
            value={cameraParam}
            onChange={(event) => setParam('camera_id', event.target.value)}
            className="rounded-[4px] border border-rule bg-panel px-2 py-1.5 text-ink"
          >
            <option value="">All cameras</option>
            {(camerasQuery.data ?? []).map((camera) => (
              <option key={camera.camera_id} value={camera.camera_id}>
                {camera.camera_id} — {camera.name}
              </option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-2 py-1.5 text-sm text-ink-2">
          <input
            type="checkbox"
            checked={fuzzy}
            onChange={(event) => setParam('fuzzy', event.target.checked ? 'true' : '')}
          />
          Include fuzzy candidates
        </label>
      </div>

      {searchQuery.isError && (
        <div className="mt-4">
          <ErrorPanel
            error={searchQuery.error}
            onRetry={() => void searchQuery.refetch()}
          />
        </div>
      )}

      {searchQuery.isPending && (
        <div className="mt-4 border border-rule">
          {Array.from({ length: 6 }, (_, index) => (
            <div
              key={index}
              className="flex items-center gap-3 border-b border-rule bg-panel px-3 py-2"
              style={{ height: ROW_HEIGHT }}
            >
              <div className="size-12 shrink-0 bg-sunken" />
              <div className="flex-1">
                <div className="h-4 w-40 bg-sunken" />
                <div className="mt-2 h-3 w-64 bg-sunken" />
              </div>
            </div>
          ))}
        </div>
      )}

      {!searchQuery.isPending && !searchQuery.isError && response !== null && (
        <>
          {/* Normalisation shown back to the user. Free honesty about what
              was actually searched. */}
          {response.query.normalized !== '' && (
            <p className="mt-4 text-sm text-ink-2">
              Searched as{' '}
              <span className="font-mono text-ink">{response.query.normalized}</span>
              {response.query.plate !== response.query.normalized && (
                <> — you typed “{response.query.plate}”</>
              )}
            </p>
          )}

          {/* A silently shorter list asserts fewer sightings than the server
              sent. Both counts reach the screen, and the server's own `count`
              is shown as the server's claim rather than reconciled with what
              survived. */}
          {(response.discardedResults > 0 ||
            response.discardedCandidates > 0 ||
            response.count !== results.length + response.discardedResults) && (
            <p className="mt-2 text-sm text-offline">
              Server reported {response.count} matches; {results.length} shown
              {response.discardedResults > 0 && (
                <>, {response.discardedResults} unusable and dropped on read</>
              )}
              {response.discardedCandidates > 0 && (
                <>, {response.discardedCandidates} candidates dropped</>
              )}
              . See contract drift on System status.
            </p>
          )}

          {results.length === 0 && candidates.length === 0 ? (
            <div className="mt-3 border border-rule bg-sunken p-4">
              <p className="text-ink">No sightings match these filters.</p>
              {activeFilters.length > 0 && (
                <>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {activeFilters.map((filter) => (
                      <FilterChip
                        key={filter.key}
                        label={filter.label}
                        onRemove={() => {
                          if (filter.key === 'plate') setPlateInput('')
                          setParam(filter.key, '')
                        }}
                      />
                    ))}
                  </div>
                  <button
                    type="button"
                    onClick={clearAll}
                    className="mt-3 rounded-[4px] border border-rule-2 bg-panel px-3 py-1.5 text-sm text-ink hover:bg-sunken"
                  >
                    Clear filters
                  </button>
                </>
              )}
            </div>
          ) : (
            <>
              {/* count comes from the response, not results.length. */}
              <h2 className="mt-4 border-b border-rule-2 pb-1 text-ink">
                Exact matches — {response.count}
              </h2>
              {results.length > 0 ? (
                <VirtualRows sightings={results} />
              ) : (
                <p className="border-x border-b border-rule bg-panel p-3 text-sm text-ink-2">
                  No exact matches.
                </p>
              )}

              {/* Separate region, always after exact, never interleaved. When
                  fuzzy is off it does not render at all -- not an empty
                  heading. Canonical 6.2. */}
              {fuzzy && candidates.length > 0 && (
                <>
                  <h2 className="mt-6 flex flex-wrap items-baseline justify-between gap-x-4 border-b border-rule-2 pb-1 text-ink">
                    <span>Fuzzy candidates — {candidates.length}</span>
                    <span className="text-sm font-normal text-lowconf">
                      Candidates require review. Not confirmed matches.
                    </span>
                  </h2>
                  <VirtualRows sightings={candidates} candidate />
                </>
              )}
            </>
          )}
        </>
      )}
    </section>
  )
}
