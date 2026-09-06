import { useQuery } from '@tanstack/react-query'
import { readCameras, readSearchResponse } from '../api/adapters'
import { apiGet } from '../api/client'
import { endpoints } from '../api/endpoints'
import { ErrorPanel } from '../components/ErrorPanel'
import { hasCoordinates } from '../lib/geo'
import { MapCanvas } from '../map/MapCanvas'
import type { VehicleSighting } from '../types/api'

function SidebarLine({ label, value }: { label: string; value: number }) {
  return (
    <li className="flex items-baseline justify-between gap-3 border-b border-rule py-1.5 last:border-b-0">
      <span className="text-sm text-ink-2">{label}</span>
      <span className="font-mono text-sm tabular-nums text-ink">{value}</span>
    </li>
  )
}

export function LiveMap() {
  const sightingsQuery = useQuery({
    queryKey: ['sightings', 'all'],
    queryFn: async () => readSearchResponse(await apiGet<unknown>(endpoints.search())),
  })
  const camerasQuery = useQuery({
    queryKey: ['cameras'],
    queryFn: async () => readCameras(await apiGet<unknown>(endpoints.cameras())),
  })

  const sightings: VehicleSighting[] = sightingsQuery.data?.results ?? []
  const plotted = sightings.filter(hasCoordinates)
  const unplacedSightings = sightings.length - plotted.length

  const cameras = camerasQuery.data ?? []
  const unplacedCameras = cameras.filter((camera) => !hasCoordinates(camera)).length

  const isPending = sightingsQuery.isPending || camerasQuery.isPending
  const error = sightingsQuery.error ?? camerasQuery.error

  return (
    <section>
      <h1 className="text-[1.35rem] font-semibold text-ink">Live map</h1>

      {error !== null && error !== undefined ? (
        <div className="mt-3">
          <ErrorPanel
            error={error}
            onRetry={() => {
              void sightingsQuery.refetch()
              void camerasQuery.refetch()
            }}
          />
        </div>
      ) : (
        <div className="mt-3 flex gap-4">
          <div className="min-w-0 flex-1 border border-rule bg-panel">
            {isPending ? (
              <div className="flex h-[60vh] items-center justify-center">
                <p className="text-sm text-ink-2">Loading sightings&hellip;</p>
              </div>
            ) : (
              <div className="h-[60vh]">
                <MapCanvas sightings={plotted} />
              </div>
            )}
          </div>

          <aside className="w-[220px] shrink-0 border border-rule bg-panel p-3">
            <h2 className="text-sm font-semibold text-ink">On this map</h2>
            {isPending ? (
              <p className="mt-2 text-sm text-ink-2">Counting&hellip;</p>
            ) : sightings.length === 0 && cameras.length === 0 ? (
              <p className="mt-2 text-sm text-ink-2">
                No sightings or cameras have been loaded.
              </p>
            ) : (
              <>
                <ul className="mt-2">
                  <SidebarLine label="Plotted sightings" value={plotted.length} />
                  {/* Excluded, not hidden. A sighting we cannot place is
                      still a vehicle that passed a camera. */}
                  <SidebarLine label="Unplaced sightings" value={unplacedSightings} />
                  <SidebarLine label="Unplaced cameras" value={unplacedCameras} />
                </ul>
                <p className="mt-3 text-sm text-ink-3">
                  Unplaced records have no surveyed coordinates. They are
                  counted here rather than plotted at a guessed position.
                </p>
              </>
            )}
          </aside>
        </div>
      )}
    </section>
  )
}
