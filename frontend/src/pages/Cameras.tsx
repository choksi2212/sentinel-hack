import { useQuery } from '@tanstack/react-query'
import { readCameras } from '../api/adapters'
import { apiGet } from '../api/client'
import { endpoints } from '../api/endpoints'
import { CameraStatusChip } from '../components/CameraStatusChip'
import { ErrorPanel } from '../components/ErrorPanel'
import { hasCoordinates } from '../lib/geo'
import { formatAbsoluteTime, formatRelativeTime } from '../lib/time'
import type { Camera } from '../types/ui'

function CameraCard({ camera }: { camera: Camera }) {
  // Same predicate the map counts with, so the grid and the map's "unplaced
  // cameras" line cannot drift apart.
  const unsurveyed = !hasCoordinates(camera)

  return (
    <li className="border border-rule bg-panel p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate font-medium text-ink" title={camera.name}>
            {camera.name}
          </h2>
          <p className="font-mono text-sm text-ink-3">{camera.camera_id}</p>
        </div>
        <CameraStatusChip status={camera.status} />
      </div>

      {/* "Last seen never seen" is not a sentence. A camera we have never
          heard from says so plainly. */}
      <p
        className="mt-3 text-sm text-ink-2"
        title={formatAbsoluteTime(camera.last_seen_at)}
      >
        {/* Absent and null both mean never seen. `=== null` alone let an
            omitted key through to the else branch. */}
        {camera.last_seen_at === null || camera.last_seen_at === undefined
          ? 'Never seen'
          : `Last seen ${formatRelativeTime(camera.last_seen_at)}`}
      </p>

      {/* The same camera appears in the map's unplaced list. Stated, not
          hidden, and never given invented coordinates. */}
      {unsurveyed && (
        <p className="mt-1 text-sm text-ink-3">Location not surveyed</p>
      )}
    </li>
  )
}

function SkeletonCard() {
  return (
    <li className="border border-rule bg-panel p-3">
      <div className="h-4 w-2/3 bg-sunken" />
      <div className="mt-2 h-3 w-1/3 bg-sunken" />
      <div className="mt-4 h-3 w-1/2 bg-sunken" />
    </li>
  )
}

export function Cameras() {
  const query = useQuery({
    queryKey: ['cameras'],
    queryFn: async () => readCameras(await apiGet<unknown>(endpoints.cameras())),
  })

  // Canonical 6.4 returns a bare array. There is no wrapper object, so there
  // is nothing to unwrap.
  const cameras = query.data ?? []

  return (
    <section>
      <h1 className="text-[1.35rem] font-semibold text-ink">Cameras</h1>
      <p className="mt-1 text-sm text-ink-2">
        A camera that leaves the catalogue is marked offline and kept, because
        historical sightings still reference it. An offline camera with recent
        sightings behind it is correct, not stale.
      </p>

      {query.isPending && (
        <ul className="mt-4 grid grid-cols-[repeat(auto-fill,minmax(15rem,1fr))] gap-3">
          {Array.from({ length: 6 }, (_, index) => (
            <SkeletonCard key={index} />
          ))}
        </ul>
      )}

      {query.isError && (
        <div className="mt-4">
          <ErrorPanel error={query.error} onRetry={() => void query.refetch()} />
        </div>
      )}

      {!query.isPending && !query.isError && cameras.length === 0 && (
        <p className="mt-4 border border-rule bg-sunken p-4 text-ink-2">
          No cameras in the registry. Nothing has been synced from the camera
          catalogue yet.
        </p>
      )}

      {!query.isPending && !query.isError && cameras.length > 0 && (
        <>
          <p className="mt-4 text-sm text-ink-2">
            {cameras.length} cameras
          </p>
          <ul className="mt-2 grid grid-cols-[repeat(auto-fill,minmax(15rem,1fr))] gap-3">
            {cameras.map((camera) => (
              <CameraCard key={camera.camera_id} camera={camera} />
            ))}
          </ul>
        </>
      )}
    </section>
  )
}
