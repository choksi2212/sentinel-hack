import { Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './layout/AppLayout'
import { Alerts } from './pages/Alerts'
import { Cameras } from './pages/Cameras'
import { Journey } from './pages/Journey'
import { LiveMap } from './pages/LiveMap'
import { Search } from './pages/Search'
import { SystemStatus } from './pages/SystemStatus'

// Plain <Routes>/<Route>. No data-router APIs, no loaders, no actions:
// TanStack Query owns data fetching, and two systems fetching means two
// caches that disagree.
export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route path="/" element={<Navigate to="/map" replace />} />
        <Route path="/map" element={<LiveMap />} />
        <Route path="/search" element={<Search />} />
        <Route path="/journey" element={<Journey />} />
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/cameras" element={<Cameras />} />
        <Route path="/status" element={<SystemStatus />} />
        <Route path="*" element={<Navigate to="/map" replace />} />
      </Route>
    </Routes>
  )
}
