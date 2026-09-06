// Every API path in one file. Three documents disagree on two of them
// (see the cross-lane findings memo, F3 and F4), so when the real answer
// arrives the change is one line here, not fifty across the app.
const V1 = "/api/v1";

export const endpoints = {
  search: () => `${V1}/search/vehicles`,
  searchNearby: () => `${V1}/search/vehicles/nearby`,
  journey: (plateNormalized: string) =>
    `${V1}/journey/${encodeURIComponent(plateNormalized)}`,
  cameras: () => `${V1}/cameras`,
  camera: (id: string) => `${V1}/cameras/${id}`,
  cameraSync: () => `${V1}/cameras/sync`,
  cameraPreview: (id: string) => `${V1}/cameras/${id}/preview.m3u8`,
  watchlist: () => `${V1}/watchlist`,
  watchlistItem: (id: string) => `${V1}/watchlist/${id}`,
  alerts: () => `${V1}/alerts`,
  alertAcknowledge: (id: string) => `${V1}/alerts/${id}/acknowledge`,
  systemStatus: () => `${V1}/system/status`,
  metricsBenchmark: () => `${V1}/metrics/benchmark`,
  healthLive: () => `/health/live`,
  healthReady: () => `/health/ready`,
} as const;

export const wsAlerts = () => `/ws/alerts`;