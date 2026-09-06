/// <reference types="vite/client" />

// Vite's default ImportMetaEnv types every variable as `any`, which the
// guard script forbids inside src/api/. Declaring them properly is both
// the fix and the correct practice.
// Values are from Canonical Contracts 8.1.
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_WS_BASE_URL: string;
  readonly VITE_APP_MODE: "mock" | "api" | "demo" | "auto";
  // Basemap source. Absent or anything else means "osm", so the online map is
  // what you get unless offline was asked for explicitly. Not a secret: a
  // two-value switch, and VITE_ variables compile into the public bundle.
  readonly VITE_BASEMAP?: "osm" | "offline";
  // Serve the execution manuals' field spellings instead of Canonical 6.5's,
  // for every unresolved conflict at once. A rehearsal for integration day.
  // Absent or anything else means "canonical". Not a secret: a two-value switch.
  readonly VITE_MOCK_SHAPE?: "canonical" | "hostile";
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}