#!/usr/bin/env node
// TRINETRA frontend guard.
// Fails the build on code that looks correct and is not.
// Node 20 compatible: no fs.globSync (that is Node 22+), because
// REPOSITORY section 14 runs CI on Node 20.
import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join } from "node:path";

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    if (name === "node_modules" || name === "dist" || name.startsWith(".")) continue;
    const full = join(dir, name).split("\\").join("/");   // normalise Windows separators
    if (statSync(full).isDirectory()) walk(full, out);
    else if (/\.(ts|tsx|css)$/.test(name)) out.push(full);
  }
  return out;
}

if (!existsSync("src")) {
  console.error("guard: no src/ directory. Run this from frontend/.");
  process.exit(1);
}

const files = walk("src");
const failures = [];

const RULES = [
  { id: "camera-id",  re: /CAM[-_]0*\d+|\bCam0\d|\bcam\d\b(?!\d)/,
    msg: "Canonical 1.1 locks camera ids to cam01..cam30 lowercase." },
  { id: "upstream",   re: /cctv\.corp8\.cloud|103\.250\.160\.189/,
    msg: "Frontend must never reference the upstream feed. Use the backend proxy." },
  { id: "secret-env", re: /VITE_[A-Z_]*(KEY|SECRET|PASSWORD|TOKEN|CREDENTIAL)/i,
    msg: "VITE_ vars are compiled into the public bundle." },
  { id: "storage",    re: /\b(localStorage|sessionStorage)\b/,
    msg: "Browser storage is not used in this app." },
  { id: "dead-field", re: /\b(sighting|alert|result|row)\.timestamp\b|["']timestamp["']\s*:/,
    msg: "Dead contract field. Use first_seen_at / last_seen_at." },
  // Two names, not six. These are the DATABASE column names for values whose
  // wire spelling differs: `plate_normalized` is the vehicle_sightings column
  // (Canonical 5.5) whose wire field is `plate` (6.5), and
  // `external_camera_id` is the cameras column (5.1) whose wire field is
  // `camera_id` (6.4). A naive ORM serializer emits the column name, so seeing
  // one on the wire is a real and specific risk worth guarding.
  //
  // is_feasible, health_state, acknowledged_at and coords_placeholder were
  // removed from this list after the canonical contract landed: they appear
  // NOWHERE in it -- not on the wire, not as columns. They came from execution
  // manuals the contract supersedes. Banning names the contract never uses made
  // the rule assert a conflict that does not exist.
  { id: "noncanon",   re: /\b(plate_normalized|external_camera_id)\b/,
    msg: "Database column name on the wire. Canonical 5.5/5.1 name these columns, but 6.5/6.4 send `plate` and `camera_id` -- this looks like a serializer leaking the schema. Allowed only in src/api/adapters.ts and src/mocks/fixtures/drifted.ts." },
  { id: "stale-path", re: /stats\/system|\/vehicles\/[^/]*\/journey/,
    msg: "Stale endpoint. Canonical 6.3/6.4: /api/v1/journey/{plate_normalized} and /api/v1/system/status." },
  { id: "native-hls", re: /<video[^>]*src=\{?[^}]*\.m3u8/,
    msg: "Native <video> for HLS works only in Safari. Use hls.js." },
];

const TYPED_ONLY = [
  { id: "any-in-types", re: /(:\s*any\b|as\s+any\b)/,
    msg: "`any` is forbidden in types/, api/, lib/ and mocks/." },
];

// Applied only to the rendering surface: pages/, components/, map/.
// Requiring the DOT is what makes this safe. `{ confidence }` destructuring,
// `confidence:` object keys and `confidence: number | null` declarations all
// contain the word and none of them render anything; a word-boundary rule
// matched 32 legitimate lines in this repo and would have been switched off
// the first day. This matches 0.
const RENDER_ONLY = [
  { id: "bare-confidence", re: /\{[^{}]*\.confidence\b[^{}]*\}/,
    msg: "Never render a raw confidence to a user. Canonical 4.4: these are relative evidence, not probabilities, and `0.87` on screen reads as \"87% certain\". Use <MatchStateChip state={...} observations={...} />, which has no confidence prop. If this line is a legitimate aggregate such as {rows.filter((r) => r.confidence > 0.5).length}, hoist the computation into a variable above the return -- do NOT weaken this rule." },
];

// Exempt from the noncanon rule ONLY, and never from any other rule.
// Exact paths, not substrings: a file at src/features/api/adapters.ts must
// not inherit an exemption by accident.
const NONCANON_EXEMPT = new Set([
  // Must NAME non-canonical fields in order to detect them at runtime and
  // count the drift.
  "src/api/adapters.ts",
  // Must CONTAIN non-canonical fields in order to simulate an API that
  // sends them, so the adapter fallbacks have something to fire on.
  "src/mocks/fixtures/drifted.ts",
]);

for (const rel of files) {
  // src/mocks is included because fixtures are exactly where a quick `as any`
  // gets typed to make a shape line up. drifted.ts keeps its noncanon
  // exemption below -- that is a different rule and stays scoped to it.
  const inTyped =
    rel.startsWith("src/types") ||
    rel.startsWith("src/api") ||
    rel.startsWith("src/lib") ||
    rel.startsWith("src/mocks");
  // The rendering surface. Only these files can put a number in front of an
  // officer, so only these carry the bare-confidence rule.
  const inRender =
    rel.startsWith("src/pages") ||
    rel.startsWith("src/components") ||
    rel.startsWith("src/map");
  let rules = inTyped ? [...RULES, ...TYPED_ONLY] : RULES;
  if (inRender) rules = [...rules, ...RENDER_ONLY];
  // Exempt from ONE rule, not the whole guard. These are the files most
  // tempted by `any` and by browser storage, so the rest of the ban stays on.
  if (NONCANON_EXEMPT.has(rel)) rules = rules.filter((r) => r.id !== "noncanon");
  readFileSync(rel, "utf8").split("\n").forEach((line, i) => {
    if (line.trimStart().startsWith("//")) return;
    for (const r of rules) {
      if (r.re.test(line)) {
        failures.push(rel + ":" + (i + 1) + "  [" + r.id + "] " + r.msg + "\n    " + line.trim());
      }
    }
  });
}

// Connector geometry must come from src/map/journeyLineStyle.ts.
//
// The previous version of this rule fired on dash syntax in files whose NAME
// matched /journey/i. It caught the wrong thing twice over: a file drawing a
// SOLID line uses no dash syntax at all and passed silently, which is the
// exact failure the rule exists to prevent; and a file named TimelineTrack.tsx
// was never examined.
//
// Inverted. The trigger is LINE CREATION, which a regex can see, rather than
// solidity, which it cannot. Any file that constructs a Leaflet polyline or
// renders an SVG <line>/<polyline> must import journeyLineStyle, the single
// place a dash pattern is defined. The filename filter is gone.
//
// <path> is deliberately NOT a trigger. LeftRail.tsx draws its eight nav icons
// with <path>, and CameraStatusChip uses strokeDasharray for the "unknown"
// status; including either would have failed the build on every run for
// legitimate code, and a rule that does that gets switched off. The residual
// holes are recorded in DECISIONS.md D-010 rather than papered over.
const CONNECTOR_CREATION =
  /\bL\.(polyline|polygon)\s*\(|\bnew\s+L\.(Polyline|Polygon)\s*\(|<\s*(Polyline|Polygon)\b|<\s*(line|polyline)\b/;

const CONNECTOR_MSG =
  "Connector geometry must come from src/map/journeyLineStyle.ts. Cameras cover a fraction of a percent of road-km, so everything between two of them is inference and a solid line asserts a route nobody observed. Import journeyLineStyle and spread its options: every branch is dashed, and feasible / infeasible / unassessed differ only in colour and weight.";

// The escape hatch is an actual IMPORT, not a mention. A substring check let
// a file opt out by naming the module in a comment, which is how the first
// version of this rule failed its own test file.
const IMPORTS_LINE_STYLE = /(?:from|import)\s*\(?\s*["'][^"']*journeyLineStyle["']/;

for (const rel of files.filter((f) => /\.tsx?$/.test(f))) {
  const src = readFileSync(rel, "utf8");
  if (IMPORTS_LINE_STYLE.test(src)) continue;
  src.split("\n").forEach((line, i) => {
    const trimmed = line.trimStart();
    if (trimmed.startsWith("//") || trimmed.startsWith("*")) return;
    if (CONNECTOR_CREATION.test(line)) {
      failures.push(
        rel + ":" + (i + 1) + "  [journey-style] " + CONNECTOR_MSG + "\n    " + line.trim(),
      );
    }
  });
}

if (failures.length) {
  console.error("\nGUARD FAILED\n" + failures.join("\n\n") + "\n");
  process.exit(1);
}
console.log("guard: ok (" + files.length + " files)");