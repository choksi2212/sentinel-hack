#!/usr/bin/env node
// Standalone mock WebSocket server. NOT imported by the app.
//
// A real killable process rather than an intercepted socket, because the claim
// worth proving is "it reconnects after the server dies", and you cannot
// Ctrl+C an interception. msw 2.15 does ship a WebSocket API (`ws` from
// msw/core/ws) -- this is a choice, not a workaround.
//
// Node 20 compatible: no fs.globSync, no syntax newer than Node 20.
import { WebSocketServer } from "ws";

// Defaults to 8000 so the VITE_WS_BASE_URL already in .env.local
// (ws://localhost:8000) works with no config change, and so the path matches
// endpoints.wsAlerts() -> /ws/alerts exactly as the real backend will.
// argv: [port] [--go-quiet <seconds>] [--burst <n>]
const args = process.argv.slice(2);

// Indices consumed as a flag's VALUE, so the positional port scan does not
// mistake them for the port. Without this, `--go-quiet 10` bound the server to
// port 10 and every reconnect test silently measured the wrong thing.
const consumed = new Set();

function flagValue(name) {
  const i = args.indexOf(name);
  if (i === -1) return null;
  consumed.add(i);
  const raw = args[i + 1];
  const n = Number(raw);
  if (!Number.isFinite(n)) {
    console.error(`[mock-ws] ${name} needs a number, got ${raw ?? "nothing"}`);
    process.exit(1);
  }
  consumed.add(i + 1);
  return n;
}

// After N seconds, stop sending EVERYTHING including heartbeats, but hold every
// socket OPEN -- no close, no error. This simulates a NAT holding a dead
// connection, or a backend that accepted the upgrade and then wedged. It is
// the only way to make the client's watchdog fire, and without it the watchdog
// is untested code.
const goQuietAfter = flagValue("--go-quiet");

// Send n alerts as fast as the socket allows on each connection, each with a
// distinct alert_id, then resume normal cadence. Exercises the 100 cap through
// real socket traffic rather than by driving the store directly.
const burstCount = flagValue("--burst");

// --hostile: send EVERY frame under `payload` instead of `data`, matching the
// backend manual rather than Canonical. Normally only every 5th alert does, so
// the tolerant path gets exercised without hiding the canonical one; under
// --hostile there is no canonical frame at all, which is what integration day
// looks like if this disagreement is never settled.
const hostile = args.includes("--hostile");

// Computed AFTER every flagValue call, so `consumed` is fully populated.
const positional = args.find(
  (a, i) => !consumed.has(i) && !a.startsWith("--") && Number.isFinite(Number(a)),
);
const port = Number(positional ?? 8000);

let quiet = false;

const PATH = "/ws/alerts";

const wss = new WebSocketServer({ port, path: PATH });

// Lowercase cam01..cam30 only. scripts/ is not in the guard's scan set --
// guard.mjs walks "src" only -- so this is correctness, not compliance.
const CAMERAS = [
  ["cam01", "Ashram Road junction"],
  ["cam04", "Paldi circle"],
  ["cam07", "Maninagar station approach"],
  ["cam12", "Odhav ring road"],
  ["cam19", "Sardar Bridge north"],
];
const PLATES = ["GJ01AB1234", "GJ05CD5678", "GJ18KL9012", "GJ27MN3456"];
// Alert.match_state is Exclude<MatchState, "low_confidence" | "unreadable">.
// The database enforces that with a CHECK, so those two are never emitted and
// no client path is built for them.
const MATCH_STATES = ["exact", "probable"];
const PRIORITIES = ["low", "medium", "high", "critical"];

let alertSeq = 0;
let sightingSeq = 0;

function pick(list, seq) {
  return list[seq % list.length];
}

function makeAlert() {
  alertSeq += 1;
  const camera = pick(CAMERAS, alertSeq);
  return {
    alert_id: `alr-live-${String(alertSeq).padStart(4, "0")}`,
    plate: pick(PLATES, alertSeq),
    camera_id: camera[0],
    camera_name: camera[1],
    match_state: pick(MATCH_STATES, alertSeq),
    confidence: null,
    // priority, not severity. acknowledged boolean, not acknowledged_at.
    priority: pick(PRIORITIES, alertSeq),
    created_at: new Date().toISOString(),
    acknowledged: false,
    snapshot_uri: null,
  };
}

function makeSighting() {
  sightingSeq += 1;
  const camera = pick(CAMERAS, sightingSeq);
  const first = new Date();
  const last = new Date(first.getTime() + 1800 + (sightingSeq % 17) * 100);
  return {
    sighting_id: `sig-live-${String(sightingSeq).padStart(4, "0")}`,
    camera_id: camera[0],
    camera_name: camera[1],
    lat: 23.02 + (sightingSeq % 7) * 0.004,
    lon: 72.57 + (sightingSeq % 5) * 0.004,
    first_seen_at: first.toISOString(),
    last_seen_at: last.toISOString(),
    source_pts_ms: sightingSeq * 1000,
    source_mode: "synthetic",
    plate: pick(PLATES, sightingSeq),
    plate_raw: pick(PLATES, sightingSeq),
    confidence: null,
    match_state: pick(MATCH_STATES, sightingSeq),
    evidence_count: 1 + (sightingSeq % 5),
    plate_width_px: 40 + (sightingSeq % 60),
    vehicle_type: "car",
    snapshot_uri: null,
  };
}

const sockets = new Set();

function broadcastRaw(text) {
  // Once quiet, nothing is sent -- but every socket stays OPEN. That is the
  // whole point: a closed socket fires onclose and the client reconnects
  // immediately, which is a different code path from the watchdog.
  if (quiet) return;
  for (const socket of sockets) {
    if (socket.readyState === socket.OPEN) socket.send(text);
  }
}

function broadcast(object) {
  broadcastRaw(JSON.stringify(object));
}

wss.on("connection", (socket) => {
  sockets.add(socket);
  console.log(`[mock-ws] client connected (${sockets.size} open)`);

  // No replay, no backfill. That is the backend's job, and faking it here
  // would prove nothing about the client's reconnect refetch.

  // A type the client has never heard of, once per connection. It must be
  // logged once and ignored: a schema addition must not white-screen mid-demo.
  if (!quiet) {
    socket.send(
      JSON.stringify({
        type: "vehicle_update",
        data: { plate: "GJ01AB1234", note: "a type this client does not know" },
      }),
    );
  }

  // Burst mode: n distinct alerts as fast as the socket takes them, so the
  // 100 cap is exercised by real traffic instead of by driving the store.
  if (burstCount !== null && !quiet) {
    for (let i = 0; i < burstCount; i += 1) {
      const alert = makeAlert();
      alert.alert_id = `alr-burst-${String(i).padStart(4, "0")}`;
      socket.send(JSON.stringify({ type: "alert", data: alert }));
    }
    console.log(`[mock-ws] burst of ${burstCount} alerts sent`);
  }

  // One deliberately malformed frame per CONNECTION, 20s in: valid-looking
  // JSON truncated mid-object. Scheduled per connection rather than once per
  // server lifetime, because a client that connects after the server has been
  // up a while would otherwise never see it and the catch path would go
  // untested -- which is exactly what happened on the first run.
  const malformed = setTimeout(() => {
    // Must respect the quiet gate. This used to call send() directly, so a
    // socket that reconnected during a quiet run got a frame 20s later, which
    // reset the client's watchdog and made the quiet simulation a lie.
    if (!quiet && socket.readyState === socket.OPEN) {
      socket.send('{"type":"alert","data":{"alert_id":"alr-trunc","plate":"GJ0');
      console.log("[mock-ws] sent one truncated frame");
    }
  }, 20_000);

  socket.on("close", () => {
    clearTimeout(malformed);
    sockets.delete(socket);
    console.log(`[mock-ws] client gone (${sockets.size} open)`);
  });
  socket.on("error", (error) => {
    console.log(`[mock-ws] socket error: ${error.message}`);
  });
});

// heartbeat: ts at the TOP LEVEL, no wrapper. A parser that blindly reads
// msg.data.x must throw here. That is the whole point of this shape.
const heartbeatTimer = setInterval(() => {
  broadcast({ type: "heartbeat", ts: new Date().toISOString() });
}, 15_000);

const alertTimer = setInterval(() => {
  // Return before logging, not after. The log used to print "sent" for frames
  // broadcast() had already dropped, so a quiet run's log claimed traffic that
  // never left the process.
  if (quiet) return;
  const alert = makeAlert();
  // Every 5th alert uses `payload` instead of `data`, so the data ?? payload
  // path is exercised rather than assumed. Canonical says data, the backend
  // manual says payload, and that disagreement is unresolved.
  if (hostile || alertSeq % 5 === 0) {
    broadcast({ type: "alert", payload: alert });
    console.log(`[mock-ws] alert ${alert.alert_id} sent under "payload"`);
  } else {
    broadcast({ type: "alert", data: alert });
  }
}, 8_000);

const sightingTimer = setInterval(() => {
  broadcast({ type: "sighting", data: makeSighting() });
}, 3_000);

const systemTimer = setInterval(() => {
  broadcast({
    type: "system",
    data: { api: "ok", postgres: true, redis: true, note: "periodic system frame" },
  });
}, 30_000);

function shutdown() {
  console.log("\n[mock-ws] shutting down");
  clearInterval(heartbeatTimer);
  clearInterval(alertTimer);
  clearInterval(sightingTimer);
  clearInterval(systemTimer);
  for (const socket of sockets) socket.close(1001, "server shutting down");
  wss.close(() => process.exit(0));
  // Do not wait forever for a lingering socket.
  setTimeout(() => process.exit(0), 500);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

if (goQuietAfter !== null) {
  setTimeout(() => {
    quiet = true;
    console.log(
      `[mock-ws] GOING QUIET after ${goQuietAfter}s -- no more frames, sockets held OPEN`,
    );
    console.log("[mock-ws] the client's watchdog should now fire on its own");
  }, goQuietAfter * 1000);
}

wss.on("listening", () => {
  console.log(`[mock-ws] listening on ws://localhost:${port}${PATH}`);
  console.log("[mock-ws] heartbeat 15s / alert 8s / sighting 3s / system 30s");
  if (hostile) console.log('[mock-ws] --hostile: every frame under "payload", never "data"');
  if (goQuietAfter !== null) console.log(`[mock-ws] --go-quiet ${goQuietAfter}s`);
  if (burstCount !== null) console.log(`[mock-ws] --burst ${burstCount}`);
  console.log("[mock-ws] Ctrl+C to stop");
});
