#!/usr/bin/env python3
"""Single-file local verification UI for datasets/trinetra-hard/index.jsonl.

Not a web app: stdlib http.server + inline HTML/JS in this one file. Shows the
cropped plate region and the candidate string, keyboard-driven:
  Enter      accept candidate text as-is -> label_source: human, confidence: certain
  (type)     edit the text box, then Enter -> accept the correction
  x          mark ineligible (no readable plate) -> eligible: false
  ?          mark probable (readable but not certain) -> label_confidence: probable
  Left/Right move between rows without labeling

Writes back to index.jsonl after every action (progress survives a browser
close -- see save_rows()). Run:
  py -3.11 scripts/verify_ui.py
then open http://127.0.0.1:8765/
"""
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "datasets" / "trinetra-hard" / "index.jsonl"
PORT = 8765


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_rows(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)  # atomic on the same filesystem -- no half-written index.jsonl


def crop_png_bytes(row: dict) -> bytes:
    """Render the plate crop for one row as PNG bytes."""
    import cv2
    import numpy as np
    from PIL import Image

    frame_path = row["frame_path"]
    x, y, w, h = row["plate_bbox"]
    if "#row" in frame_path:
        parquet_rel, row_part = frame_path.split("#row")
        import pandas as pd

        df = pd.read_parquet(ROOT / parquet_rel)
        img_bytes = df.iloc[int(row_part)]["image"]["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
        x0, y0 = max(int(x), 0), max(int(y), 0)
        x1, y1 = min(int(x + w), img_np.shape[1]), min(int(y + h), img_np.shape[0])
        crop = img_np[y0:y1, x0:x1]
        ok, buf = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        return buf.tobytes()
    else:
        # whole-image source (e.g. synthetic_plates) -- serve the file directly
        with open(ROOT / frame_path, "rb") as f:
            return f.read()


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>TRINETRA-HARD verify</title>
<style>
body { font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 2rem; }
#crop { max-width: 90vw; max-height: 40vh; image-rendering: pixelated; border: 2px solid #555; background: #222; }
#text { font-size: 1.5rem; padding: 0.4rem; width: 20rem; }
.row { margin: 0.6rem 0; }
#progress { color: #8f8; }
#status { color: #ff8; min-height: 1.4rem; }
kbd { background: #333; padding: 0.1rem 0.4rem; border-radius: 3px; }
</style></head>
<body>
<div id="progress">loading...</div>
<div class="row"><img id="crop" src=""></div>
<div class="row">obs_id: <b id="obs_id"></b> | slice: <b id="slice"></b> | width_bucket: <b id="width_bucket"></b></div>
<div class="row"><input id="text" autocomplete="off"></div>
<div class="row">
  <kbd>Enter</kbd> accept &nbsp;
  <kbd>x</kbd> ineligible &nbsp;
  <kbd>?</kbd> mark probable &nbsp;
  <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> navigate
</div>
<div id="status"></div>
<script>
let i = 0, rows = [];
const $ = (id) => document.getElementById(id);

async function loadState() {
  const res = await fetch('/api/rows');
  rows = await res.json();
  i = rows.findIndex(r => r.label_source !== 'human');
  if (i === -1) i = 0;
  render();
}

function render() {
  if (rows.length === 0) { $('progress').textContent = 'no rows'; return; }
  const r = rows[i];
  const done = rows.filter(x => x.label_source === 'human').length;
  $('progress').textContent = `${done}/${rows.length} verified (row ${i+1}/${rows.length})`;
  $('obs_id').textContent = r.obs_id;
  $('slice').textContent = r.slice;
  $('width_bucket').textContent = r.width_bucket;
  $('text').value = r.plate_text || '';
  $('crop').src = `/api/image/${r.obs_id}?t=${Date.now()}`;
  $('status').textContent = r.label_source === 'human' ? `already verified: ${r.label_confidence}${r.eligible ? '' : ' (ineligible)'}` : '';
  $('text').focus();
}

async function label(patch) {
  const r = rows[i];
  Object.assign(r, patch);
  await fetch(`/api/label/${r.obs_id}`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(patch)
  });
  $('status').textContent = 'saved';
  if (i < rows.length - 1) i++;
  render();
}

$('text').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    label({ plate_text: $('text').value.trim(), label_source: 'human',
             label_confidence: rows[i].label_confidence === 'probable' ? 'probable' : 'certain',
             eligible: true });
  } else if (e.key === 'x' && e.target.value === '') {
    label({ label_source: 'human', eligible: false, plate_text: null, label_confidence: 'certain' });
  } else if (e.key === '?') {
    rows[i].label_confidence = 'probable';
    $('status').textContent = 'marked probable -- press Enter to save';
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault(); i = Math.max(0, i - 1); render();
  } else if (e.key === 'ArrowRight') {
    e.preventDefault(); i = Math.min(rows.length - 1, i + 1); render();
  }
});

loadState();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    rows: list[dict] = []

    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet during a labeling session

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/rows":
            self._json(Handler.rows)
        elif path.startswith("/api/image/"):
            obs_id = path.split("/")[-1]
            row = next((r for r in Handler.rows if r["obs_id"] == obs_id), None)
            if row is None:
                self.send_response(404)
                self.end_headers()
                return
            try:
                png = crop_png_bytes(row)
            except Exception as exc:  # noqa: BLE001 -- surface the failure, don't crash the server
                self._json({"error": str(exc)}, code=500)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        if path.startswith("/api/label/"):
            obs_id = path.split("/")[-1]
            length = int(self.headers.get("Content-Length", 0))
            patch = json.loads(self.rfile.read(length) or b"{}")
            for row in Handler.rows:
                if row["obs_id"] == obs_id:
                    row.update(patch)
                    break
            save_rows(INDEX_PATH, Handler.rows)  # continuous save
            self._json({"ok": True})
        else:
            self.send_response(404)
            self.end_headers()


def main() -> int:
    Handler.rows = load_rows(INDEX_PATH)
    if not Handler.rows:
        print(f"BLOCKED: no rows in {INDEX_PATH}", file=sys.stderr)
        return 1
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"verify_ui: {len(Handler.rows)} rows loaded, serving http://127.0.0.1:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def demo():
    """Self-check: assert-based, round-trip load/save with a temp index."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "index.jsonl"
        rows = [
            {"obs_id": "th_0001", "label_source": "ocr_candidate", "plate_text": "AB"},
            {"obs_id": "th_0002", "label_source": "ocr_candidate", "plate_text": "CD"},
        ]
        save_rows(p, rows)
        loaded = load_rows(p)
        assert loaded == rows
        loaded[0]["label_source"] = "human"
        loaded[0]["plate_text"] = "AB1"
        save_rows(p, loaded)
        reloaded = load_rows(p)
        assert reloaded[0]["plate_text"] == "AB1"
        assert reloaded[0]["label_source"] == "human"
        assert reloaded[1]["label_source"] == "ocr_candidate"  # untouched row survives
    print("demo: all assertions passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        sys.exit(main())
