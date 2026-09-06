#!/usr/bin/env node
// Writes PLACEHOLDER raster tiles into public/tiles/{z}/{x}/{y}.png.
//
// These are not a basemap. They are flat grey squares with a grid and a z/x/y
// label, and their only job is to prove the offline plumbing end to end: that
// the tile URL resolves, that Leaflet lays the grid out correctly, that
// attribution renders, and that nothing reaches the network. Seeing streets is
// a separate problem with a separate answer (see WORKLOG: obtaining real
// tiles), and shipping these as though they were a map would be exactly the
// kind of plausible fake this project refuses everywhere else.
//
// They are deliberately ugly for that reason. Nobody can mistake one for
// Ahmedabad.
//
// Node 20 compatible: zlib and Buffer only, no image dependency.
import { deflateSync } from "node:zlib";
import { mkdirSync, writeFileSync, rmSync, existsSync } from "node:fs";

// The demo bbox, measured from the fixture coordinates plus ~2.2 km of pad so
// panning to the edge does not reveal blank tiles.
const PAD = 0.02;
const SOUTH = 22.9871 - PAD, NORTH = 23.0521 + PAD;
const WEST = 72.5115 - PAD, EAST = 72.6647 + PAD;

// z11 is the opening view, z15 is the fitBounds cap, z16 is one level of
// scroll-zoom headroom. z17+ is 2,982 tiles and climbing; see WORKLOG.
const MIN_Z = 10;
const MAX_Z = 16;

const SIZE = 256;

// Tiles of margin around the data bbox, at every zoom.
//
// Measured, not guessed: seeding the bbox alone left 13 of 15 tiles blank at
// z11, because at low zoom the VIEWPORT is far wider than the data. A 1500px
// viewport is ~6 tiles across, while the demo bbox at z11 is 2. The margin
// covers the screen at low zoom, where tiles are cheap, and costs proportionally
// little at high zoom where the bbox already dominates.
const MARGIN = 3;

const lon2x = (lon, z) => Math.floor(((lon + 180) / 360) * 2 ** z);
const lat2y = (lat, z) => {
  const r = (lat * Math.PI) / 180;
  return Math.floor(((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * 2 ** z);
};

// --- minimal PNG writer -------------------------------------------------
const CRC_TABLE = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function crc32(buf) {
  let c = -1;
  for (let i = 0; i < buf.length; i += 1) {
    c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ -1) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
}

function png(rgb) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(SIZE, 0);
  ihdr.writeUInt32BE(SIZE, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // colour type: truecolour
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(rgb, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

// 3x5 digit glyphs, enough to letter z/x/y onto the tile.
const GLYPHS = {
  0: ["111", "101", "101", "101", "111"], 1: ["010", "110", "010", "010", "111"],
  2: ["111", "001", "111", "100", "111"], 3: ["111", "001", "111", "001", "111"],
  4: ["101", "101", "111", "001", "001"], 5: ["111", "100", "111", "001", "111"],
  6: ["111", "100", "111", "101", "111"], 7: ["111", "001", "010", "010", "010"],
  8: ["111", "101", "111", "101", "111"], 9: ["111", "101", "111", "001", "111"],
  "/": ["001", "001", "010", "100", "100"],
};

function tile(z, x, y) {
  // Raw RGB with a filter byte per row.
  const rows = [];
  for (let row = 0; row < SIZE; row += 1) {
    const line = Buffer.alloc(1 + SIZE * 3);
    for (let col = 0; col < SIZE; col += 1) {
      // Flat grey, with a lighter grid every 64px and a hard tile border, so
      // the tile seams are obvious and a misaligned grid is visible at a glance.
      const edge = row === 0 || col === 0 || row === SIZE - 1 || col === SIZE - 1;
      const grid = row % 64 === 0 || col % 64 === 0;
      const v = edge ? 150 : grid ? 205 : 226;
      const o = 1 + col * 3;
      line[o] = v;
      line[o + 1] = v;
      line[o + 2] = v;
    }
    rows.push(line);
  }

  // Label, scaled 3x, near the top-left.
  const text = `${z}/${x}/${y}`;
  let penX = 10;
  for (const ch of text) {
    const glyph = GLYPHS[ch];
    if (glyph) {
      for (let gy = 0; gy < 5; gy += 1) {
        for (let gx = 0; gx < 3; gx += 1) {
          if (glyph[gy][gx] !== "1") continue;
          for (let sy = 0; sy < 3; sy += 1) {
            for (let sx = 0; sx < 3; sx += 1) {
              const px = penX + gx * 3 + sx;
              const py = 10 + gy * 3 + sy;
              if (px >= SIZE || py >= SIZE) continue;
              const line = rows[py];
              const o = 1 + px * 3;
              line[o] = 90;
              line[o + 1] = 90;
              line[o + 2] = 90;
            }
          }
        }
      }
    }
    penX += 12;
  }

  return png(Buffer.concat(rows));
}

// --- write --------------------------------------------------------------
const out = "public/tiles";
if (existsSync(out) && process.argv.includes("--clean")) {
  rmSync(out, { recursive: true, force: true });
  console.log("removed existing public/tiles");
}

let count = 0;
let bytes = 0;
for (let z = MIN_Z; z <= MAX_Z; z += 1) {
  const span = 2 ** z;
  const clamp = (v) => Math.max(0, Math.min(span - 1, v));
  const x0 = clamp(lon2x(WEST, z) - MARGIN), x1 = clamp(lon2x(EAST, z) + MARGIN);
  const y0 = clamp(lat2y(NORTH, z) - MARGIN), y1 = clamp(lat2y(SOUTH, z) + MARGIN);
  for (let x = x0; x <= x1; x += 1) {
    mkdirSync(`${out}/${z}/${x}`, { recursive: true });
    for (let y = y0; y <= y1; y += 1) {
      const buf = tile(z, x, y);
      writeFileSync(`${out}/${z}/${x}/${y}.png`, buf);
      count += 1;
      bytes += buf.length;
    }
  }
  console.log(`z${z}: x ${x0}..${x1}, y ${y0}..${y1}`);
}

console.log(`wrote ${count} placeholder tiles, ${(bytes / 1024 / 1024).toFixed(2)} MB`);
console.log("These are NOT a basemap. See WORKLOG for obtaining real tiles.");
