#!/usr/bin/env node
// Mojibake gate.
//
// This shell's `Get-Content -Raw` reads UTF-8 as ANSI, and a Set-Content
// round-trip writes the mangled bytes back. That has silently destroyed an
// em-dash twice. The gate catches the damage; this file exists so its
// DENOMINATOR STOPS MOVING.
//
// It moved before: session F scanned 45 files, session G scanned 43, and the
// difference was never stated. Reconciled afterwards -- F additionally scanned
// the four root .md files (excluding WORKLOG.md) while src/ was two files
// smaller, so 41 code + 4 docs = 45, and G scanned 43 code files and no docs.
// Nothing was ever dropped silently, but nobody could tell that from the
// numbers, and a gate you cannot audit is not a gate. Hence a fixed set,
// stated here and printed on every run.
//
// Node 20 compatible: no fs.globSync (Node 22+), because CI runs Node 20.
import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join } from "node:path";

// THE FIXED SCAN SET. Changing it is a decision, not a tweak.
//   roots:      src/ and scripts/, recursively, plus root-level *.md
//   extensions: .ts .tsx .css .mjs .js .md .json
//   skipped:    node_modules, dist, any dot-directory
//   excluded:   nothing by name. WORKLOG.md is scanned like everything else.
const ROOTS = ["src", "scripts"];
const EXT = /\.(ts|tsx|css|mjs|js|md|json)$/;

// Sequences that a UTF-8 file read as ANSI produces. Written as escapes on
// purpose: spelling them literally would make this file trip its own gate, and
// a gate that has to exempt itself by name is one edit away from exempting
// something real.
const PATTERNS = [
  "\u00c3", // a UTF-8 lead byte shown as Latin-1
  "\u00e2\u20ac", // the start of a mangled em-dash or curly quote
  "\u00c2", // a mangled non-breaking space or middot
  "\ufeff", // a byte-order mark that survived into the text
  "\ufffd", // U+FFFD: bytes lost outright rather than re-encoded
];

// A line carrying this token is documentation ABOUT the gate and legitimately
// contains the patterns. Line-scoped, not file-scoped, so a real corruption
// elsewhere in the same file is still caught.
const ALLOW = "MOJIBAKE-" + "ALLOW";

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    if (name === "node_modules" || name === "dist" || name.startsWith(".")) continue;
    const full = join(dir, name).split("\\").join("/");
    if (statSync(full).isDirectory()) walk(full, out);
    else if (EXT.test(name)) out.push(full);
  }
  return out;
}

const files = [];
for (const root of ROOTS) {
  if (existsSync(root)) files.push(...walk(root));
  else console.error(`mojibake: root "${root}" does not exist`);
}
for (const name of readdirSync(".")) {
  if (name.endsWith(".md") && statSync(name).isFile()) files.push(name);
}
files.sort();

const hits = [];
for (const file of files) {
  // Decoded as UTF-8 explicitly. Reading it any other way would reproduce the
  // very bug this gate exists to catch.
  const lines = readFileSync(file, "utf8").split("\n");
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (line === undefined || line.includes(ALLOW)) continue;
    for (const pattern of PATTERNS) {
      if (line.includes(pattern)) {
        hits.push(`${file}:${i + 1}: ${line.trim().slice(0, 90)}`);
        break;
      }
    }
  }
}

console.log(`mojibake: scan set = ${ROOTS.join(", ")} (recursive) + root *.md`);
console.log("mojibake: extensions = .ts .tsx .css .mjs .js .md .json");
console.log(`mojibake: ${files.length} files scanned, ${hits.length} hits`);
for (const hit of hits) console.log(`  ${hit}`);
process.exit(hits.length === 0 ? 0 : 1);
