#!/usr/bin/env python3
"""Real-footage stability diagnostic (Phase 4S item 4). NOT an accuracy
measurement -- indian_road's unverified_real rows have no known true plate
string (eligible: false always), so "correct" is undefined here. What IS
measurable: does fusion make the predictor's output more self-consistent
across frames of the same TrackKey = (camera_id, stream_session_id, track_id)?

Agreement for a track = (count of its most common non-null prediction) /
(frame count). A predictor with a stub-level temporal aggregation should show
higher agreement with fusion on than off. This validates the stability-
reporting plumbing; it says nothing about whether predicted strings are
correct.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def track_key(row: dict) -> tuple:
    return (row["camera_id"], row["stream_session_id"], row["track_id"])


def load_unverified_real(dataset: str) -> list[dict]:
    path = ROOT / "datasets" / dataset / "index.jsonl"
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r.get("label_source") == "unverified_real"]


def stub_stability_predict(row: dict, fusion_enabled: bool, rng: random.Random) -> str | None:
    """Illustrative stub only (SPEC_BENCHMARK §5 spirit): per-frame single-shot
    guesses are noisy; fusion simulates temporal aggregation converging on one
    answer per track. Not a real OCR/fusion model."""
    if not fusion_enabled:
        return rng.choice(["GUESS_A", "GUESS_B", "GUESS_C", None])
    # fusion: same seed per TrackKey -> same "converged" answer every frame
    track_rng = random.Random(str(track_key(row)))
    return track_rng.choice(["GUESS_A", "GUESS_B", "GUESS_C"])


def agreement(predictions: list[str | None]) -> tuple[float, int]:
    non_null = [p for p in predictions if p is not None]
    if not non_null:
        return 0.0, 0
    top_count = Counter(non_null).most_common(1)[0][1]
    return top_count / len(predictions), len(predictions)


def build_report(rows: list[dict], seed: int) -> dict:
    by_track: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_track[track_key(r)].append(r)

    per_track = []
    for key, track_rows in sorted(by_track.items()):
        rng_off = random.Random(f"{seed}:{key}:off")
        rng_on = random.Random(f"{seed}:{key}:on")
        preds_off = [stub_stability_predict(r, False, rng_off) for r in track_rows]
        preds_on = [stub_stability_predict(r, True, rng_on) for r in track_rows]
        agree_off, n = agreement(preds_off)
        agree_on, _ = agreement(preds_on)
        per_track.append({
            "track_key": list(key), "n_frames": n,
            "agreement_off": round(agree_off, 3), "agreement_on": round(agree_on, 3),
            "improved": agree_on > agree_off,
        })

    n_tracks = len(per_track)
    n_improved = sum(1 for t in per_track if t["improved"])
    mean_off = sum(t["agreement_off"] for t in per_track) / n_tracks if n_tracks else None
    mean_on = sum(t["agreement_on"] for t in per_track) / n_tracks if n_tracks else None
    return {
        "n_tracks": n_tracks,
        "n_frames_total": sum(t["n_frames"] for t in per_track),
        "mean_agreement_off": mean_off,
        "mean_agreement_on": mean_on,
        "n_tracks_improved": n_improved,
        "per_track": per_track,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "**Predictor: `stub_stability_predict`** (`benchmarks/stability.py`, canned/illustrative "
        "— not a real fusion model. Replace before citing this table as a measurement.)",
        "",
        "# STABILITY — real-footage diagnostic",
        "",
        "**This is a stability diagnostic, not an accuracy measurement.** "
        "indian_road's `unverified_real` rows have no known true plate string "
        "(`eligible: false` always) -- correctness is undefined here. This table "
        "only asks whether fusion makes predictions more self-consistent across "
        "frames of the same TrackKey. It carries no accuracy claim; see "
        "`FUSION_DELTA.md` for that (synthetic_truth only).",
        "",
        f"Tracks: {report['n_tracks']} | Frames: {report['n_frames_total']} | "
        f"Tracks with improved agreement (fusion on vs off): "
        f"{report['n_tracks_improved']}/{report['n_tracks']}"
        if report['n_tracks'] else "No tracks.",
        "",
        (f"Mean agreement -- fusion OFF: {report['mean_agreement_off']:.3f} | "
         f"fusion ON: {report['mean_agreement_on']:.3f}") if report['n_tracks'] else "",
        "",
        "| TrackKey (camera, session, track) | n_frames | agreement OFF | agreement ON | improved |",
        "|---|---|---|---|---|",
    ]
    for t in report["per_track"][:50]:  # cap the printed table, full data is in the JSON twin
        key = " / ".join(str(x) for x in t["track_key"])
        lines.append(
            f"| {key} | {t['n_frames']} | {t['agreement_off']} | {t['agreement_on']} | "
            f"{'yes' if t['improved'] else 'no'} |"
        )
    if len(report["per_track"]) > 50:
        lines.append(f"\n... {len(report['per_track']) - 50} more tracks omitted from this table, see STABILITY.json")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="trinetra-hard")
    p.add_argument("--out", default="benchmarks/reports/STABILITY.md")
    p.add_argument("--seed", type=int, default=20260905)
    args = p.parse_args(argv)

    rows = load_unverified_real(args.dataset)
    report = build_report(rows, args.seed)

    out_md = ROOT / args.out
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), encoding="utf-8")
    out_json = out_md.with_suffix(".json")
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_md} and {out_json} ({report['n_tracks']} tracks)")
    return 0


def demo():
    rows = [
        {"camera_id": "c1", "stream_session_id": "s1", "track_id": 1},
        {"camera_id": "c1", "stream_session_id": "s1", "track_id": 1},
        {"camera_id": "c1", "stream_session_id": "s1", "track_id": 1},
    ]
    report = build_report(rows, seed=1)
    assert report["n_tracks"] == 1
    assert report["per_track"][0]["n_frames"] == 3
    assert report["mean_agreement_on"] == 1.0  # fusion stub always converges within a track
    print("demo: all assertions passed")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        sys.exit(main([a for a in sys.argv[1:] if a != "--demo"]))
