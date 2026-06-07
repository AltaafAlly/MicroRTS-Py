#!/usr/bin/env python3
"""
Rank maps by *seat asymmetry* for a fixed AI pair (default: lightRushAI vs workerRushAI).

For each map we run:
  • Ordering A: ai1 left, ai2 right  → ai1_win_rate_o1 = left_wins / decisive
  • Ordering B: ai2 left, ai1 right → ai1_win_rate_o2 = right_wins / decisive

seat_asymmetry = |rate_o1 - rate_o2|
  • Near 0: AI1 wins ~similarly often whether it starts left or right → **good** for
    stat-sweep experiments (stats can matter without pure mirror 0% vs 100%).
  • Near 1: one seating is almost always won by the same side → **strong map/seat bias**
    (your 16–0 / 0–16 mirror).

Outputs a CSV sorted by seat_asymmetry ascending (best maps first).

Env (optional):
  MAP_SWEEP_AI1, MAP_SWEEP_AI2  (default lightRushAI, workerRushAI)
  MAP_SWEEP_GAMES               (default 24; lower = faster, noisier)
  MAP_SWEEP_MAX_STEPS           (default 20000; match GA)
  MAP_SWEEP_MAX_MAPS            (default 40; cap for long runs)
  MAP_SWEEP_GLOB_SUBSTR         (default "basesWorkers"; only paths containing this)
  MAP_SWEEP_SKIP_ERRORS=1       (default: skip map if JVM/XML error)

Run from repo root:
  python scripts/Running\\ Simulations/sweep_maps_seat_balance.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _discover_maps(microrts_root: Path, glob_substr: str, max_maps: int) -> list[str]:
    rels: list[str] = []
    for p in sorted(microrts_root.rglob("*.xml")):
        path_s = p.as_posix()
        if glob_substr not in p.name and glob_substr not in path_s:
            continue
        try:
            r = p.relative_to(microrts_root).as_posix()
        except ValueError:
            continue
        rels.append(r)
        if len(rels) >= max_maps:
            break
    return rels


def main() -> None:
    root = _project_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sim_dir = Path(__file__).resolve().parent
    if str(sim_dir) not in sys.path:
        sys.path.insert(0, str(sim_dir))

    ai1 = os.environ.get("MAP_SWEEP_AI1", "lightRushAI").strip()
    ai2 = os.environ.get("MAP_SWEEP_AI2", "workerRushAI").strip()
    games = int(os.environ.get("MAP_SWEEP_GAMES", "24"))
    max_steps = int(os.environ.get("MAP_SWEEP_MAX_STEPS", "20000"))
    max_maps = int(os.environ.get("MAP_SWEEP_MAX_MAPS", "40"))
    glob_substr = os.environ.get("MAP_SWEEP_GLOB_SUBSTR", "basesWorkers")
    skip_errors = os.environ.get("MAP_SWEEP_SKIP_ERRORS", "1") == "1"
    utt_p0 = os.environ.get("MAP_SWEEP_UTT_P0", "utts/TestUnitTypeTable.json")
    utt_p1 = os.environ.get("MAP_SWEEP_UTT_P1", "utts/TestUnitTypeTable.json")

    microrts_root = root / "gym_microrts" / "microrts"
    if not microrts_root.is_dir():
        raise SystemExit(f"Expected MicroRTS root at {microrts_root}")

    map_paths = _discover_maps(microrts_root, glob_substr, max_maps)
    if not map_paths:
        raise SystemExit("No maps matched; widen MAP_SWEEP_GLOB_SUBSTR or check microrts/maps.")

    from run_match_configured import run_pair  # noqa: E402

    rows: list[dict] = []
    print(f"Maps to evaluate: {len(map_paths)}  games/order={games}  ai1={ai1} ai2={ai2}\n", flush=True)

    for i, rel in enumerate(map_paths):
        print(f"[{i+1}/{len(map_paths)}] {rel} ...", flush=True)
        try:
            r1 = run_pair(
                ai1, ai2, rel, max_steps, games, False, None, utt_p0, utt_p1
            )
            r2 = run_pair(
                ai2, ai1, rel, max_steps, games, False, None, utt_p0, utt_p1
            )
        except Exception as e:
            if skip_errors:
                print(f"  SKIP: {e}", flush=True)
                rows.append(
                    {
                        "map_path": rel,
                        "error": str(e),
                        "seat_asymmetry": 999.0,
                        "ord1_ai1_rate": "",
                        "ord2_ai1_rate": "",
                    }
                )
                continue
            raise

        o1_l, o1_r, o1_d = (
            int(r1.get("left_wins", 0)),
            int(r1.get("right_wins", 0)),
            int(r1.get("draws", 0)),
        )
        o2_l, o2_r, o2_d = (
            int(r2.get("left_wins", 0)),
            int(r2.get("right_wins", 0)),
            int(r2.get("draws", 0)),
        )
        d1 = o1_l + o1_r
        d2 = o2_l + o2_r
        rate_o1 = (o1_l / d1) if d1 > 0 else 0.5
        rate_o2 = (o2_r / d2) if d2 > 0 else 0.5
        asym = abs(rate_o1 - rate_o2)
        agg_ai1 = (o1_l + o2_r) / (d1 + d2) if (d1 + d2) > 0 else 0.5

        rows.append(
            {
                "map_path": rel,
                "seat_asymmetry": asym,
                "ord1_ai1_wins": o1_l,
                "ord1_ai2_wins": o1_r,
                "ord1_draws": o1_d,
                "ord1_ai1_rate": round(rate_o1, 4),
                "ord2_ai1_wins": o2_r,
                "ord2_ai2_wins": o2_l,
                "ord2_draws": o2_d,
                "ord2_ai1_rate": round(rate_o2, 4),
                "aggregate_ai1_win_rate": round(agg_ai1, 4),
                "error": "",
            }
        )
        print(
            f"  asym={asym:.3f}  o1_ai1%={rate_o1:.2f}  o2_ai1%={rate_o2:.2f}  agg_ai1={agg_ai1:.2f}",
            flush=True,
        )

    rows_ok = [r for r in rows if r.get("error") == ""]
    rows_ok.sort(key=lambda r: float(r["seat_asymmetry"]))
    rows_bad = [r for r in rows if r.get("error") != ""]

    out_dir = root / "scripts" / "Running Simulations" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "map_seat_balance_sweep.csv"
    fieldnames = [
        "map_path",
        "seat_asymmetry",
        "ord1_ai1_rate",
        "ord2_ai1_rate",
        "aggregate_ai1_win_rate",
        "ord1_ai1_wins",
        "ord1_ai2_wins",
        "ord1_draws",
        "ord2_ai1_wins",
        "ord2_ai2_wins",
        "ord2_draws",
        "error",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows_ok + rows_bad:
            w.writerow(r)

    print(f"\nWrote {out_csv}", flush=True)
    print(
        "\nPick maps with **low** seat_asymmetry first for single-gene UTT experiments.\n"
        "Then set e.g. SINGLE_GENE_MAP_PATH=maps/8x8/basesWorkers8x8.xml",
        flush=True,
    )
    if rows_ok:
        best = rows_ok[0]
        print(
            f"\nBest (lowest asym): {best['map_path']}  asym={best['seat_asymmetry']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
