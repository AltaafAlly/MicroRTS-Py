#!/usr/bin/env python3
"""
Compare two Light.moveTime settings (min vs max of current PARAMETER_BOUNDS) on a
fixed matchup: few games each, prints fitness and raw moveTime.

Run from anywhere:
  python diagnose_light_move_time_impact.py

Optional env (same family as single-gene GA):
  SINGLE_GENE_PARAM_MIN / SINGLE_GENE_PARAM_MAX — override Light.moveTime range for contrast.

Requires JVM / gym_microrts like the main GA scripts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_GA_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_GA_DIR) not in sys.path:
    sys.path.insert(0, str(_GA_DIR))


def main() -> None:
    os.chdir(_PROJECT_ROOT)

    from core.ga_chromosome import MicroRTSChromosome
    from run_single_gene_balance_decrease import (  # noqa: E402
        build_evaluator,
        compute_gene_index,
        denormalize,
        midpoint_genome,
    )

    pmin_e = os.environ.get("SINGLE_GENE_PARAM_MIN")
    pmax_e = os.environ.get("SINGLE_GENE_PARAM_MAX")
    lo, hi = MicroRTSChromosome.PARAMETER_BOUNDS["Light"]["moveTime"]
    if pmin_e is not None:
        lo = int(pmin_e)
    if pmax_e is not None:
        hi = int(pmax_e)
    if lo >= hi:
        raise SystemExit(f"Invalid moveTime bounds: [{lo}, {hi}]")
    MicroRTSChromosome.PARAMETER_BOUNDS["Light"] = dict(MicroRTSChromosome.PARAMETER_BOUNDS["Light"])
    MicroRTSChromosome.PARAMETER_BOUNDS["Light"]["moveTime"] = (lo, hi)

    spec = compute_gene_index("Light", "moveTime")
    games = int(os.environ.get("DIAG_MOVE_TIME_GAMES", "8"))
    use_both = os.environ.get("DIAG_MOVE_TIME_BOTH_ORDERINGS", "1") == "1"

    ev = build_evaluator(
        ai1="lightRushAI",
        ai2="workerRushAI",
        map_path="maps/10x10/basesWorkers10x10.xml",
        games_per_eval=games,
        use_both_orderings=use_both,
        max_steps=int(os.environ.get("SINGLE_GENE_MAX_STEPS", "20000")),
    )
    if hasattr(ev, "close_cached_env"):
        ev.close_cached_env()

    def run_at(norm: float):
        g = midpoint_genome()
        g[spec.index] = max(0.0, min(1.0, norm))
        chrom = MicroRTSChromosome.from_genome(g)
        raw = denormalize(spec, norm)
        fit = ev.evaluate_chromosome(chrom)
        if hasattr(ev, "close_cached_env"):
            ev.close_cached_env()
        return raw, fit

    print(f"Light.moveTime denormalized range: [{lo}, {hi}] (lower = faster)")
    print(f"games_per_eval={games} use_both_orderings={use_both}\n")

    for label, norm in (("fast (min raw)", 0.0), ("slow (max raw)", 1.0)):
        raw, fit = run_at(norm)
        print(f"--- {label}: genome={norm:.3f} -> moveTime={int(round(raw))}")
        print(f"    overall_fitness={fit.overall_fitness:.4f}  balance={fit.balance:.4f}  duration={fit.duration:.4f}")
if __name__ == "__main__":
    main()
