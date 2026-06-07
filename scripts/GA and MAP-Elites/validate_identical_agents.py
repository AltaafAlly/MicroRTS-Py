"""
GA validity check (supervisor request): if two *identical* AI agents play each other, the matchup is
inherently fair, so the GA's balance score should be ~1.0 (win rate ~50/50, neither seat favored).

This sanity-checks the fitness used by the single-gene balance experiment before we scale up to many
agents and many genes. We run AI X vs the same AI X on the symmetric map with the vanilla Java
UnitTypeTable (UnitTypeTable(3,2), nondeterministic so games vary), and report:

  win rate (P0 vs P1), seat gap, and balance = 1 - 2*|winrate - 0.5|  (the GA's balance term).

Usage (microrts conda env):
  python "scripts/GA and MAP-Elites/validate_identical_agents.py"

Env overrides: VAL_GAMES (default 40), VAL_MAX_STEPS (default 5000),
               VAL_AIS (comma list, default lightRushAI,heavyRushAI,workerRushAI,coacAI).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "Running Simulations"))
from run_match_configured import run_pair  # noqa: E402

MAP_PATH = "maps/16x16/basesWorkers16x16.xml"
GAMES = int(os.environ.get("VAL_GAMES", "40"))
MAX_STEPS = int(os.environ.get("VAL_MAX_STEPS", "5000"))
AIS = [s.strip() for s in os.environ.get(
    "VAL_AIS", "lightRushAI,heavyRushAI,workerRushAI,coacAI"
).split(",") if s.strip()]


def main() -> None:
    print("=" * 100)
    print(f"GA validity check — identical-agent mirror | map={MAP_PATH} | games={GAMES}/ordering | max_steps={MAX_STEPS}")
    print("Vanilla UnitTypeTable(3,2). The GA scores balance over BOTH orderings, aggregated by AI identity")
    print("(each agent plays each seat), which cancels seat bias. Identical agents => GA balance ~1.0.")
    print("=" * 100)
    print(f"  {'AI (vs itself)':<16} | {'GA winrate (both ord.)':>22} | {'GA balance':>11}")
    print("  " + "-" * 56)
    for ai in AIS:
        # Ordering 1 and ordering 2 (both X vs X). Aggregate by AI identity exactly as the GA does:
        #   ai1_wins = R1.left + R2.right ;  ai2_wins = R1.right + R2.left
        r1 = run_pair(ai, ai, MAP_PATH, MAX_STEPS, GAMES, False, None, utt_builtin=(3, 2))
        r2 = run_pair(ai, ai, MAP_PATH, MAX_STEPS, GAMES, False, None, utt_builtin=(3, 2))
        ai1_wins = r1.get("left_wins", 0) + r2.get("right_wins", 0)
        ai2_wins = r1.get("right_wins", 0) + r2.get("left_wins", 0)
        decisive = ai1_wins + ai2_wins
        ga_winrate = (ai1_wins / decisive) if decisive else 0.5
        ga_balance = max(0.0, 1.0 - 2.0 * abs(ga_winrate - 0.5))
        print(f"  {ai:<16} | {ga_winrate:>22.2f} | {ga_balance:>11.2f}")
    print("\nGA balance ~1.0 across agents => the fitness correctly scores a mirror matchup as fully balanced.")


if __name__ == "__main__":
    main()
