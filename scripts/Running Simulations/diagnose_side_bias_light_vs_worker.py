"""
Diagnostic: lightRushAI vs workerRushAI — seat asymmetry and matchup balance per map.

Default map set is **all** ``maps/BroodWar/**/*.xml`` (StarCraft-style layouts). For each map we run:

  • Ordering 1: ai1 left, ai2 right  → AI1 win rate = left_wins / decisive
  • Ordering 2: swapped              → AI1 win rate = right_wins / decisive

We record ``seat_asymmetry = |rate_o1 - rate_o2|`` (lower = fairer seating for stat experiments)
and ``aggregate_ai1_win_rate`` (by AI identity across both orderings).

Outputs (under ``scripts/Running Simulations/diagnostics/``):

  • ``light_vs_worker_map_balance.csv`` — one row per map, sorted **best first**:
      primary: low seat_asymmetry, secondary: close to 50% aggregate (competitive matchup).

Optional per-game CSV (single map only): set ``DIAG_PER_GAME_CSV=1`` and
``DIAG_MAP_SET=single`` with ``DIAG_MAP_PATH=...`` (uses a separate env loop).

**Per-game console log:** Each game prints the winner (via ``run_pair(..., verbose_games=True)``). By default this is
**on** so you can see who won each match on each map (output also goes to the timestamped ``.txt`` log via stdout tee).
Set ``DIAG_VERBOSE_GAMES=0`` to disable for large sweeps (many maps × many games).

Env (optional):

  DIAG_MAP_SET=broodwar|single|basesworkers   (default: broodwar)
  DIAG_MAP_PATH=maps/10x10/basesWorkers10x10.xml   (used when DIAG_MAP_SET=single)
  DIAG_GAMES=24
  DIAG_MAX_STEPS=20000
  DIAG_MAX_MAPS=0            (0 = no cap; else limit number of maps)
  DIAG_SKIP_ERRORS=1
  DIAG_AI1, DIAG_AI2        (default lightRushAI, workerRushAI)
  DIAG_UTT_P0, DIAG_UTT_P1  (default utts/TestUnitTypeTable.json; used only when DIAG_USE_FE_UTT=0)
  DIAG_USE_FE_UTT=1         (default **on**: Java ``UnitTypeTable(3,1)`` = FE **Nondeterministic-Both**; ignores JSON paths)
  DIAG_PER_GAME_CSV=0
  DIAG_VERBOSE_GAMES=        (unset = on; 0 to silence per-game lines on big sweeps)

Run from repo root::

  python \"scripts/Running Simulations/diagnose_side_bias_light_vs_worker.py\"

Then pick a row from the CSV and set e.g.::

  SINGLE_GENE_MAP_PATH=maps/BroodWar/(2)Benzene.scxA.xml
"""

from __future__ import annotations

import csv
import os
import sys
import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Matches MicroRTS Front End preset "Nondeterministic-Both" (``UnitTypeTable`` version + move conflict).
FE_UTT_NON_DETERMINISTIC_BOTH: Tuple[int, int] = (3, 1)


def _env_flag_true(name: str, default: str = "1") -> bool:
    v = os.environ.get(name, default).strip().lower()
    return v not in ("0", "false", "no", "off", "")


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sim_dir() -> Path:
    return Path(__file__).resolve().parent


def _discover_broodwar_maps(microrts_root: Path) -> List[str]:
    """Paths relative to ``microrts_root`` (the ``.../microrts`` folder), e.g. ``maps/BroodWar/....xml``."""
    rels: List[str] = []
    brood = microrts_root / "maps" / "BroodWar"
    if not brood.is_dir():
        return rels
    for p in sorted(brood.rglob("*.xml")):
        try:
            rels.append(p.relative_to(microrts_root).as_posix())
        except ValueError:
            continue
    return rels


def _discover_basesworkers_maps(microrts_root: Path, max_maps: int) -> List[str]:
    rels: List[str] = []
    for p in sorted(microrts_root.rglob("*.xml")):
        if "basesWorkers" not in p.name:
            continue
        try:
            rels.append(p.relative_to(microrts_root).as_posix())
        except ValueError:
            continue
        if max_maps and len(rels) >= max_maps:
            break
    return rels


def _resolve_map_paths(
    project_root: Path,
    map_set: str,
    single_path: str,
    max_maps: int,
) -> List[str]:
    microrts = project_root / "gym_microrts" / "microrts"
    ms = map_set.strip().lower()
    if ms == "single":
        return [single_path.strip()]
    if ms == "broodwar":
        rels = _discover_broodwar_maps(microrts)
        if max_maps and len(rels) > max_maps:
            rels = rels[:max_maps]
        return rels
    if ms == "basesworkers":
        cap = max_maps if max_maps else 10_000
        return _discover_basesworkers_maps(microrts, cap)
    raise ValueError(f"Unknown DIAG_MAP_SET={map_set!r}; use broodwar, single, or basesworkers")


def _run_ordering(
    ordering_label: str,
    ai_left_name: str,
    ai_right_name: str,
    games: int,
    map_path: str,
    utt_json_p0: Optional[str],
    utt_json_p1: Optional[str],
    max_steps: int,
    writer: csv.DictWriter | None,
    utt_builtin: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, int]:
    """Run one ordering; optional per-game CSV rows. Returns (left_wins, right_wins, draws)."""
    project_root = _get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    sim = _sim_dir()
    if str(sim) not in sys.path:
        sys.path.insert(0, str(sim))

    from run_match_configured import resolve_winner_side_p0_p1  # noqa: E402

    from gym_microrts.envs.vec_env import MicroRTSBotVecEnv  # type: ignore
    from gym_microrts import microrts_ai  # type: ignore

    ai_left_func = getattr(microrts_ai, ai_left_name)
    ai_right_func = getattr(microrts_ai, ai_right_name)

    print(f"\n=== {ordering_label}: {ai_left_name} (P0/left) vs {ai_right_name} (P1/right) ===")

    jp0 = utt_json_p0 if utt_builtin is None else None
    jp1 = utt_json_p1 if utt_builtin is None else None
    env = MicroRTSBotVecEnv(
        ai1s=[ai_left_func],
        ai2s=[ai_right_func],
        map_paths=[map_path],
        max_steps=max_steps,
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        utt_json_p0=jp0,
        utt_json_p1=jp1,
        utt_builtin=utt_builtin,
        autobuild=False,
    )

    left_wins = 0
    right_wins = 0
    draws = 0

    for g in range(games):
        obs = env.reset()
        steps = 0
        winner_side = "draw"
        winner_ai = "draw"
        raw_reward0 = 0.0

        while True:
            h, w = env.height, env.width
            L = 7 * h * w
            actions = [[[0] * L, [0] * L]]
            obs, rewards, dones, infos = env.step(actions)
            steps += 1

            done_flag = bool(dones[0]) if isinstance(dones, (list, tuple)) and len(dones) else bool(dones)
            if not done_flag:
                continue

            info0 = infos[0] if isinstance(infos, list) and infos else infos
            raw_rewards_full = None
            if isinstance(info0, dict) and "raw_rewards" in info0:
                rr = info0["raw_rewards"]
                rr = rr.tolist() if hasattr(rr, "tolist") else rr
                raw_rewards_full = rr
                if isinstance(rr, (list, tuple)) and rr:
                    raw_reward0 = float(rr[0])
            winner_side = resolve_winner_side_p0_p1(env, info0 if isinstance(info0, dict) else None)
            if winner_side == "left":
                winner_ai = ai_left_name
                left_wins += 1
            elif winner_side == "right":
                winner_ai = ai_right_name
                right_wins += 1
            else:
                winner_ai = "draw"
                draws += 1

            if writer is not None:
                writer.writerow(
                    {
                        "map_path": map_path,
                        "ordering": ordering_label,
                        "game_index": g,
                        "winner_side": winner_side,
                        "winner_ai": winner_ai,
                        "raw_reward0": raw_reward0,
                        "raw_rewards_full": repr(raw_rewards_full),
                        "steps": steps,
                    }
                )
            break

    print(f"{ordering_label} summary over {games} games: left_wins={left_wins}, right_wins={right_wins}, draws={draws}")
    return left_wins, right_wins, draws


def main() -> None:
    project_root = _get_project_root()
    os.chdir(project_root)
    sim = _sim_dir()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(sim) not in sys.path:
        sys.path.insert(0, str(sim))

    from run_match_configured import run_pair  # noqa: E402

    ai1 = os.environ.get("DIAG_AI1", "lightRushAI").strip()
    ai2 = os.environ.get("DIAG_AI2", "workerRushAI").strip()
    max_steps = int(os.environ.get("DIAG_MAX_STEPS", "100000"))
    games = int(os.environ.get("DIAG_GAMES", "1"))
    map_set = os.environ.get("DIAG_MAP_SET", "broodwar").strip()
    single_map = os.environ.get(
        "DIAG_MAP_PATH",
        "maps/10x10/basesWorkers10x10.xml",
    ).strip()
    max_maps = int(os.environ.get("DIAG_MAX_MAPS", "0"))
    skip_errors = os.environ.get("DIAG_SKIP_ERRORS", "1") == "1"
    utt_p0 = os.environ.get("DIAG_UTT_P0", "utts/TestUnitTypeTable.json").strip()
    utt_p1 = os.environ.get("DIAG_UTT_P1", "utts/TestUnitTypeTable.json").strip()
    use_fe_utt = _env_flag_true("DIAG_USE_FE_UTT", "1")
    utt_builtin = FE_UTT_NON_DETERMINISTIC_BOTH if use_fe_utt else None
    utt_p0_run: Optional[str] = None if utt_builtin else utt_p0
    utt_p1_run: Optional[str] = None if utt_builtin else utt_p1
    per_game = os.environ.get("DIAG_PER_GAME_CSV", "0") == "1"

    map_paths = _resolve_map_paths(project_root, map_set, single_map, max_maps)
    _dvg = os.environ.get("DIAG_VERBOSE_GAMES", "").strip().lower()
    if _dvg in ("0", "false", "no", "off"):
        verbose_games = False
    elif _dvg in ("1", "true", "yes", "on"):
        verbose_games = True
    else:
        # Default on so terminal (and tee’d .txt log) show each game’s winner; broodwar sweeps are long—set DIAG_VERBOSE_GAMES=0 to quiet.
        verbose_games = True
    if not map_paths:
        raise SystemExit(
            "No maps found. Check DIAG_MAP_SET / gym_microrts/microrts/maps/BroodWar/"
        )

    out_dir = project_root / "scripts" / "Running Simulations" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "light_vs_worker_map_balance.csv"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_txt = out_dir / f"light_vs_worker_map_balance_{ts}.txt"
    per_game_csv = out_dir / "light_vs_worker_per_game.csv"

    rows: List[dict] = []

    class _Tee:
        def __init__(self, terminal, file_obj):
            self.terminal = terminal
            self.file_obj = file_obj

        def write(self, data):
            self.terminal.write(data)
            self.file_obj.write(data)

        def flush(self):
            self.terminal.flush()
            self.file_obj.flush()

    with out_txt.open("w", encoding="utf-8") as out_f:
        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout, out_f)
        try:
            _utt_line = (
                f"utt_builtin={utt_builtin} (FE Nondeterministic-Both)"
                if utt_builtin
                else f"utt_json P0={utt_p0!r} P1={utt_p1!r}"
            )
            print(
                f"Map set: {map_set}  ({len(map_paths)} maps)  games/order={games}  "
                f"max_steps={max_steps}  ai1={ai1} ai2={ai2}  "
                f"verbose_per_game={verbose_games}  DIAG_USE_FE_UTT={int(use_fe_utt)}  {_utt_line}\n",
                flush=True,
            )

            for i, rel in enumerate(map_paths):
                print(f"\n[{i + 1}/{len(map_paths)}] {rel}", flush=True)
                try:
                    if verbose_games:
                        print(
                            f"  Ordering 1 — P0/left={ai1}  vs  P1/right={ai2}",
                            flush=True,
                        )
                    r1 = run_pair(
                        ai1,
                        ai2,
                        rel,
                        max_steps,
                        games,
                        False,
                        None,
                        utt_p0_run,
                        utt_p1_run,
                        verbose_games=verbose_games,
                        utt_builtin=utt_builtin,
                    )
                    if verbose_games:
                        print(
                            f"  Ordering 2 — P0/left={ai2}  vs  P1/right={ai1}  (swap)",
                            flush=True,
                        )
                    r2 = run_pair(
                        ai2,
                        ai1,
                        rel,
                        max_steps,
                        games,
                        False,
                        None,
                        utt_p0_run,
                        utt_p1_run,
                        verbose_games=verbose_games,
                        utt_builtin=utt_builtin,
                    )
                except Exception as e:
                    if skip_errors:
                        print(f"  SKIP: {e}", flush=True)
                        rows.append(
                            {
                                "map_path": rel,
                                "seat_asymmetry": 999.0,
                                "aggregate_ai1_win_rate": "",
                                "balance_distance": 999.0,
                                "ord1_ai1_rate": "",
                                "ord2_ai1_rate": "",
                                "error": str(e),
                            }
                        )
                        continue
                    raise

                o1_l = int(r1.get("left_wins", 0))
                o1_r = int(r1.get("right_wins", 0))
                o1_d = int(r1.get("draws", 0))
                o2_l = int(r2.get("left_wins", 0))
                o2_r = int(r2.get("right_wins", 0))
                o2_d = int(r2.get("draws", 0))

                d1 = o1_l + o1_r
                d2 = o2_l + o2_r
                rate_o1 = (o1_l / d1) if d1 > 0 else 0.5
                rate_o2 = (o2_r / d2) if d2 > 0 else 0.5
                asym = abs(rate_o1 - rate_o2)
                agg_ai1 = (o1_l + o2_r) / (d1 + d2) if (d1 + d2) > 0 else 0.5
                bal_dist = abs(agg_ai1 - 0.5)

                rows.append(
                    {
                        "map_path": rel,
                        "seat_asymmetry": round(asym, 6),
                        "aggregate_ai1_win_rate": round(agg_ai1, 4),
                        "balance_distance": round(bal_dist, 4),
                        "ord1_ai1_rate": round(rate_o1, 4),
                        "ord2_ai1_rate": round(rate_o2, 4),
                        "ord1_ai1_wins": o1_l,
                        "ord1_ai2_wins": o1_r,
                        "ord1_draws": o1_d,
                        "ord2_ai1_wins": o2_r,
                        "ord2_ai2_wins": o2_l,
                        "ord2_draws": o2_d,
                        "error": "",
                    }
                )
                print(
                    f"  asym={asym:.3f}  agg_{ai1}={agg_ai1:.2f}  "
                    f"o1_rate={rate_o1:.2f}  o2_rate={rate_o2:.2f}",
                    flush=True,
                )

            rows_ok = [r for r in rows if r.get("error") == ""]
            rows_bad = [r for r in rows if r.get("error") != ""]
            rows_ok.sort(
                key=lambda r: (
                    float(r["seat_asymmetry"]),
                    float(r["balance_distance"]),
                )
            )

            fieldnames = [
                "map_path",
                "seat_asymmetry",
                "balance_distance",
                "aggregate_ai1_win_rate",
                "ord1_ai1_rate",
                "ord2_ai1_rate",
                "ord1_ai1_wins",
                "ord1_ai2_wins",
                "ord1_draws",
                "ord2_ai1_wins",
                "ord2_ai2_wins",
                "ord2_draws",
                "error",
            ]
            with summary_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                for r in rows_ok + rows_bad:
                    w.writerow(r)

            print(f"\nWrote {summary_csv}", flush=True)
            if rows_ok:
                best = rows_ok[0]
                print(
                    f"\nBest overall balance (low seat asym, near 50% aggregate): "
                    f"{best['map_path']}\n"
                    f"  seat_asymmetry={best['seat_asymmetry']}  "
                    f"aggregate_ai1_win_rate={best['aggregate_ai1_win_rate']}",
                    flush=True,
                )

            # Optional per-game detail: single map only
            if per_game and map_set == "single" and len(map_paths) == 1:
                with per_game_csv.open("w", newline="", encoding="utf-8") as f:
                    fieldnames = [
                        "map_path",
                        "ordering",
                        "game_index",
                        "winner_side",
                        "winner_ai",
                        "raw_reward0",
                        "raw_rewards_full",
                        "steps",
                    ]
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    mp = map_paths[0]
                    _run_ordering(
                        "ordering_1_ai1_left",
                        ai1,
                        ai2,
                        games,
                        mp,
                        utt_p0_run,
                        utt_p1_run,
                        max_steps,
                        w,
                        utt_builtin=utt_builtin,
                    )
                    _run_ordering(
                        "ordering_2_ai1_left",
                        ai2,
                        ai1,
                        games,
                        mp,
                        utt_p0_run,
                        utt_p1_run,
                        max_steps,
                        w,
                        utt_builtin=utt_builtin,
                    )
                print(f"Per-game CSV: {per_game_csv}", flush=True)
            elif per_game:
                print(
                    "DIAG_PER_GAME_CSV=1 only runs with DIAG_MAP_SET=single and one map.",
                    flush=True,
                )
        finally:
            sys.stdout = old_stdout

    print(f"Text log: {out_txt}")


if __name__ == "__main__":
    main()
