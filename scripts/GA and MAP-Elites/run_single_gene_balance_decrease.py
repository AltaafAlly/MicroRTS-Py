"""
One-gene GA experiment with explicit "decrease when dominating" behavior.

What it does:
- Evolves exactly one gene (defaults: ``SINGLE_GENE_AI2=heavyRushAI``, ``SINGLE_GENE_UNIT=Heavy``, ``SINGLE_GENE_PARAM=hp`` —
  **Heavy.hp** is the best single lever for **lightRush vs heavyRush**: tankier Heavies favor the side that masses them.
  Override with ``SINGLE_GENE_UNIT`` / ``SINGLE_GENE_PARAM`` (e.g. ``Light`` + ``maxDamage``, or ``Worker`` + ``hp``).
- Starts that gene near maximum value (strong Heavies first for the default matchup).
- Evaluates with two AIs and real matches (WorkingGAEvaluator).
- Applies dominance-aware fitness shaping toward ~50% AI1 wins; direction on the genome depends on whether the stat
  mostly buffs **AI1** or **AI2** (see ``SINGLE_GENE_HIGH_RAW_BUFFS_AI1`` below).
- Saves CSV logs and plots, including a graph that shows gene decrease.

**Heavy gene + lightRush (AI1) vs heavyRush (AI2) — default:**
- Higher ``Heavy.hp`` helps **both** players’ Heavies, but heavy rush stacks them more, so it mainly helps **AI2**.
  Auto ``high_raw_buffs_ai1=False`` (same inversion idea as Worker vs worker rush).

**Worker gene + lightRush vs workerRush** (optional): same inversion when unit is ``Worker`` and AI2 name contains ``worker``.

- Override with ``SINGLE_GENE_HIGH_RAW_BUFFS_AI1=1`` or ``0`` if auto-detection does not match your AI names.

**“Light crushes workers first, then GA backs off”** (``SINGLE_GENE_UNIT=Light`` …):
- Use ``SINGLE_GENE_HIGH_RAW_BUFFS_AI1=1`` (default for non-Worker units). When AI1 wins too much, penalize a **high**
  gene so the population weakens Lights. You still need bounds/map where AI1 can actually dominate at the start.
- Set ``SINGLE_GENE_DEC_START_AT_BUFF_EXTREME=1`` (optional ``SINGLE_GENE_DEC_START_NOISE=0``) for a clamped start.

Defaults favor measurable balance signal:
- SINGLE_GENE_USE_BOTH_ORDERINGS=1 (default): run (ai1,ai2) and (ai2,ai1); fitness uses wins by AI identity (not seat),
  with optional side-lock penalty. Set to **0** / **false** / **off** for single seating only.
- SINGLE_GENE_DEC_GAMES: if **unset**, **3** games per ordering on BroodWar maps, **8** on other maps (local default).
  Raise explicitly (e.g. ``12``–``16``) for stabler win-rate estimates on long runs.
  With both orderings on, each eval runs 2× that many games per pair.

**“Draws” in logs:** each match is simulated to completion or until ``max_steps``. A draw here usually means the episode **hit
the step cap** without elimination — the sim ran; the cap is just too short for a decisive outcome on that map.

For time-like stats where lower raw value means a stronger unit (moveTime, …),
fitness inverts a "strength" gauge; for damage/hp-style stats, higher raw = stronger
and starts are not inverted. SINGLE_GENE_DEC_START_NORM = buff strength (0–1).

SINGLE_GENE_SHAPING_WIN_RATE:
  cross_seat_min (default, aka mirror_min): min(order1,order2) — fitness tracks AI1's WORST
    seating. This is the only honest balance signal on a seat-decided rush matchup: a gene that
    wins 100% in one seat and 0% in the other (a "side flip") reads min≈0, so it is NOT mistaken
    for balanced. Combined with SINGLE_GENE_SIDE_LOCK_WEIGHT (which penalizes |seat1-seat2|),
    this drives the gene toward ~50% in BOTH seats instead of stalling on the side-flip plateau.
  aggregate: overall AI1 win rate across both seats. DEGENERATE on seat-decided matchups —
    a 0/1 side flip averages to 0.5 and looks "perfectly balanced," so the gene never converges.
  auto: legacy — max(order1,order2) when seats diverge ≥0.5; stops the decrease early.
  ordering1 / ordering2: explicit single-seat rates.

SINGLE_GENE_AI1 / SINGLE_GENE_AI2 (optional), SINGLE_GENE_MAP_PATH, SINGLE_GENE_DEC_START_NOISE,
SINGLE_GENE_SIDE_LOCK_WEIGHT (default 0.12 so shaped fitness is not clamped to zero).

SINGLE_GENE_RUN_PARENT (optional): if set, new runs are created under this directory
(``<parent>/single_gene_balance_decrease_<timestamp>[_job<SLURM_JOB_ID>]``) instead of
``single_gene_experiment/runs/`` under the repo — useful on clusters (see ``cluster/submit_single_gene_balance_decrease.sbatch``).

Default map is ``maps/16x16/basesWorkers16x16.xml`` (16×16 **basesWorkers**). Set ``SINGLE_GENE_MAP_PATH`` for others
(e.g. ``maps/BroodWar/(4)BloodBath.scmA.xml`` for BroodWar-style **Blood Bath**). Paths are **relative to**
``gym_microrts/microrts/``; absolute paths under that folder are normalized automatically.

**Steps per game:** If ``SINGLE_GENE_MAX_STEPS`` is **unset**, BroodWar paths default to **300000** ticks per game.
Other maps default to **50000** (16×16 local default; raise if logs show mostly timeout-draws).

If logs show all draws and total steps ≈ games×max_steps, you are still **hitting the cap** — raise
``SINGLE_GENE_MAX_STEPS`` further (try 400000–600000) or reduce ``SINGLE_GENE_DEC_GAMES`` while debugging.

**JVM:** JPype allows one JVM per Python process. If you interrupt the run (Ctrl+C), restart the script in a **new**
process — “JVM cannot be restarted” is expected afterward, not a BroodWar bug.

**Local defaults** match validated run ``single_gene_balance_decrease_20260604_122522``:
40 gens, pop 16, 8 games/ordering, 50k max_steps, ``SINGLE_GENE_DEC_JUMP_RATE=0.05`` (lower jump avoids
late-gen reinjection to ~100k HP — runs with jump **0.08** often spike mean gene back up around gen 34).
``SINGLE_GENE_SMOKE=1`` forces a smaller profile via ``cluster/run_single_gene_local_smoke.sh``.
Replay exactly: ``cluster/run_single_gene_local_default.sh``.

Open run_config.txt in the run folder for resolved settings.

Non-evolved genes use a **fixed playable background** (not full-midpoint genomes) so small maps like
``basesWorkers16x16`` can still afford Barracks and combat units; only ``SINGLE_GENE_UNIT`` /
``SINGLE_GENE_PARAM`` is meant to vary. Set ``SINGLE_GENE_USE_MIDPOINT_BACKGROUND=1`` to restore the old
all-midpoint template (can make Barracks unaffordable when midpoint cost exceeds mineable resources).

Each evaluation also copies the UTT JSON to ``<run_dir>/utt_snapshots/gen{N}_ind{M}.json``.
After the run, ``matches.csv`` and ``<run_dir>/match_outputs/gen{N}_ind{M}/<ai>_vs_<ai>.txt`` mirror the local GA test
(compositions and optional step snapshots when ``SINGLE_GENE_SAVE_GAME_DETAILS`` or ``GA_SAVE_GAME_DETAILS`` is on;
default is **off** for faster runs; set ``SINGLE_GENE_SAVE_GAME_DETAILS=1`` to enable).
After the run:

- ``best_balanced_chromosome_utt.json`` — genome whose **shaping win rate** was closest to 50%
  (tie-break: lower ``Heavy.hp`` / denormalized gene). This is the primary “balanced matchup” artifact.
- ``best_chromosome_utt.json`` — highest **shaped fitness** over all generations (legacy / comparison).
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from core.ga_chromosome import MicroRTSChromosome
from core.ga_working_evaluator import WorkingGAEvaluator
from core.single_gene_balance_fitness import shape_one_gene_balancing_fitness

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default map for this script (microrts-relative). Override with SINGLE_GENE_MAP_PATH.
DEFAULT_SINGLE_GENE_MAP_PATH = "maps/16x16/basesWorkers16x16.xml"

# Java UnitTypeTable baselines (gym_microrts/microrts/src/rts/units/UnitTypeTable.java).
# Matches in WorkingGAEvaluator overlay evolved JSON on UnitTypeTable(3, 1) — Nondeterministic + CancelBoth.
_EVAL_BASELINE_UNITTYPE_TABLE: dict[tuple[str, str], int] = {
    ("Heavy", "hp"): 8,
}
# ``new UnitTypeTable()`` default: VERSION_ORIGINAL (1) — true vanilla constructor.
_JAVA_DEFAULT_UNITTYPE_TABLE: dict[tuple[str, str], int] = {
    ("Heavy", "hp"): 4,
}
# Optional JSON override only when SINGLE_GENE_REFERENCE_UTT is set (not the game default).
DEFAULT_REFERENCE_UTT_PATH = (
    PROJECT_ROOT / "gym_microrts" / "microrts" / "utts" / "CustomDemoUTT.json"
)

# Stats that use log-scale genome encoding when the raw range span is large (see resolve_use_log_scale).
_LOG_SCALE_PARAMS = frozenset({"hp", "minDamage", "maxDamage", "cost", "harvestAmount"})


def resolve_high_raw_buffs_ai1(
    *, target_unit: str, target_param: str, ai1: str, ai2: str
) -> tuple[bool, str]:
    """
    If True, ``shape_one_gene_balancing_fitness`` treats a *higher* normalized gene as helping **AI1**.

    When the evolved unit is what **AI2’s strategy stacks** (Workers vs worker rush, Heavies vs heavy rush), higher raw
    mostly helps **AI2**; we auto-return False for those pairings unless ``SINGLE_GENE_HIGH_RAW_BUFFS_AI1`` is set.
    """
    raw = (os.environ.get("SINGLE_GENE_HIGH_RAW_BUFFS_AI1") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True, "env=1"
    if raw in ("0", "false", "no", "off"):
        return False, "env=0"
    u = (target_unit or "").strip().lower()
    p = (target_param or "").strip()
    a1 = (ai1 or "").lower()
    a2 = (ai2 or "").lower()
    if u == "worker" and "light" in a1 and "worker" in a2:
        return False, "auto(lightRush_vs_workerRush+Worker)"
    # Heavier Heavies / harder hits favor heavyRush more than lightRush; do not auto-invert ambiguous econ (e.g. cost).
    if (
        u == "heavy"
        and "light" in a1
        and "heavy" in a2
        and p in ("hp", "minDamage", "maxDamage", "attackTime", "moveTime", "attackRange", "sightRadius")
    ):
        return False, "auto(lightRush_vs_heavyRush+Heavy)"
    # Ranged stat buffs usually help rangedRush more than heavyRush in heavy vs ranged experiments.
    if (
        u == "ranged"
        and "heavy" in a1
        and "ranged" in a2
        and p in ("hp", "minDamage", "maxDamage", "attackTime", "moveTime", "attackRange", "sightRadius")
    ):
        return False, "auto(heavyRush_vs_rangedRush+Ranged)"
    return True, "default"


def normalize_single_gene_map_path(raw: str) -> str:
    """
    MicroRTS JNI loads map XML paths **relative to** ``gym_microrts/microrts/``.
    An absolute path (e.g. pasted from the IDE) breaks Java resource loading with
    ``MalformedURLException: ... spec is null``. If ``raw`` is under that directory,
    return the microrts-relative posix path (``maps/...``).
    """
    s = (raw or "").strip()
    if not s:
        return DEFAULT_SINGLE_GENE_MAP_PATH
    p = Path(s)
    if not p.is_absolute():
        return s.replace("\\", "/")
    microrts = (PROJECT_ROOT / "gym_microrts" / "microrts").resolve()
    try:
        return p.resolve().relative_to(microrts).as_posix()
    except ValueError:
        return s.replace("\\", "/")


def _paths_include_broodwar(map_paths: list[str]) -> bool:
    return any("BroodWar" in (p or "").replace("\\", "/") for p in map_paths)


def resolve_max_steps_for_maps(map_paths: list[str]) -> tuple[int, str]:
    """
    Per-game step cap. BroodWar layouts are large; 100k ticks often ends in timeout-draws with no fitness signal.

    Returns (max_steps, source) where source is 'env', 'broodwar_default', or 'default'.
    """
    raw = (os.environ.get("SINGLE_GENE_MAX_STEPS") or "").strip()
    if raw:
        return int(raw), "env"
    if _paths_include_broodwar(map_paths):
        return 300_000, "broodwar_default"
    return 50_000, "default"


def resolve_games_per_eval_for_maps(map_paths: list[str]) -> tuple[int, str]:
    """
    BroodWar evals are expensive (few games × long horizons). Default to 3 games when unset unless user sets env.
    Other maps default to 20 games per ordering when unset. Higher than before on purpose:
    near balance the win rate is noisy/quantized, so more games are needed to resolve a real
    50/50 from sampling noise (raise to 24-32 for final runs; lower for quick smoke).

    Returns (games, source) where source is 'env', 'broodwar_default', or 'default'.
    """
    raw = (os.environ.get("SINGLE_GENE_DEC_GAMES") or "").strip()
    if raw:
        return max(1, int(raw)), "env"
    if _paths_include_broodwar(map_paths):
        return 3, "broodwar_default"
    return 20, "default"


# MicroRTS normalization: genome value 0 -> min_val, 1 -> max_val. For time-like
# parameters, lower raw = faster/stronger; balance fitness assumes "high gene = buff".
_PARAM_LOWER_RAW_IS_STRONGER = frozenset(
    {"moveTime", "attackTime", "produceTime", "harvestTime", "returnTime"},
)


def normalized_gene_for_balance_strength(param_name: str, genome_value: float) -> float:
    """Map stored genome [0,1] to 'buff strength' in [0,1] for fitness shaping."""
    g = max(0.0, min(1.0, float(genome_value)))
    if param_name in _PARAM_LOWER_RAW_IS_STRONGER:
        return 1.0 - g
    return g


def resolve_shaping_ai1_win_rate(
    *,
    aggregate_rate: float,
    use_both_orderings: bool,
    o1_rate: float,
    o2_rate: float,
    o1_decisive: int,
    o2_decisive: int,
    mode: str,
) -> tuple[float, str]:
    """
    Win rate used for imbalance / shaped fitness.

    When both orderings are on and runs are mirrored (e.g. 0—16 vs 16—0),
    aggregate-by-AI is always 0.5. ``auto`` then uses **max(ordering1, ordering2)**:
    if AI1 crushes in *either* seating, shaping sees high win rate → dominance_penalty
    mode **nerfs** the gene (the intended “start buffed, then decrease” story).
    Using only ordering1 would show 0% when AI1 loses on the left but wins on the
    right — fitness would incorrectly **buff** because it looks like AI1 is behind.
    """
    m = mode.strip().lower()
    if m == "aggregate":
        return aggregate_rate, "aggregate"
    if m == "ordering1":
        return o1_rate, "ordering1"
    if m == "ordering2":
        return o2_rate, "ordering2"
    if m == "mirror_max":
        return max(o1_rate, o2_rate), "mirror_max"
    if m == "mirror_min":
        return min(o1_rate, o2_rate), "mirror_min"
    if m == "cross_seat_min":
        # Same numeric rate as mirror_min; name encodes goal: optimize worst seating.
        return min(o1_rate, o2_rate), "cross_seat_min"
    # auto
    if (
        use_both_orderings
        and o1_decisive > 0
        and o2_decisive > 0
        and abs(o1_rate - o2_rate) >= 0.5
    ):
        return max(o1_rate, o2_rate), "auto_mirror_max"
    return aggregate_rate, "auto_aggregate"


class StreamCapture:
    """Capture stdout/stderr to file, optionally mirroring to console."""

    def __init__(self, path: Path, mirror_to_console: bool = True):
        self.path = path
        self.mirror_to_console = mirror_to_console
        self._old_stdout = None
        self._old_stderr = None
        self._file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", buffering=1)
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        if self._file:
            self._file.close()

    def write(self, data):
        if self._file:
            self._file.write(data)
            if "\n" in data:
                self._file.flush()
        if self.mirror_to_console and self._old_stdout is not None:
            self._old_stdout.write(data)

    def flush(self):
        if self._file:
            self._file.flush()
        if self.mirror_to_console and self._old_stdout is not None:
            self._old_stdout.flush()


@dataclass
class GeneSpec:
    unit_type: str
    param_name: str
    index: int
    min_val: int
    max_val: int


def compute_gene_index(unit_type: str, param_name: str) -> GeneSpec:
    idx = 0
    for ut in MicroRTSChromosome.UNIT_TYPES:
        bounds = MicroRTSChromosome.PARAMETER_BOUNDS[ut]
        for pname, (lo, hi) in bounds.items():
            if ut == unit_type and pname == param_name:
                return GeneSpec(unit_type=ut, param_name=pname, index=idx, min_val=lo, max_val=hi)
            idx += 1
    raise ValueError(f"Gene not found: {unit_type}.{param_name}")


def load_reference_unit_param_from_json(
    unit_type: str,
    param_name: str,
    *,
    utt_path: Path,
) -> int | None:
    """Read a unit stat from a UTT JSON file (only when explicitly requested via env)."""
    if not utt_path.is_file():
        return None
    try:
        data = json.loads(utt_path.read_text(encoding="utf-8"))
        for ut in data.get("unitTypes", []):
            if ut.get("name") == unit_type and param_name in ut:
                return int(ut[param_name])
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def resolve_reference_unit_param(unit_type: str, param_name: str) -> tuple[int | None, str, int | None]:
    """
    Return (eval_baseline, source_label, java_constructor_default).

    Eval baseline matches WorkingGAEvaluator: UnitTypeTable version 3 overlay (not CustomDemoUTT).
    """
    key = (unit_type, param_name)
    utt_json = (os.environ.get("SINGLE_GENE_REFERENCE_UTT") or "").strip()
    if utt_json:
        val = load_reference_unit_param_from_json(
            unit_type, param_name, utt_path=Path(utt_json).expanduser()
        )
        if val is not None:
            return val, f"json:{Path(utt_json).name}", _JAVA_DEFAULT_UNITTYPE_TABLE.get(key)
    if key in _EVAL_BASELINE_UNITTYPE_TABLE:
        return (
            _EVAL_BASELINE_UNITTYPE_TABLE[key],
            "UnitTypeTable(3,1)_eval_baseline",
            _JAVA_DEFAULT_UNITTYPE_TABLE.get(key),
        )
    return None, "unknown", _JAVA_DEFAULT_UNITTYPE_TABLE.get(key)


def resolve_use_log_scale(gene: GeneSpec, param_name: str) -> bool:
    """
    Log-scale encoding maps genome [0,1] across log(raw) so low HP values are reachable.

    Linear [1,100000] makes norm≈0.001 still mean raw≈100+ (your cluster plateau ~145).
    """
    raw = (os.environ.get("SINGLE_GENE_LOG_SCALE") or "auto").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    span = gene.max_val - gene.min_val
    return param_name in _LOG_SCALE_PARAMS and span >= 30


def genome_to_raw_int(gene: GeneSpec, norm: float, *, use_log_scale: bool) -> int:
    """Decode one genome slot to an integer raw stat."""
    n = max(0.0, min(1.0, float(norm)))
    lo, hi = int(gene.min_val), int(gene.max_val)
    if hi <= lo:
        return lo
    if use_log_scale:
        lo_eff = max(1, lo)
        hi_eff = max(lo_eff + 1, hi)
        log_lo = math.log(lo_eff)
        log_hi = math.log(hi_eff)
        raw = int(round(math.exp(log_lo + n * (log_hi - log_lo))))
    else:
        raw = int(round(lo + n * (hi - lo)))
    return max(lo, min(hi, raw))


def raw_to_genome_norm(gene: GeneSpec, raw: int, *, use_log_scale: bool) -> float:
    """Encode integer raw stat into genome slot [0,1]."""
    lo, hi = int(gene.min_val), int(gene.max_val)
    raw = max(lo, min(hi, int(raw)))
    if hi <= lo:
        return 0.0
    if use_log_scale:
        lo_eff = max(1, lo)
        hi_eff = max(lo_eff + 1, hi)
        raw_eff = max(lo_eff, raw)
        log_lo = math.log(lo_eff)
        log_hi = math.log(hi_eff)
        if log_hi <= log_lo:
            return 0.0
        return (math.log(raw_eff) - log_lo) / (log_hi - log_lo)
    return (raw - lo) / (hi - lo)


def denormalize(gene: GeneSpec, norm: float, *, use_log_scale: bool = False) -> float:
    return float(genome_to_raw_int(gene, norm, use_log_scale=use_log_scale))


def _save_chromosome_utt_files(
    evaluator: WorkingGAEvaluator,
    genome: List[float],
    target_gene: GeneSpec,
    *,
    use_log_scale: bool,
    utt_path: Path,
    meta: dict,
    meta_path: Path,
    ai1: str,
    ai2: str,
    map_path: str,
    target_unit: str,
    target_param: str,
) -> None:
    chrom = chromosome_from_genome(genome, target_gene, use_log_scale=use_log_scale)
    tmp_utt = evaluator._create_utt_file(chrom)
    try:
        shutil.copy2(tmp_utt, utt_path)
    finally:
        if tmp_utt.exists():
            tmp_utt.unlink()
    meta_out = {
        **meta,
        "unit": target_unit,
        "param": target_param,
        "ai1": ai1,
        "ai2": ai2,
        "map": map_path,
    }
    meta_path.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")


def should_replace_balanced_candidate(
    *,
    new_distance: float,
    new_denorm: float,
    best_distance: float,
    best_denorm: float,
) -> bool:
    """Closer to 50% win rate wins; ties prefer lower gene (more decrease from high start)."""
    if new_distance < best_distance - 1e-12:
        return True
    if abs(new_distance - best_distance) <= 1e-12 and new_denorm < best_denorm:
        return True
    return False


def chromosome_from_genome(
    genome: List[float],
    target_gene: GeneSpec,
    *,
    use_log_scale: bool,
) -> MicroRTSChromosome:
    """Build chromosome; target gene uses log/linear decode (from_genome alone is always linear)."""
    chrom = MicroRTSChromosome.from_genome(genome)
    raw = genome_to_raw_int(target_gene, genome[target_gene.index], use_log_scale=use_log_scale)
    unit = chrom.unit_params[target_gene.unit_type]
    setattr(unit, target_gene.param_name, raw)
    return chrom


def mutate_single_gene(
    genome: List[float],
    gene: GeneSpec,
    sigma: float,
    jump_rate: float = 0.0,
    *,
    use_log_scale: bool = False,
) -> List[float]:
    out = list(genome)
    if jump_rate > 0.0 and random.random() < jump_rate:
        out[gene.index] = random.random()
        return out
    if use_log_scale:
        lo, hi = int(gene.min_val), int(gene.max_val)
        if hi > lo:
            lo_eff = max(1, lo)
            hi_eff = max(lo_eff + 1, hi)
            log_lo = math.log(lo_eff)
            log_hi = math.log(hi_eff)
            raw = genome_to_raw_int(gene, out[gene.index], use_log_scale=True)
            log_raw = math.log(max(lo_eff, raw))
            log_raw += random.gauss(0.0, sigma * (log_hi - log_lo))
            log_raw = max(log_lo, min(log_hi, log_raw))
            raw = int(round(math.exp(log_raw)))
            out[gene.index] = raw_to_genome_norm(gene, raw, use_log_scale=True)
            return out
    out[gene.index] = max(0.0, min(1.0, out[gene.index] + random.gauss(0.0, sigma)))
    return out


def midpoint_genome() -> List[float]:
    g: List[float] = []
    for ut in MicroRTSChromosome.UNIT_TYPES:
        for _pname, (lo, hi) in MicroRTSChromosome.PARAMETER_BOUNDS[ut].items():
            g.append(0.0 if hi <= lo else 0.5)
    g.append(0.5)
    return g


# Background (non-target) integer stats for single-gene runs. Midpoint_genome() puts every cost near
# the middle of PARAMETER_BOUNDS (e.g. Barracks.cost 35), which can exceed total mineable income on
# ``basesWorkers10x10`` (~5 start + 20 from one corner node). Only the evolvable gene should move;
# freeze the rest to values that still allow Barracks + combat units on small two-node maps.
_SINGLE_GENE_BACKGROUND_INT_OVERRIDES: dict[tuple[str, str], int] = {
    ("Barracks", "cost"): 20,
    ("Barracks", "produceTime"): 10,
    ("Base", "cost"): 50,
    ("Light", "cost"): 9,
    ("Heavy", "cost"): 18,
    ("Ranged", "cost"): 12,
    ("Worker", "cost"): 2,
    ("Worker", "harvestAmount"): 4,
    ("Worker", "harvestTime"): 9,
    ("Worker", "returnTime"): 6,
}


def single_gene_background_reference_genome() -> List[float]:
    """
    Genome slots in the same order as ``MicroRTSChromosome.from_genome`` / ``to_genome``:
    one float per (unit, param), then global moveConflictResolutionStrategy (0.5 → CRS 2).
    """
    g: List[float] = []
    for ut in MicroRTSChromosome.UNIT_TYPES:
        for pname, (lo, hi) in MicroRTSChromosome.PARAMETER_BOUNDS[ut].items():
            if lo == hi == 0:
                g.append(0.0)
            elif lo == hi:
                g.append(0.5)
            else:
                raw = _SINGLE_GENE_BACKGROUND_INT_OVERRIDES.get((ut, pname))
                if raw is None:
                    raw = (lo + hi) // 2
                raw = int(max(lo, min(hi, raw)))
                g.append((raw - lo) / (hi - lo) if hi > lo else 0.5)
    g.append(0.5)
    return g


def make_run_dir() -> Path:
    raw_parent = (os.environ.get("SINGLE_GENE_RUN_PARENT") or "").strip()
    if raw_parent:
        base = Path(raw_parent).expanduser().resolve()
    else:
        base = PROJECT_ROOT / "scripts" / "GA and MAP-Elites" / "single_gene_experiment" / "runs"
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job = (os.environ.get("SLURM_JOB_ID") or "").strip()
    name = f"single_gene_balance_decrease_{ts}"
    if job:
        name += f"_job{job}"
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _resolve_single_gene_save_game_details() -> bool:
    """Default off for single-gene runs; mirror GA env when set."""
    if "SINGLE_GENE_SAVE_GAME_DETAILS" in os.environ:
        return os.environ["SINGLE_GENE_SAVE_GAME_DETAILS"].strip().lower() in ("1", "true", "yes", "on")
    if "GA_SAVE_GAME_DETAILS" in os.environ:
        return os.environ["GA_SAVE_GAME_DETAILS"].strip().lower() in ("1", "true", "yes", "on")
    return False


def _write_match_log_artifacts(run_dir: Path, match_log: List[dict], save_game_details: bool) -> None:
    """Write ``matches.csv`` and per-matchup ``match_outputs/`` .txt (same layout as ``run_ga_local_test``)."""
    matches_path = run_dir / "matches.csv"
    match_outputs_dir = run_dir / "match_outputs"
    with matches_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "generation",
                "individual_index",
                "ai_left",
                "ai_right",
                "left_wins",
                "right_wins",
                "draws",
                "winner",
                "left_unit_composition",
                "right_unit_composition",
            ]
        )
        for m in match_log:
            w.writerow(
                [
                    m.get("generation", ""),
                    m.get("individual_index", ""),
                    m.get("ai_left", ""),
                    m.get("ai_right", ""),
                    m.get("left_wins", ""),
                    m.get("right_wins", ""),
                    m.get("draws", ""),
                    m.get("winner", ""),
                    m.get("left_unit_composition", "N/A"),
                    m.get("right_unit_composition", "N/A"),
                ]
            )
            if not save_game_details:
                continue
            gen = m.get("generation", 0)
            ind = m.get("individual_index", 0)
            ai_left = str(m.get("ai_left", "")).replace(" ", "_")
            ai_right = str(m.get("ai_right", "")).replace(" ", "_")
            subfolder = match_outputs_dir / f"gen{gen}_ind{ind}"
            subfolder.mkdir(parents=True, exist_ok=True)
            matchup_name = f"{ai_left}_vs_{ai_right}".replace("/", "_")
            txt_path = subfolder / f"{matchup_name}.txt"
            left_comp = m.get("left_unit_composition", "N/A")
            right_comp = m.get("right_unit_composition", "N/A")
            winner = m.get("winner", "")
            with txt_path.open("w", encoding="utf-8") as tf:
                tf.write(f"Match: {m.get('ai_left', '')} (left) vs {m.get('ai_right', '')} (right)\n")
                tf.write(f"Result: {m.get('left_wins', 0)}-{m.get('right_wins', 0)} (draws: {m.get('draws', 0)})\n")
                tf.write(f"Winner: {winner}\n")
                tf.write(f"Left unit composition (end of last game): {left_comp or '(none)'}\n")
                tf.write(f"Right unit composition (end of last game): {right_comp or '(none)'}\n")
                tf.write("\n--- Win condition (MicroRTS) ---\n")
                tf.write(
                    "A game ends when one player has no units left (elimination). That player loses; the other wins.\n"
                )
                tf.write("Draw if both still have units when max steps is reached, or both have zero units.\n")
                tf.write("\n--- Why this result ---\n")
                if left_comp not in ("N/A", "", None) or right_comp not in ("N/A", "", None):
                    if winner == "left":
                        tf.write(f"Left ({m.get('ai_left', '')}) won: Right had no units left (eliminated). ")
                        tf.write(f"End state: Left had {left_comp or 'units'}; Right had {right_comp or 'none'}.\n")
                    elif winner == "right":
                        tf.write(f"Right ({m.get('ai_right', '')}) won: Left had no units left (eliminated). ")
                        tf.write(f"End state: Left had {left_comp or 'none'}; Right had {right_comp or 'units'}.\n")
                    elif winner == "draw":
                        tf.write("Draw: both sides still had units (or tied at zero). ")
                        tf.write(f"End state: Left had {left_comp or 'none'}; Right had {right_comp or 'none'}.\n")
                else:
                    tf.write("(Unit composition not captured; see snapshots below for end state.)\n")
                tf.write(
                    "\n(Unit composition is captured from the last game of the matchup when capture_composition is enabled.)\n"
                )
                all_game_snapshots = m.get("_game_snapshots") or []
                per_game_compositions = m.get("_per_game_compositions") or []
                games_per_ordering = m.get("_games_per_ordering")

                def _comp_str(comp_dict):
                    if not comp_dict or not isinstance(comp_dict, dict):
                        return "none"
                    return ",".join(f"{k}:{v}" for k, v in sorted(comp_dict.items()))

                if all_game_snapshots:
                    tf.write("\n" + "=" * 60 + "\n")
                    tf.write("Game state snapshots – every game, step 0 then every N steps then final\n")
                    tf.write("=" * 60 + "\n")
                    if isinstance(all_game_snapshots[0], list):
                        for game_idx, snapshots in enumerate(all_game_snapshots, 1):
                            if games_per_ordering and game_idx == 1:
                                tf.write(
                                    f"\n(Ordering 1: Left={m.get('ai_left', '')}, Right={m.get('ai_right', '')} "
                                    f"— Games 1–{games_per_ordering})\n"
                                )
                            elif games_per_ordering and game_idx == games_per_ordering + 1:
                                tf.write(
                                    f"\n(Ordering 2: Left={m.get('ai_right', '')}, Right={m.get('ai_left', '')} "
                                    f"— Games {games_per_ordering + 1}–{len(all_game_snapshots)})\n"
                                )
                            tf.write(f"\n--- Game {game_idx} ---\n")
                            if game_idx <= len(per_game_compositions):
                                pg = per_game_compositions[game_idx - 1]
                                pg_winner = pg.get("winner", "")
                                left_c = _comp_str(pg.get("left"))
                                right_c = _comp_str(pg.get("right"))
                                if pg_winner == "left":
                                    tf.write(
                                        f"Winner: Left ({m.get('ai_left', '')}) — Right had no units (elimination). "
                                        f"Left: {left_c}; Right: {right_c}.\n"
                                    )
                                elif pg_winner == "right":
                                    tf.write(
                                        f"Winner: Right ({m.get('ai_right', '')}) — Left had no units (elimination). "
                                        f"Left: {left_c}; Right: {right_c}.\n"
                                    )
                                else:
                                    tf.write(
                                        f"Winner: Draw — Both sides still had units. Left: {left_c}; Right: {right_c}.\n"
                                    )
                            for step, text in snapshots:
                                tf.write(f"\n  Step {step}\n")
                                tf.write(text)
                                if not text.endswith("\n"):
                                    tf.write("\n")
                    else:
                        tf.write("\n--- Game 1 (legacy single-game capture) ---\n")
                        if per_game_compositions:
                            pg = per_game_compositions[0]
                            pg_winner = pg.get("winner", "")
                            left_c = _comp_str(pg.get("left"))
                            right_c = _comp_str(pg.get("right"))
                            if pg_winner == "left":
                                tf.write(
                                    f"Winner: Left — Right had no units (elimination). Left: {left_c}; Right: {right_c}.\n"
                                )
                            elif pg_winner == "right":
                                tf.write(
                                    f"Winner: Right — Left had no units (elimination). Left: {left_c}; Right: {right_c}.\n"
                                )
                            else:
                                tf.write(
                                    f"Winner: Draw — Both sides still had units. Left: {left_c}; Right: {right_c}.\n"
                                )
                        for step, text in all_game_snapshots:
                            tf.write(f"\n  Step {step}\n")
                            tf.write(text)
                            if not text.endswith("\n"):
                                tf.write("\n")
                else:
                    tf.write(
                        "\n(No snapshots: enable capture_snapshots and ensure game_state_utils is available.)\n"
                    )


def build_evaluator(
    *,
    ai1: str,
    ai2: str,
    map_path: str,
    games_per_eval: int,
    use_both_orderings: bool,
    max_steps: int = 100000,
    duration_scoring: str = "longer_better",
    duration_longer_softness_scale: float = 3000.0,
) -> WorkingGAEvaluator:
    # Two-AI experiment: no strategy-diversity term (supervisor: diversity not meaningful with only two agents).
    # Duration: longer games score higher (no fixed target/tolerance band); see WorkingGAEvaluator.duration_scoring.
    return WorkingGAEvaluator(
        alpha=0.8,
        beta=0.2,
        gamma=0.0,
        max_steps=max_steps,
        map_path=map_path,
        map_paths=[map_path],
        games_per_eval=games_per_eval,
        ai_agents=[ai1, ai2],
        use_nondeterministic=True,
        use_both_orderings=use_both_orderings,
        target_duration=500,
        duration_tolerance=400,
        duration_scoring=duration_scoring,
        duration_longer_softness_scale=duration_longer_softness_scale,
    )


def apply_smoke_run_defaults() -> bool:
    """
    Fast local profile: same algorithm, fewer gens/pop/games/steps.

    Enable with SINGLE_GENE_SMOKE=1 or run cluster/run_single_gene_local_smoke.sh.
    Only sets env vars that are not already set.
    """
    if os.environ.get("SINGLE_GENE_SMOKE", "0").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    defaults = {
        "SINGLE_GENE_DEC_GENS": "6",
        "SINGLE_GENE_DEC_POP": "8",
        "SINGLE_GENE_DEC_GAMES": "4",
        "SINGLE_GENE_MAX_STEPS": "20000",
        "SINGLE_GENE_DEC_MUTATION_SIGMA": "0.25",
        "SINGLE_GENE_DEC_JUMP_RATE": "0.05",
        "SINGLE_GENE_LOG_SCALE": "1",
        "SINGLE_GENE_PARAM_MAX_WIDE": "2000",
        "SINGLE_GENE_SAVE_GAME_DETAILS": "0",
        "SINGLE_GENE_QUIET_TERMINAL": "0",
    }
    for key, val in defaults.items():
        os.environ.setdefault(key, val)
    return True


def main() -> None:
    os.chdir(PROJECT_ROOT)
    random.seed(42)
    smoke_run = apply_smoke_run_defaults()

    # --- Main configuration ---
    ai1 = os.environ.get("SINGLE_GENE_AI1", "lightRushAI").strip()
    ai2 = os.environ.get("SINGLE_GENE_AI2", "heavyRushAI").strip()
    map_paths = [
        normalize_single_gene_map_path(
            os.environ.get("SINGLE_GENE_MAP_PATH", DEFAULT_SINGLE_GENE_MAP_PATH),
        ),
    ]
    games_per_eval, games_per_eval_src = resolve_games_per_eval_for_maps(map_paths)
    generations = int(os.environ.get("SINGLE_GENE_DEC_GENS", "40"))
    population_size = int(os.environ.get("SINGLE_GENE_DEC_POP", "16"))
    _both_ord = os.environ.get("SINGLE_GENE_USE_BOTH_ORDERINGS", "1").strip().lower()
    use_both_orderings = _both_ord not in ("0", "false", "no", "off")
    mutation_sigma = float(os.environ.get("SINGLE_GENE_DEC_MUTATION_SIGMA", "0.08"))
    mutation_jump_rate = float(os.environ.get("SINGLE_GENE_DEC_JUMP_RATE", "0.05"))
    crossover_rate = float(os.environ.get("SINGLE_GENE_DEC_CROSSOVER", "0.5"))
    # Convergence controls (so the gene settles instead of wandering the noisy balance plateau):
    #   elites      — top-N genomes copied unchanged into the next generation (best balance never lost).
    #   sigma floor — mutation sigma decays linearly from full (gen 0) to floor*full (final gen).
    #   jump-off    — jump-mutation rate anneals to 0 by this fraction of the run (early explore, late settle).
    elite_count = max(0, int(os.environ.get("SINGLE_GENE_ELITES", "2")))
    elite_count = min(elite_count, population_size)
    sigma_decay_floor = float(os.environ.get("SINGLE_GENE_SIGMA_DECAY_FLOOR", "0.3"))
    jump_off_frac = float(os.environ.get("SINGLE_GENE_JUMP_OFF_FRAC", "0.5"))
    # Large default was clamping shaped fitness to ~0 after subtracting from small bases.
    # Bumped from 0.12 -> 0.20 so per-seat divergence (side flips) is more strongly penalized,
    # reinforcing the cross_seat_min objective (balanced in BOTH seats, not just on average).
    side_lock_weight = float(os.environ.get("SINGLE_GENE_SIDE_LOCK_WEIGHT", "0.20"))
    target_gene_weight = float(os.environ.get("SINGLE_GENE_TARGET_GENE_WEIGHT", "0.30"))
    experiment_mode = os.environ.get(
        "SINGLE_GENE_EXPERIMENT_MODE",
        "dominance_penalty_mode",
    ).strip()
    if experiment_mode not in {"dominance_penalty_mode", "causal_balance_mode"}:
        raise ValueError(
            "SINGLE_GENE_EXPERIMENT_MODE must be one of: "
            "'dominance_penalty_mode', 'causal_balance_mode'"
        )
    dominance_imbalance_threshold = float(
        os.environ.get("SINGLE_GENE_DOMINANCE_IMBALANCE_THRESHOLD", "0.08")
    )
    shaping_win_rate_mode = os.environ.get("SINGLE_GENE_SHAPING_WIN_RATE", "cross_seat_min").strip()

    target_unit = os.environ.get("SINGLE_GENE_UNIT", "Heavy")
    target_param = os.environ.get("SINGLE_GENE_PARAM", "hp")
    target_gene = compute_gene_index(target_unit, target_param)
    high_raw_buffs_ai1, high_raw_buffs_ai1_src = resolve_high_raw_buffs_ai1(
        target_unit=target_unit, target_param=target_param, ai1=ai1, ai2=ai2
    )
    start_strength = float(
        os.environ.get(
            "SINGLE_GENE_DEC_START_STRENGTH",
            os.environ.get("SINGLE_GENE_DEC_START_NORM", "0.999"),
        ),
    )
    start_noise = float(os.environ.get("SINGLE_GENE_DEC_START_NOISE", "0.004"))
    start_at_buff_extreme = os.environ.get("SINGLE_GENE_DEC_START_AT_BUFF_EXTREME", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    max_steps, max_steps_src = resolve_max_steps_for_maps(map_paths)
    duration_scoring = os.environ.get("SINGLE_GENE_DURATION_SCORING", "longer_better").strip()
    duration_longer_softness_scale = float(
        os.environ.get("SINGLE_GENE_DURATION_SOFTNESS_SCALE", "3000")
    )

    # Wide bounds: start near max (dominance), evolve down when matches are imbalanced.
    # Reference UTT is logged for context only — it is NOT a fitness target.
    # Override via SINGLE_GENE_PARAM_MIN / SINGLE_GENE_PARAM_MAX / SINGLE_GENE_PARAM_MAX_WIDE.
    # Disable widening with SINGLE_GENE_USE_WIDE_PARAM_BOUNDS=0 (uses ga_chromosome bounds only).
    use_wide_param_bounds = os.environ.get("SINGLE_GENE_USE_WIDE_PARAM_BOUNDS", "1") == "1"
    pmin_e = os.environ.get("SINGLE_GENE_PARAM_MIN")
    pmax_e = os.environ.get("SINGLE_GENE_PARAM_MAX")
    reference_raw, reference_src, java_default_raw = resolve_reference_unit_param(
        target_unit, target_param
    )
    if pmin_e is None and pmax_e is None and use_wide_param_bounds:
        if target_param in ("moveTime", "attackTime", "produceTime"):
            pmin_e, pmax_e = "1", "30"
        else:
            pmin_e = "1"
            wide_max_env = (os.environ.get("SINGLE_GENE_PARAM_MAX_WIDE") or "").strip()
            # High-but-bounded ceiling: still starts overpowered (heavy dominates well below this on 16x16),
            # but small enough that jump mutations / log-scale don't keep re-injecting absurd values and
            # prevent convergence. Raise via SINGLE_GENE_PARAM_MAX_WIDE if balance sits near the ceiling.
            pmax_e = wide_max_env if wide_max_env else "2000"
    if pmin_e is not None or pmax_e is not None:
        lo, hi = MicroRTSChromosome.PARAMETER_BOUNDS[target_unit][target_param]
        new_lo = int(pmin_e) if pmin_e is not None else lo
        new_hi = int(pmax_e) if pmax_e is not None else hi
        if new_lo < new_hi:
            MicroRTSChromosome.PARAMETER_BOUNDS[target_unit][target_param] = (new_lo, new_hi)
            target_gene = compute_gene_index(target_unit, target_param)

    use_log_scale = resolve_use_log_scale(target_gene, target_param)

    run_dir = make_run_dir()
    save_game_details = _resolve_single_gene_save_game_details()
    validate_utt = os.environ.get("SINGLE_GENE_VALIDATE_UTT", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    use_midpoint_background = os.environ.get("SINGLE_GENE_USE_MIDPOINT_BACKGROUND", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # Vanilla-patch mode (default ON): evaluate every individual on the vanilla Java UnitTypeTable with
    # ONLY the target gene overlaid, instead of the full evolved "buffed background". Diagnostics show the
    # buffed background is what made lightRush-vs-heavyRush seat-decided (unbalanceable by one gene); on a
    # vanilla background both seatings agree and Heavy.hp cleanly balances at ~10. See diagnose_seat_advantage.py.
    use_vanilla_patch = os.environ.get("SINGLE_GENE_VANILLA_PATCH", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    per_individual_csv = run_dir / "gene_trajectory.csv"
    per_generation_csv = run_dir / "generations_summary.csv"
    decrease_plot = run_dir / "gene_decrease_trajectory.png"
    outcome_plot = run_dir / "outcome_trajectory.png"
    fitness_plot = run_dir / "fitness_trajectory.png"
    log_path = run_dir / "run.log"
    terminal_output_path = run_dir / "terminal_output.log"
    quiet_terminal = os.environ.get("SINGLE_GENE_QUIET_TERMINAL", "1") == "1"
    stream_capture = StreamCapture(
        terminal_output_path,
        mirror_to_console=not quiet_terminal,
    )
    stream_capture.__enter__()

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    _cfg_lines = [
        "Single-gene balance-decrease — resolved settings",
        f"smoke_run={smoke_run}",
        f"created_utc={datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"run_dir={run_dir}",
        f"use_both_orderings={use_both_orderings}",
        f"games_per_eval={games_per_eval}",
        f"games_per_eval_source={games_per_eval_src}",
        f"total_games_per_chromosome_when_both_orderings={2 * games_per_eval if use_both_orderings else games_per_eval}",
        f"generations={generations} population_size={population_size}",
        f"SINGLE_GENE_DEC_GAMES env={os.environ.get('SINGLE_GENE_DEC_GAMES', '(unset)')}",
        f"SINGLE_GENE_USE_BOTH_ORDERINGS env={os.environ.get('SINGLE_GENE_USE_BOTH_ORDERINGS', '(unset, default on)')}",
        f"SINGLE_GENE_DEC_JUMP_RATE={mutation_jump_rate}",
        f"SINGLE_GENE_DEC_MUTATION_SIGMA={mutation_sigma}",
        f"SINGLE_GENE_ELITES={elite_count}",
        f"SINGLE_GENE_SIGMA_DECAY_FLOOR={sigma_decay_floor}",
        f"SINGLE_GENE_JUMP_OFF_FRAC={jump_off_frac}",
        f"SINGLE_GENE_LOG_SCALE env={os.environ.get('SINGLE_GENE_LOG_SCALE', '(unset, auto)')}",
        f"SINGLE_GENE_PARAM_MAX_WIDE env={os.environ.get('SINGLE_GENE_PARAM_MAX_WIDE', '(unset, 2000)')}",
        f"validated_local_profile=20260604_122522",
        f"SINGLE_GENE_SHAPING_WIN_RATE={shaping_win_rate_mode}",
        f"ai1={ai1} ai2={ai2}",
        f"map={map_paths[0]}",
        f"max_steps={max_steps}",
        f"max_steps_source={max_steps_src}",
        f"SINGLE_GENE_SIDE_LOCK_WEIGHT={side_lock_weight}",
        f"SINGLE_GENE_DEC_START_NOISE={start_noise}",
        f"SINGLE_GENE_DEC_START_AT_BUFF_EXTREME env={os.environ.get('SINGLE_GENE_DEC_START_AT_BUFF_EXTREME', '(unset)')}",
        f"SINGLE_GENE_HIGH_RAW_BUFFS_AI1={high_raw_buffs_ai1} ({high_raw_buffs_ai1_src})",
        f"target_gene_bounds=({target_gene.min_val},{target_gene.max_val})",
        f"reference_{target_unit}_{target_param}={reference_raw}",
        f"reference_source={reference_src}",
        f"java_default_UnitTypeTable_ORIGINAL_{target_unit}_{target_param}={java_default_raw}",
        f"use_log_scale={use_log_scale}",
        f"SINGLE_GENE_BALANCE_TARGET_RATE={os.environ.get('SINGLE_GENE_BALANCE_TARGET_RATE', '0.5')}",
        f"utt_snapshots_dir={run_dir / 'utt_snapshots'}",
        f"save_game_details={save_game_details}",
        f"match_outputs_dir={run_dir / 'match_outputs'}",
        f"use_midpoint_background={use_midpoint_background}",
        f"validate_utt={validate_utt}",
        f"SINGLE_GENE_VANILLA_PATCH={use_vanilla_patch} (only {target_unit}.{target_param} overlaid on vanilla UnitTypeTable; background frozen at MicroRTS defaults)",
    ]
    (run_dir / "run_config.txt").write_text("\n".join(_cfg_lines) + "\n", encoding="utf-8")

    try:
        log("=" * 60)
        log("Single-gene balance-decrease")
        log(f"Full resolved settings: {run_dir / 'run_config.txt'}")
        log(f"Run dir: {run_dir}")
        log(f"Both orderings: {use_both_orderings}")
        log(
            f"Games per eval: {games_per_eval} per seating "
            f"(source: {games_per_eval_src}; set SINGLE_GENE_DEC_GAMES to override)"
        )
        if use_both_orderings:
            log(
                f"With both orderings: {games_per_eval} games × 2 orderings = "
                f"{2 * games_per_eval} total games per chromosome eval (same map)."
            )
        log("=" * 60)

        # causal_balance_mode used to use pre-run probe direction; without probe, treat as unknown (0).
        probe_effect_direction = 0.0

        evaluator = build_evaluator(
            ai1=ai1,
            ai2=ai2,
            map_path=map_paths[0],
            games_per_eval=games_per_eval,
            use_both_orderings=use_both_orderings,
            max_steps=max_steps,
            duration_scoring=duration_scoring,
            duration_longer_softness_scale=duration_longer_softness_scale,
        )
        utt_snapshots_dir = run_dir / "utt_snapshots"
        utt_snapshots_dir.mkdir(parents=True, exist_ok=True)
        evaluator.utt_log_dir = str(utt_snapshots_dir)
        setattr(evaluator, "validate_utt", validate_utt)
        setattr(
            evaluator,
            "utt_patch_only_fields",
            [(target_gene.unit_type, target_gene.param_name)] if use_vanilla_patch else None,
        )
        setattr(evaluator, "run_match_capture_composition", save_game_details)
        setattr(evaluator, "run_match_capture_snapshots", save_game_details)
        setattr(evaluator, "run_match_snapshot_interval", 15)
        if hasattr(evaluator, "close_cached_env"):
            evaluator.close_cached_env()

        # Initialize population: shared frozen background + only target gene varied.
        base = midpoint_genome() if use_midpoint_background else single_gene_background_reference_genome()
        population: List[List[float]] = []
        for _ in range(population_size):
            g = list(base)
            if start_at_buff_extreme:
                # Strongest end of the genome interval: high raw for damage/hp-like, low raw for time-like.
                if target_param in _PARAM_LOWER_RAW_IS_STRONGER:
                    g[target_gene.index] = max(
                        0.0,
                        min(1.0, 0.0 + random.uniform(0.0, start_noise)),
                    )
                else:
                    g[target_gene.index] = max(
                        0.0,
                        min(1.0, 1.0 + random.uniform(-start_noise, 0.0)),
                    )
            elif target_param in _PARAM_LOWER_RAW_IS_STRONGER:
                g[target_gene.index] = max(
                    0.0,
                    min(1.0, (1.0 - start_strength) + random.uniform(-start_noise, 0.0)),
                )
            else:
                g[target_gene.index] = max(
                    0.0,
                    min(1.0, start_strength + random.uniform(-start_noise, 0.0)),
                )
            population.append(g)

        log("=" * 60)
        log("Evolution setup")
        log(f"Run dir: {run_dir}")
        log(f"AIs: {ai1} vs {ai2}")
        log(f"Map(s): {map_paths}")
        log(f"max_steps per game: {max_steps} (source: {max_steps_src}; set SINGLE_GENE_MAX_STEPS to override)")
        _per_eval = games_per_eval * max_steps * (2 if use_both_orderings else 1)
        log(
            f"Approx upper-bound sim ticks per chromosome eval: {_per_eval} "
            f"(games×max_steps×orderings). BroodWar default: 3 games/ordering unless SINGLE_GENE_DEC_GAMES is set; other maps: 8."
        )
        log(f"Use both orderings: {use_both_orderings}")
        log(
            f"Target gene bounds: {target_unit}.{target_param} raw in "
            f"[{target_gene.min_val}, {target_gene.max_val}]"
        )
        log(
            f"Reference {target_unit}.{target_param} (context only, not a fitness target): "
            f"eval_baseline={reference_raw} ({reference_src}); "
            f"Java new UnitTypeTable() default={java_default_raw}"
        )
        log(
            "Experiment story: start gene near max (heavy dominates) -> shaped fitness pushes "
            "gene down when imbalanced -> stop near balance, not at reference."
        )
        log(
            f"Genome encoding: {'log-scale' if use_log_scale else 'linear'} "
            f"(SINGLE_GENE_LOG_SCALE=auto|0|1)"
        )
        if use_log_scale:
            for _n in (0.0, 0.25, 0.5, 0.75, 1.0):
                _r = int(genome_to_raw_int(target_gene, _n, use_log_scale=True))
                log(f"  log-scale norm {_n:.2f} -> raw {_r}")
        log(f"Terminal output log: {terminal_output_path}")
        log(f"UTT snapshots (one JSON per chromosome eval): {utt_snapshots_dir}")
        log(
            f"Match log: {run_dir / 'matches.csv'}; per-matchup text under {run_dir / 'match_outputs'} "
            f"(save_game_details={save_game_details}; set SINGLE_GENE_SAVE_GAME_DETAILS=0 to skip snapshots/extra .txt)"
        )
        log(
            "Genome background: "
            + (
                "all-midpoint (SINGLE_GENE_USE_MIDPOINT_BACKGROUND=1)"
                if use_midpoint_background
                else "playable fixed non-target loci (default; see _SINGLE_GENE_BACKGROUND_INT_OVERRIDES)"
            )
        )
        log(f"Experiment mode: {experiment_mode}")
        if experiment_mode == "causal_balance_mode":
            log(
                "NOTE: causal_balance_mode previously used probe direction; with no probe, "
                "probe_effect_direction=0 so target-gene pressure stays neutral (0.5). Prefer dominance_penalty_mode."
            )
        log(f"Target gene weight: {target_gene_weight:.3f}")
        log(f"Dominance imbalance threshold: {dominance_imbalance_threshold:.3f}")
        log(
            f"Shaping win rate mode: {shaping_win_rate_mode} "
            f"(cross_seat_min=AI1 worst-seat rate [default, genuine per-seat balance, immune to side flips]; "
            f"aggregate=overall rate which is fooled by 0/1 side flips; auto=legacy max() that stalls early)"
        )
        log(f"Gene: {target_gene.unit_type}.{target_gene.param_name} in [{target_gene.min_val}, {target_gene.max_val}]")
        if target_unit == "Light" and target_param == "maxDamage":
            log(
                "Light.maxDamage: higher integer ⇒ more damage per hit (with nondeterministic UTT, pair with minDamage). "
                "If workers still win at the ceiling, this knob may be saturated — try Light.hp or Light.minDamage."
            )
        elif target_unit == "Light" and target_param == "minDamage":
            log(
                "Light.minDamage: raises damage floor vs workers (VERSION_NON_DETERMINISTIC rolls in [min,max]). "
                "Often a clearer lever than maxDamage alone when max is already huge."
            )
        elif target_unit == "Light" and target_param == "hp":
            log(
                "Light.hp: higher integer ⇒ Lights survive longer in worker swarms — strong default for lightRush vs workerRush. "
                "Fitness still assumes higher raw = stronger Light (same dominance → decrease story)."
            )
        elif target_unit == "Light" and target_param == "moveTime":
            log(
                "Light.moveTime: lower integer ⇒ faster movement per engine tick. "
                "Symmetric UTT: both players’ Lights use the same value; lightRushAI is more sensitive."
            )
        elif target_unit == "Worker" and target_param == "hp":
            log(
                "Worker.hp: higher integer ⇒ tankier workers on **both** sides; worker rush gains more from it. "
                "Default lightRush vs workerRush uses inverted dominance mapping (see high_raw_buffs_ai1 log line)."
            )
        elif target_unit == "Worker" and target_param in ("minDamage", "maxDamage"):
            log(
                "Worker melee damage: higher ⇒ better worker-vs-worker/light brawls; worker-heavy AI usually gains more. "
                "Same inverted mapping as Worker.hp when AI1 is lightRush and AI2 is workerRush."
            )
        elif target_unit == "Heavy" and target_param == "hp":
            log(
                "Heavy.hp: higher integer ⇒ tankier Heavies on **both** sides; heavyRushAI stacks Heavies — main lever "
                "for lightRush vs heavyRush. Auto high_raw_buffs_ai1=False (nerf Heavies when light is losing)."
            )
        elif target_unit == "Heavy" and target_param in ("minDamage", "maxDamage"):
            log(
                "Heavy damage: higher ⇒ Heavies kill Lights faster; heavy-heavy brawls too. "
                "Usually use inverted mapping vs lightRush when AI2 is heavyRush (same as Heavy.hp)."
            )
        log(
            f"Dominance→gene mapping: high_raw_buffs_ai1={high_raw_buffs_ai1} ({high_raw_buffs_ai1_src}). "
            "If False: AI1 crushing ⇒ raise opponent-stack unit stats; AI1 losing ⇒ lower them (e.g. Heavy/Worker hp)."
        )
        if target_param in _PARAM_LOWER_RAW_IS_STRONGER:
            log(
                "Inverted strength gauge for fitness: lower raw value = stronger unit; "
                "SINGLE_GENE_DEC_START_NORM is start strength (not raw genome value)."
            )
            log(
                "Plot/csv denorm: for time-like stats, INCREASING raw ticks = slower = nerf "
                "(expected trajectory when AI1 is seen as dominating and balance pushes down)."
            )
        log(
            f"Duration scoring: {duration_scoring}  "
            f"(softness_scale τ={duration_longer_softness_scale} in 1-exp(-steps/τ); max_steps={max_steps})"
        )
        log(f"Fitness mix: alpha=0.8 balance, beta=0.2 duration, gamma=0 (no strategy diversity)")
        log(
            f"Start strength: {start_strength:.3f}  "
            f"(SINGLE_GENE_DEC_START_NORM / SINGLE_GENE_DEC_START_STRENGTH; "
            f"time-like params use inverted genome init; damage-like use genome≈strength)  "
            f"SINGLE_GENE_DEC_START_AT_BUFF_EXTREME={'on' if start_at_buff_extreme else 'off'}  "
            f"generations={generations} population={population_size}"
        )
        log("=" * 60)
        log("")
        log("#" * 80)
        log("Evolution — evaluate_chromosome output follows (both orderings unless SINGLE_GENE_USE_BOTH_ORDERINGS=0).")
        log("#" * 80)
        log("")

        gen_mean_gene: List[float] = []
        gen_std_gene: List[float] = []
        gen_best_gene: List[float] = []
        gen_mean_ai1_win: List[float] = []
        gen_best_ai1_win: List[float] = []
        gen_mean_shaped: List[float] = []
        gen_best_shaped: List[float] = []
        gen_best_base: List[float] = []
        gen_mean_side_lock: List[float] = []
        all_match_log: List[dict] = []

        best_shaped_overall = float("-inf")
        best_genome_overall: List[float] | None = None
        best_utt_meta: dict = {}

        balance_target_rate = float(os.environ.get("SINGLE_GENE_BALANCE_TARGET_RATE", "0.5"))
        best_balanced_distance = float("inf")
        best_balanced_denorm = float("inf")
        best_balanced_genome: List[float] | None = None
        best_balanced_meta: dict = {}

        with per_individual_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "generation",
                    "individual_index",
                    "normalized_gene",
                    "denormalized_gene",
                    "aggregate_ai1_win_rate",
                    "ai1_win_rate_for_shaping",
                    "shaping_win_rate_mode_tag",
                    "draw_rate",
                    "base_overall",
                    "dominance_penalty",
                    "draw_penalty",
                    "high_gene_penalty",
                    "target_gene_penalty",
                    "side_lock_penalty",
                    "ordering1_ai1_win_rate",
                    "ordering2_ai1_win_rate",
                    "desired_gene",
                    "experiment_mode",
                    "shaped_overall",
                ]
            )

            for gen in range(generations):
                run_log: List[dict] = []
                setattr(evaluator, "run_match_log", run_log)
                setattr(evaluator, "run_match_log_generation", gen)
                if hasattr(evaluator, "close_cached_env"):
                    evaluator.close_cached_env()

                base_scores = []
                shaped_scores: List[float] = []
                denorm_gene_values: List[float] = []
                ai1_rates: List[float] = []
                side_lock_penalties: List[float] = []

                for i, g in enumerate(population):
                    setattr(evaluator, "run_match_log_individual_index", i)
                    chromosome = chromosome_from_genome(
                        g, target_gene, use_log_scale=use_log_scale
                    )
                    fc = evaluator.evaluate_chromosome(chromosome)
                    base_scores.append(fc)

                    row = next(
                        (
                            r
                            for r in reversed(run_log)
                            if r.get("generation") == gen and r.get("individual_index") == i
                        ),
                        None,
                    ) or {}

                    ai1_wins = int(row.get("left_wins", 0))
                    ai2_wins = int(row.get("right_wins", 0))
                    draws = int(row.get("draws", 0))
                    decisive = ai1_wins + ai2_wins
                    total = decisive + draws
                    draw_rate = (draws / total) if total > 0 else 0.0
                    if use_both_orderings:
                        o1_ai1 = int(row.get("_ordering1_ai1_wins", 0))
                        o1_ai2 = int(row.get("_ordering1_ai2_wins", 0))
                        o2_ai1 = int(row.get("_ordering2_ai1_wins", 0))
                        o2_ai2 = int(row.get("_ordering2_ai2_wins", 0))
                        o1_decisive = o1_ai1 + o1_ai2
                        o2_decisive = o2_ai1 + o2_ai2
                        o1_rate = (o1_ai1 / o1_decisive) if o1_decisive > 0 else 0.5
                        o2_rate = (o2_ai1 / o2_decisive) if o2_decisive > 0 else 0.5
                        side_lock_penalty = side_lock_weight * abs(o1_rate - o2_rate)
                    else:
                        o1_decisive = decisive
                        o2_decisive = decisive
                        single_rate = (ai1_wins / decisive) if decisive > 0 else 0.5
                        o1_rate = single_rate
                        o2_rate = single_rate
                        side_lock_penalty = 0.0

                    aggregate_rate = (ai1_wins / decisive) if decisive > 0 else 0.5
                    shaping_rate, shaping_tag = resolve_shaping_ai1_win_rate(
                        aggregate_rate=aggregate_rate,
                        use_both_orderings=use_both_orderings,
                        o1_rate=o1_rate,
                        o2_rate=o2_rate,
                        o1_decisive=o1_decisive,
                        o2_decisive=o2_decisive,
                        mode=shaping_win_rate_mode,
                    )
                    ai1_win_rate = shaping_rate

                    imbalance = abs(ai1_win_rate - 0.5)
                    if experiment_mode == "dominance_penalty_mode":
                        # Target genome in [0,1] "buff strength" space used by target_gene_penalty.
                        # When high_raw_buffs_ai1: high gene helps AI1 → if AI1 wins too much, pull gene down (0).
                        # When not (e.g. Worker.hp vs lightRush): high gene helps AI2 → if AI1 wins too much, pull up (1).
                        if imbalance <= dominance_imbalance_threshold:
                            desired_gene = 0.5
                        elif ai1_win_rate > 0.5:
                            desired_gene = 0.0 if high_raw_buffs_ai1 else 1.0
                        else:
                            desired_gene = 1.0 if high_raw_buffs_ai1 else 0.0
                    else:
                        # Causal mode: intended for a signed gene→win-rate effect (probe removed: direction stays 0).
                        desired_gene = (
                            1.0
                            if (
                                (probe_effect_direction > 0.02 and ai1_win_rate < 0.5)
                                or (probe_effect_direction < -0.02 and ai1_win_rate > 0.5)
                            )
                            else (
                                0.0
                                if (
                                    (probe_effect_direction > 0.02 and ai1_win_rate > 0.5)
                                    or (probe_effect_direction < -0.02 and ai1_win_rate < 0.5)
                                )
                                else 0.5
                            )
                        )

                    parts = shape_one_gene_balancing_fitness(
                        base_overall=fc.overall_fitness,
                        normalized_gene=normalized_gene_for_balance_strength(
                            target_param, g[target_gene.index]
                        ),
                        ai1_win_rate=ai1_win_rate,
                        draw_rate=draw_rate,
                        desired_gene=desired_gene,
                        target_gene_weight=target_gene_weight,
                        high_raw_buffs_ai1=high_raw_buffs_ai1,
                    )

                    shaped_overall = max(0.0, parts.shaped_overall - side_lock_penalty)
                    shaped_scores.append(shaped_overall)
                    if shaped_overall > best_shaped_overall:
                        best_shaped_overall = shaped_overall
                        best_genome_overall = list(g)
                        best_utt_meta = {
                            "selection": "highest_shaped_fitness",
                            "generation": gen,
                            "individual_index": i,
                            "shaped_overall": shaped_overall,
                            "base_overall": float(fc.overall_fitness),
                            "denormalized_gene": float(
                                denormalize(
                                    target_gene,
                                    g[target_gene.index],
                                    use_log_scale=use_log_scale,
                                )
                            ),
                        }
                    denorm_gene = denormalize(
                        target_gene, g[target_gene.index], use_log_scale=use_log_scale
                    )
                    if decisive > 0:
                        # Per-seat balance: a genome is "balanced" only when the WORST seat is near the
                        # target. This is immune to side flips (0/1 -> worst-seat dev 0.5, not 0), unlike
                        # the old aggregate distance which crowned mirages.
                        if use_both_orderings:
                            balance_distance = max(
                                abs(o1_rate - balance_target_rate),
                                abs(o2_rate - balance_target_rate),
                            )
                        else:
                            balance_distance = abs(aggregate_rate - balance_target_rate)
                        if should_replace_balanced_candidate(
                            new_distance=balance_distance,
                            new_denorm=float(denorm_gene),
                            best_distance=best_balanced_distance,
                            best_denorm=best_balanced_denorm,
                        ):
                            best_balanced_distance = balance_distance
                            best_balanced_denorm = float(denorm_gene)
                            best_balanced_genome = list(g)
                            best_balanced_meta = {
                                "selection": "closest_to_balanced_win_rate",
                                "balance_distance_metric": (
                                    "max_seat_deviation_from_target"
                                    if use_both_orderings
                                    else "aggregate_deviation_from_target"
                                ),
                                "balance_target_rate": balance_target_rate,
                                "balance_distance": balance_distance,
                                "seat_gap": abs(o1_rate - o2_rate),
                                "generation": gen,
                                "individual_index": i,
                                "ai1_win_rate_for_shaping": ai1_win_rate,
                                "aggregate_ai1_win_rate": aggregate_rate,
                                "shaping_win_rate_mode_tag": shaping_tag,
                                "denormalized_gene": float(denorm_gene),
                                "shaped_overall": shaped_overall,
                                "base_overall": float(fc.overall_fitness),
                                "draw_rate": draw_rate,
                                "ordering1_ai1_win_rate": o1_rate,
                                "ordering2_ai1_win_rate": o2_rate,
                            }
                    denorm_gene_values.append(denorm_gene)
                    ai1_rates.append(ai1_win_rate)
                    side_lock_penalties.append(side_lock_penalty)

                    w.writerow(
                        [
                            gen,
                            i,
                            g[target_gene.index],
                            denorm_gene,
                            aggregate_rate,
                            ai1_win_rate,
                            shaping_tag,
                            draw_rate,
                            parts.base_overall,
                            parts.dominance_penalty,
                            parts.draw_penalty,
                            parts.high_gene_penalty,
                            parts.target_gene_penalty,
                            side_lock_penalty,
                            o1_rate,
                            o2_rate,
                            desired_gene,
                            experiment_mode,
                            shaped_overall,
                        ]
                    )

                mean_gene = sum(denorm_gene_values) / len(denorm_gene_values)
                variance = sum((x - mean_gene) ** 2 for x in denorm_gene_values) / len(denorm_gene_values)
                std_gene = math.sqrt(variance)
                best_idx = max(range(len(shaped_scores)), key=lambda i: shaped_scores[i])

                gen_mean_gene.append(mean_gene)
                gen_std_gene.append(std_gene)
                gen_best_gene.append(denorm_gene_values[best_idx])
                gen_mean_ai1_win.append(sum(ai1_rates) / len(ai1_rates))
                gen_best_ai1_win.append(ai1_rates[best_idx])
                gen_mean_shaped.append(sum(shaped_scores) / len(shaped_scores))
                gen_best_shaped.append(shaped_scores[best_idx])
                gen_best_base.append(max(fc.overall_fitness for fc in base_scores))
                gen_mean_side_lock.append(sum(side_lock_penalties) / len(side_lock_penalties))

                log(
                    f"[Gen {gen}] mean_gene={mean_gene:.3f} best_gene={denorm_gene_values[best_idx]:.3f} "
                    f"| mean_ai1_win={gen_mean_ai1_win[-1]:.3f} best_ai1_win={ai1_rates[best_idx]:.3f} "
                    f"| mean_side_lock_pen={gen_mean_side_lock[-1]:.3f} "
                    f"| best_shaped={shaped_scores[best_idx]:.4f}"
                )

                all_match_log.extend(run_log)

                def tournament_pick() -> List[float]:
                    k = min(3, population_size)
                    cand = random.sample(range(population_size), k)
                    return population[max(cand, key=lambda i: shaped_scores[i])]

                # Anneal exploration over the run so the gene settles instead of wandering:
                #   frac 0 -> 1 across generations; sigma decays to floor, jump rate decays to 0.
                frac = gen / max(1, generations - 1)
                sigma_eff = mutation_sigma * (sigma_decay_floor + (1.0 - sigma_decay_floor) * (1.0 - frac))
                if jump_off_frac <= 0.0:
                    jump_eff = mutation_jump_rate
                else:
                    jump_eff = mutation_jump_rate * max(0.0, 1.0 - frac / jump_off_frac)

                # Elitism: carry the top-N genomes unchanged so a found balance is never lost.
                elite_idx = sorted(
                    range(population_size), key=lambda i: shaped_scores[i], reverse=True
                )[:elite_count]
                next_population: List[List[float]] = [list(population[i]) for i in elite_idx]

                while len(next_population) < population_size:
                    p1, p2 = tournament_pick(), tournament_pick()
                    c1, c2 = list(p1), list(p2)
                    if random.random() < crossover_rate and random.random() < 0.5:
                        c1[target_gene.index], c2[target_gene.index] = c2[target_gene.index], c1[target_gene.index]
                    c1 = mutate_single_gene(
                        c1,
                        target_gene,
                        sigma_eff,
                        jump_eff,
                        use_log_scale=use_log_scale,
                    )
                    c2 = mutate_single_gene(
                        c2,
                        target_gene,
                        sigma_eff,
                        jump_eff,
                        use_log_scale=use_log_scale,
                    )
                    next_population.append(c1)
                    if len(next_population) < population_size:
                        next_population.append(c2)
                population = next_population[:population_size]

        _write_match_log_artifacts(run_dir, all_match_log, save_game_details)

        if best_balanced_genome is not None:
            balanced_utt_path = run_dir / "best_balanced_chromosome_utt.json"
            balanced_meta_path = run_dir / "best_balanced_chromosome_meta.json"
            _save_chromosome_utt_files(
                evaluator,
                best_balanced_genome,
                target_gene,
                use_log_scale=use_log_scale,
                utt_path=balanced_utt_path,
                meta=best_balanced_meta,
                meta_path=balanced_meta_path,
                ai1=ai1,
                ai2=ai2,
                map_path=map_paths[0],
                target_unit=target_unit,
                target_param=target_param,
            )
            log(
                f"Saved best balanced UTT (closest to {balance_target_rate:.0%} win rate, "
                f"distance={best_balanced_distance:.4f}, gene={best_balanced_denorm:.0f}): "
                f"{balanced_utt_path}"
            )
            log(f"Best balanced meta: {balanced_meta_path}")

        if best_genome_overall is not None:
            best_utt_path = run_dir / "best_chromosome_utt.json"
            meta_path = run_dir / "best_chromosome_meta.json"
            _save_chromosome_utt_files(
                evaluator,
                best_genome_overall,
                target_gene,
                use_log_scale=use_log_scale,
                utt_path=best_utt_path,
                meta=best_utt_meta,
                meta_path=meta_path,
                ai1=ai1,
                ai2=ai2,
                map_path=map_paths[0],
                target_unit=target_unit,
                target_param=target_param,
            )
            log(f"Saved best UTT (highest shaped fitness over all gens): {best_utt_path}")
            log(f"Best shaped-fitness meta: {meta_path}")

        with per_generation_csv.open("w", newline="", encoding="utf-8") as f:
            sw = csv.writer(f)
            sw.writerow(
                [
                    "generation",
                    "mean_gene_denorm",
                    "std_gene_denorm",
                    "best_gene_denorm",
                    "mean_ai1_win_rate",
                    "best_ai1_win_rate",
                    "mean_side_lock_penalty",
                    "mean_shaped_fitness",
                    "best_shaped_fitness",
                    "best_base_fitness",
                ]
            )
            for g in range(generations):
                sw.writerow(
                    [
                        g,
                        gen_mean_gene[g],
                        gen_std_gene[g],
                        gen_best_gene[g],
                        gen_mean_ai1_win[g],
                        gen_best_ai1_win[g],
                        gen_mean_side_lock[g],
                        gen_mean_shaped[g],
                        gen_best_shaped[g],
                        gen_best_base[g],
                    ]
                )

        x = list(range(generations))
        plt.figure(figsize=(9, 4.5))
        plt.plot(x, gen_mean_gene, label="Mean gene", color="C0")
        plt.plot(x, gen_best_gene, label="Best individual gene", color="C1", linestyle="--")
        plt.fill_between(
            x,
            [m - s for m, s in zip(gen_mean_gene, gen_std_gene)],
            [m + s for m, s in zip(gen_mean_gene, gen_std_gene)],
            color="C0",
            alpha=0.2,
            label="±1 std",
        )
        plt.xlabel("Generation")
        if target_param in _PARAM_LOWER_RAW_IS_STRONGER:
            plt.ylabel(
                f"{target_gene.unit_type}.{target_gene.param_name} (ticks; ↑ slower = nerf)"
            )
            plt.title("One-gene trajectory (time stat: rising curve = slowing unit)")
        else:
            plt.ylabel(f"{target_gene.unit_type}.{target_gene.param_name}")
            plt.title("One-gene decrease trajectory")
        plt.legend()
        plt.tight_layout()
        plt.savefig(decrease_plot, dpi=120)
        plt.close()

        plt.figure(figsize=(9, 4.5))
        plt.plot(x, gen_mean_ai1_win, "m-o", markersize=4, label=f"Mean {ai1} win-rate")
        plt.plot(x, gen_best_ai1_win, "c--", label=f"Best-individual {ai1} win-rate")
        plt.axhline(0.5, color="gray", linestyle=":", label="Balanced target (0.5)")
        plt.xlabel("Generation")
        plt.ylabel("Win rate")
        plt.title("Outcome balance trajectory")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outcome_plot, dpi=120)
        plt.close()

        plt.figure(figsize=(9, 4.5))
        plt.plot(x, gen_best_shaped, "b-o", markersize=4, label="Best shaped fitness")
        plt.plot(x, gen_mean_shaped, "g-s", markersize=4, label="Mean shaped fitness")
        plt.plot(x, gen_best_base, color="gray", linestyle=":", label="Best base overall")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness trajectory")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fitness_plot, dpi=120)
        plt.close()

        log("")
        log("Finished single-gene balance-decrease run")
        log(f"Per-individual CSV: {per_individual_csv}")
        log(f"Per-generation CSV: {per_generation_csv}")
        log(f"Decrease plot: {decrease_plot}")
        log(f"Outcome plot: {outcome_plot}")
        log(f"Fitness plot: {fitness_plot}")
        log(f"UTT snapshots dir: {utt_snapshots_dir}")
        log(f"Best balanced UTT: {run_dir / 'best_balanced_chromosome_utt.json'}")
        log(f"Best shaped-fitness UTT: {run_dir / 'best_chromosome_utt.json'}")
        log("=" * 60)
    finally:
        stream_capture.__exit__(None, None, None)


if __name__ == "__main__":
    main()
