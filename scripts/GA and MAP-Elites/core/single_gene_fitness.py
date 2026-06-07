"""
Single-gene fitness shaping for MicroRTS GA experiments.

The default WorkingGAEvaluator already rewards matchup balance (e.g. light vs worker
win rates near 50/50). When one parameter is intentionally started at an extreme
(e.g. Light maxDamage near maximum), balance alone can be a weak or slow signal.

This module combines the base fitness with an optional **repair** term that
prefers lowering the normalized gene value when balance is poor — matching the
intuition “if the matchup is one-sided and damage was cranked up, try reducing it.”

Modes:
- ``base``: use evaluator output unchanged (only selection uses base overall_fitness).
- ``repair_imbalance``: add a bonus proportional to (1 - normalized_gene) when
  balance is below a threshold, so evolution can move the gene down across generations
  while still primarily optimizing for real match balance.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.ga_fitness_evaluator import FitnessComponents


def apply_single_gene_fitness(
    base: "FitnessComponents",
    normalized_gene: float,
    *,
    mode: str = "repair_imbalance",
    balance_threshold: float = 0.55,
    w_repair: float = 0.22,
    ai1_win_rate: float | None = None,
    target_ai1_win_rate: float = 0.5,
    w_directional: float = 0.5,
    w_gene_down: float = 0.0,
    cap_overall: float = 1.0,
) -> "FitnessComponents":
    """
    Return a copy of ``base`` with ``overall_fitness`` optionally reshaped.

    ``repair_imbalance``:
        stress = max(0, balance_threshold - balance)
        bonus = w_repair * stress * (1.0 - normalized_gene)
        overall = min(cap_overall, base.overall_fitness + bonus)

    When balance is good (>= threshold), bonus is zero — selection is driven by
    balance/duration/diversity only. When balance is bad, genomes with lower
    normalized gene values get a higher overall score (for parameters where
    "high" means overpowered, e.g. maxDamage).
    """
    from core.ga_fitness_evaluator import FitnessComponents  # local import for runtime

    g = max(0.0, min(1.0, float(normalized_gene)))
    if mode == "base":
        return base

    if mode == "repair_imbalance":
        b = float(base.balance)
        stress = max(0.0, balance_threshold - b)
        bonus = w_repair * stress * (1.0 - g)
        overall = min(cap_overall, float(base.overall_fitness) + bonus)
        return replace(base, overall_fitness=overall)

    if mode == "directional_balance_gene":
        # Use directional pressure when AI1 (light) wins too much:
        # - if ai1_win_rate > target, reward lower gene values
        # - if ai1_win_rate <= target, no directional bonus/penalty
        # This avoids the "flat 0.8 balance plateau" where high gene values survive.
        r = target_ai1_win_rate if ai1_win_rate is None else float(ai1_win_rate)
        overpowered = max(0.0, r - target_ai1_win_rate)  # only when AI1 is too strong
        directional_bonus = w_directional * overpowered * (1.0 - g)
        mild_gene_pressure = w_gene_down * (1.0 - g)
        overall = min(cap_overall, float(base.overall_fitness) + directional_bonus + mild_gene_pressure)
        return replace(base, overall_fitness=overall)

    raise ValueError(f"Unknown single-gene fitness mode: {mode!r}")


def stagnation_penalty(
    *,
    ai1_win_rate: float,
    draw_rate: float,
    prev_ai1_win_rate: float | None,
    prev_draw_rate: float | None,
    tolerance: float = 0.01,
    weight: float = 0.08,
    imbalance_gate: float = 0.08,
) -> float:
    """
    Penalty for no outcome movement across generations.

    If current outcome metrics are very close to the previous generation's
    mean metrics, return a positive penalty to subtract from shaped fitness.
    """
    if prev_ai1_win_rate is None or prev_draw_rate is None:
        return 0.0

    # Only penalize stagnation if we are in an imbalanced regime.
    # If outcomes are already near 50-50 with low draws, no stagnation penalty should apply.
    curr_imb = abs(float(ai1_win_rate) - 0.5)
    prev_imb = abs(float(prev_ai1_win_rate) - 0.5)
    if max(curr_imb, prev_imb) < imbalance_gate:
        return 0.0

    d_win = abs(float(ai1_win_rate) - float(prev_ai1_win_rate))
    d_draw = abs(float(draw_rate) - float(prev_draw_rate))
    d = d_win + d_draw
    if d >= tolerance:
        return 0.0
    # Maximum penalty when d == 0; linear decay to 0 at tolerance
    return max(0.0, weight * (1.0 - (d / tolerance)))
