"""
Fitness shaping for one-gene balancing experiments.

This module is intentionally simple: when AI1 wins too often and the selected
gene is high, fitness gets penalized more strongly so evolution prefers lower
gene values and moves back toward balanced outcomes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OneGeneFitnessParts:
    base_overall: float
    dominance_penalty: float
    draw_penalty: float
    high_gene_penalty: float
    target_gene_penalty: float
    shaped_overall: float


def shape_one_gene_balancing_fitness(
    *,
    base_overall: float,
    normalized_gene: float,
    ai1_win_rate: float,
    draw_rate: float,
    target_win_rate: float = 0.5,
    dominance_tolerance: float = 0.08,
    dominance_weight: float = 0.65,
    draw_weight: float = 0.30,
    high_gene_weight: float = 0.12,
    desired_gene: Optional[float] = None,
    target_gene_weight: float = 0.25,
    high_raw_buffs_ai1: bool = True,
) -> OneGeneFitnessParts:
    """
    Shape fitness for one-gene balancing runs.

    Key behavior:
    - Keep shaped fitness informative (avoid flattening everything to zero).
    - Penalize imbalance, but scale base fitness instead of subtracting a large
      constant so selection still has gradient.
    - Apply directional gene pressure on ``normalized_gene`` in [0,1]
      ("buff strength" for the chosen stat: higher = stronger in the usual damage/hp sense).
    - If ``high_raw_buffs_ai1`` is True (default): higher gene helps AI1 — AI1 overpowered
      penalize high gene (push down); AI1 underpowered penalize low gene (push up).
    - If ``high_raw_buffs_ai1`` is False (e.g. evolving ``Worker.hp`` while AI1 is light rush):
      higher gene helps AI2 — flip those two branches so balance pressure moves the right way.
    """
    g = max(0.0, min(1.0, float(normalized_gene)))
    r = max(0.0, min(1.0, float(ai1_win_rate)))
    dr = max(0.0, min(1.0, float(draw_rate)))
    base = max(0.0, float(base_overall))

    imbalance = abs(r - target_win_rate)
    effective_imbalance = max(0.0, imbalance - dominance_tolerance)
    # Scale fitness by imbalance instead of subtracting a large constant.
    # This avoids total collapse (all zeros), which destroys selection pressure.
    dominance_scale = max(0.25, 1.0 - dominance_weight * effective_imbalance * 2.0)
    dominance_penalty = base * (1.0 - dominance_scale)

    # Draw penalty scales with base, for the same reason as dominance scaling.
    draw_penalty = draw_weight * dr * base

    # Directional pressure on normalized gene g ("buff strength" for the stat).
    over_target = max(0.0, r - target_win_rate)
    under_target = max(0.0, target_win_rate - r)
    if high_raw_buffs_ai1:
        high_gene_penalty = high_gene_weight * (over_target * g + under_target * (1.0 - g))
    else:
        high_gene_penalty = high_gene_weight * (over_target * (1.0 - g) + under_target * g)

    target_gene_penalty = 0.0
    if desired_gene is not None:
        dg = max(0.0, min(1.0, float(desired_gene)))
        target_gene_penalty = target_gene_weight * effective_imbalance * abs(g - dg)

    shaped = base * dominance_scale - draw_penalty - high_gene_penalty - target_gene_penalty
    shaped = max(0.0, shaped)

    return OneGeneFitnessParts(
        base_overall=base,
        dominance_penalty=dominance_penalty,
        draw_penalty=draw_penalty,
        high_gene_penalty=high_gene_penalty,
        target_gene_penalty=target_gene_penalty,
        shaped_overall=shaped,
    )
