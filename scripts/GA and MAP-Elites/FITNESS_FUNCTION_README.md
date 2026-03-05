# GA Fitness Function: How It’s Calculated and Why

This document describes how fitness is computed for the MicroRTS UTT (Unit Type Table) genetic algorithm and the rationale behind each choice. The goal is to **evolve UTTs that make fair, interesting games when both players use the same UTT** — i.e. balanced matchups and reasonable game length, not “which side” wins.

---

## 1. Overall fitness formula

```text
overall_fitness = α × balance + β × duration + γ × strategy_diversity
```

- **α (alpha)** — weight for balance (default 0.4; often raised to ~0.7 when balance is the main goal).  
  **Why:** Balance (fairness between the two AIs) is the primary design goal; the other terms avoid degenerate UTTs and add selection pressure.

- **β (beta)** — weight for duration.  
  **Why:** We want games that end in a reasonable time (not instant wipes or endless draws), so duration contributes to “interesting” games.

- **γ (gamma)** — weight for strategy diversity.  
  **Why:** We want the UTT to behave sensibly across different AI pairs and outcome patterns, not just one specific matchup.

All three components are in [0, 1]; weights are typically set so α + β + γ = 1.0.

---

## 2. Balance (fairness between the two AIs)

**What we measure:** How close the win rate is to 50–50 **between the two AIs** (e.g. lightRushAI vs workerRushAI), not “left vs right.”

### 2.1 Why “by AI” instead of “by side (left/right)”?

When we run **both orderings** — (ai1 vs ai2) and (ai2 vs ai1) — we aggregate wins by **AI identity**:

- `ai1_wins = wins when ai1 was left + wins when ai1 was right`
- `ai2_wins = wins when ai2 was left + wins when ai2 was right`

**Why:** We care whether the **matchup** is fair (each AI wins about half the time), not whether “the left player” wins half the time. If we used left/right, a 5–0 in one ordering and 0–5 in the other would still look 5–5 by side, but we explicitly want “AI1 wins 5, AI2 wins 5” so that balance rewards **fairness between the two strategies**, independent of which side they play on.

### 2.2 Win ratio and imbalance

- **Decisive games** = games that ended in a win (not a draw): `decisible = ai1_wins + ai2_wins`.
- **Win ratio** (for AI1) = `ai1_wins / decisive`.
- **Imbalance** = `|win_ratio - 0.5|` (0 = perfect 50–50, 0.5 = one AI wins everything).

**Why use only decisive games for the ratio?** Draws don’t tell us which AI is stronger; we want “of the games that had a winner, did each AI win about half?” So we measure balance on decisive games only.

### 2.3 Base balance score

- **Base balance** = `1.0 - imbalance × 2`  
  So: 50–50 → 1.0, 100–0 → 0.0, 60–40 → 0.8.

**Why linear in imbalance?** Simple, interpretable gradient: the further from 50–50, the lower the score.

### 2.4 Stricter penalty for very imbalanced matchups (`use_strict_balance`)

When `use_strict_balance` is True we apply **extra** penalty for highly imbalanced results:

- **Imbalance > 0.4** (e.g. 80–20 or worse): multiply base balance by `(imbalance / 0.5)^2`.
- **Imbalance > 0.3** (e.g. 60–40 to 80–20): multiply by `(imbalance / 0.5)^1.5`.
- **Imbalance ≤ 0.3**: use base balance as is.

**Why:** We want to strongly discourage UTTs where **any** matchup is a landslide. One 90–10 matchup should hurt overall balance more than a linear penalty would, so the GA is pushed toward UTTs that are fair across matchups, not just on average.

### 2.5 All draws → balance = 0

If every game in a matchup is a draw (`decisible == 0`), we set **balance = 0** for that matchup.

**Why:** All-draw outcomes usually mean games timed out or got stuck (e.g. hitting max steps). We don’t want the GA to “solve” balance by making games never end; we want real wins/losses and a 50–50 split between the two AIs.

### 2.6 Aggregating balance across multiple matchups (geometric mean)

When we have several matchups (e.g. multiple AI pairs or maps), we don’t use a simple average of per-matchup balance scores. We use the **geometric mean** of those scores (with a small epsilon to avoid log(0)).

**Why:** The geometric mean is dominated by the **lowest** scores. So one very imbalanced matchup (e.g. 10–0) drags the overall balance down a lot. That matches the goal: we want UTTs that are balanced in **every** tested matchup, not UTTs that are balanced on average but terrible in one pairing.

### 2.7 Minimum balance threshold (`min_balance_threshold`)

If **any** single matchup has a balance score below `min_balance_threshold`, we apply an extra penalty to the overall balance (square-root factor based on how far below the threshold the worst matchup is).

**Why:** Another way to enforce “no matchup should be a total stomp.” Even if the geometric mean is okay, one 0–10 matchup is unacceptable, so we penalize it explicitly.

---

## 3. Duration (game length)

We want games that are **neither too short (curb-stomp) nor too long (endless draws/timeouts)**.

### 3.1 Step-based duration (preferred)

When we have **total steps** for the matchup (sum of steps over all games):

- **Avg steps per game** = `total_steps / total_games`.
- We define a **target duration** (e.g. 500 steps) and a **tolerance** (e.g. 400 steps).
- **Duration score**:
  - 1.0 when avg steps = target.
  - Linear decay to 0.0 when avg steps = target ± tolerance.
  - 0.0 when avg steps is outside that band.

**Why step-based?** Under a **symmetric UTT** (same UTT for both players), many UTT changes affect **both** sides equally, so win rates alone may not change much. Game **length** (steps) does change with unit costs, HP, damage, etc. So step-based duration gives the GA a **gradient** that still moves when we optimize for “fair, interesting games” with both players on the same UTT.

**Why a band (target ± tolerance) instead of “longer is better”?** We explicitly do **not** want the GA to maximize game length (that would favor endless or draw-heavy games). We want a “sweet spot”: long enough to be interesting, short enough to finish. The band encodes that.

### 3.2 Fallback: draw-ratio–based duration

When total steps are not available (e.g. older logs or a different evaluation path), we use **draw ratio** = draws / total_games:

- Low draw ratio (e.g. ≤ 0.3): duration score 1.0 (games are finishing with a winner).
- Higher draw ratio: stepwise lower scores (0.8, 0.5, 0.15).

**Why:** Many draws usually mean games are going to timeout or are too long. So “fewer draws” is a proxy for “games complete in a reasonable time” when we don’t have step counts.

---

## 4. Strategy diversity

**What we measure:** Variety in how the UTT behaves across different AI pairs and outcome patterns.

### 4.1 Why we include diversity

If we only optimized balance and duration for **one** AI pair, we could get a UTT that is balanced only for that pair and breaks others. Diversity encourages the GA to consider **multiple** matchups and **different** outcome types (some 5–5, some 6–4, etc.), so the evolved UTT is more robust and not overfitted to a single matchup.

### 4.2 How diversity is computed (short version)

- **If there are ≥ 2 matchups**  
  `strategy_diversity = 0.4·ai_diversity + 0.3·variance_score + 0.3·pattern_diversity`

- **If there is only 1 matchup**  
  `strategy_diversity = 0.3·ai_diversity`

Where (all in [0, 1]):

We combine three sub-metrics:

1. **AI diversity**  
   Formula: `ai_diversity = min(1.0, #unique AIs seen / #AIs configured)`  
   **Why:** More distinct AIs tested ⇒ more evidence that the UTT is reasonable across strategies.

2. **Variance in balance scores**  
   - Let `balance_scores` be the per‑matchup balance values and `μ` their mean.  
   - `balance_variance = average( (score − μ)² )`  
   - `variance_score = min(1.0, balance_variance × 2)`  
   **Why:** We want **some** spread: not every matchup should be identical (e.g. all 5–5), but we also don’t want one matchup 10–0 and another 0–10. Variance captures “varied but not chaotic” outcomes.

3. **Outcome-pattern diversity**  
   - For each matchup, form a triple `(left_wins, right_wins, draws)`.  
   - Let `P = #distinct triples` and `M = #matchups`.  
   - `pattern_diversity = min(1.0, P / M)`  
   **Why:** Different patterns (e.g. 5–5 vs 6–4) indicate that the UTT produces different kinds of games across matchups, which we treat as more interesting and robust.

When there is **more than one** matchup:

- `strategy_diversity = 0.4 × ai_diversity + 0.3 × variance_score + 0.3 × pattern_diversity`

When there is **only one** matchup (e.g. a single AI pair):

- `strategy_diversity = 0.3 × ai_diversity`  
  **Why:** Variance and pattern diversity need multiple matchups to be meaningful; with one matchup we only reward “we at least ran the evaluation” via AI diversity.

---

## 5. Both orderings (why we run (ai1, ai2) and (ai2, ai1))

For each AI pair we run:

1. **Ordering 1:** ai1 (left) vs ai2 (right) — N games.  
2. **Ordering 2:** ai2 (left) vs ai1 (right) — N games.

Wins are then aggregated **by AI** (see Section 2.1), so we get `ai1_wins` and `ai2_wins` over 2N games.

**Why both orderings?**

- **Removes side bias:** Any first-player or map asymmetry is averaged over the two orderings, so we measure “AI1 vs AI2” rather than “left vs right.”
- **Balance by AI:** With both orderings we can define balance as “each AI wins about half the time,” which is what we want for fairness.
- **More data:** 2N games per pair gives a less noisy estimate of the true win rate for each AI.

So “both orderings” is what allows the fitness to depend on **the AIs themselves**, not on which side they played.

---

## 6. Symmetric UTT (both players use the same UTT)

In this GA setup, **both** players use the **same** evolved UTT in every game.

**Why:**

- The design goal is “find UTTs that make **fair, interesting games when both use it**.” So we evaluate under the same rules for both sides.
- If we gave only one side the evolved UTT (asymmetric), we would be optimizing “how well this UTT does vs the default,” which is a different objective (and can lead to overpowered UTTs rather than balanced ones).

So symmetric UTT + balance (by AI) + duration + diversity together implement “fair, interesting games under one shared UTT.”

---

## 7. Summary table

| Component        | What we measure                          | Why we do it this way                                                                 |
|-----------------|------------------------------------------|----------------------------------------------------------------------------------------|
| **Balance**     | 50–50 win rate between the two AIs       | Fair matchup; by AI (not side) so fairness is about strategies, not left/right.     |
| **Strict balance** | Extra penalty for 80–20 or worse     | Discourage any highly imbalanced matchup, not just low average balance.              |
| **All draws = 0** | Balance = 0 if no decisive games      | Avoid rewarding timeouts / endless games as “balanced.”                              |
| **Geometric mean** | Over matchups for balance              | One bad matchup should strongly reduce overall balance.                               |
| **Duration**    | Avg steps per game in a target band      | Sweet-spot length; gives gradient under symmetric UTT; avoid stomps and timeouts.    |
| **Diversity**   | AI coverage, balance variance, patterns  | Robustness across matchups; avoid overfitting to one pair or one outcome type.        |
| **Both orderings** | (ai1 vs ai2) and (ai2 vs ai1)         | Balance by AI, remove side bias, more stable win-rate estimate.                      |
| **Symmetric UTT** | Same UTT for both players              | Objective is “fair, interesting when both use it,” not “beat default.”                |

---

## 8. Where this is implemented

- **Evaluator and fitness calculation:** `scripts/GA and MAP-Elites/core/ga_working_evaluator.py`  
  - `_calculate_fitness()` — balance, duration, diversity and overall formula.  
  - Both-orderings aggregation and `ai1_wins` / `ai2_wins` — in `_test_utt_file()` (round-robin / match loop).
- **Weights and options:** Set in `GAConfig` (e.g. in `run_ga_local_test.py`) and passed into `WorkingGAEvaluator` (e.g. `fitness_alpha`, `fitness_beta`, `fitness_gamma`, `use_both_orderings`, `target_duration`, `duration_tolerance`, `use_strict_balance`, `min_balance_threshold`).
