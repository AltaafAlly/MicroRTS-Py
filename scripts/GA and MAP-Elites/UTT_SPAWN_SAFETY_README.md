## UTT Spawn Safety: Why Some UTTs Made “Nothing Spawn” and How the GA Avoids It Now

This note explains **why some evolved UTTs caused matches where almost nothing new was produced** (beyond the starting units) and **what changes we made in the GA to prevent that pattern from being generated again**.

The goal is that **every evolved UTT should still lead to “real games”**: bases build workers, workers harvest, barracks and armies get produced, and the fitness function evaluates meaningful battles – not frozen economies.

---

### 1. What went wrong (examples: `gen0_ind3`, `gen15_ind2`)

We saw UTTs like:

- `gym_microrts/microrts/utts/gen0_ind3.json`
- `gym_microrts/microrts/utts/gen15_ind2.json`

When loaded in the MicroRTS GUI, **bases and AIs essentially stopped producing anything**, even though there were resources on the map.

The root cause was a combination of:

- **Map starting resources** on the standard 8×8 maps:
  - `basesWorkers8x8*.xml` give each player **5 starting resources**.
- **Evolved Worker parameters** that made the opening economy too tight or fragile:
  - Example: Worker **cost = 5** (or higher in earlier runs).
  - Slow or extreme harvest parameters (very long harvest/return times or very high harvestAmount).

Concretely:

- If **Worker cost ≥ starting resources**, the Base **cannot afford to build more than one Worker** (or zero in the worst case).
- If that **first Worker dies early** (rushes, bad micro, etc.) before returning enough resources to fund more Workers/Barracks, the player is **stuck with no harvesters**.
- With expensive Barracks and combat units, and no Worker production, **the economy freezes** and from the GUI’s point of view it looks like “nothing ever spawns.”

These UTTs are technically valid, but they lead to **degenerate games** that the fitness function doesn’t really want to reward.

---

### 2. How the GA now avoids “no-spawn” UTTs

We fixed this **at the GA / UTT-generation level**, rather than manually editing every bad UTT, by:

#### 2.1 Worker cost bounds tightened (chromosome side)

In `core/ga_chromosome.py` (`DEFAULT_PARAMETER_BOUNDS`):

- **Before (early versions)**: Worker cost/econ bounds were wide enough to allow both “too expensive to open” and “ultra-slow econ” Workers.
- **Now**:
  - Worker **cost** bounds are **`(2, 4)`** (with the validator clamping to ≤ 4).
  - Worker **econ** is also constrained:
    - `harvestTime` in **`[6, 12]`**
    - `returnTime` in **`[4, 8]`**
    - `harvestAmount` in **`[2, 4]`**

This means:

- On 8×8 maps with **5 starting resources**, the Base can **always afford at least one extra Worker** with a reasonable path toward Barracks/units.
- The GA can no longer evolve Workers that are **too expensive to start** or so slow that income is effectively frozen.

#### 2.2 Worker econ bounds aligned and clamped (validator side)

In `core/ga_utt_validator.py` (`SAFE_BOUNDS`):

- Worker **cost** bounds are now **`(1, 4)`**.
- Worker economy parameters are constrained to **reasonable ranges**:
  - `harvestTime` in **`[6, 16]`**
  - `returnTime` in **`[4, 10]`**
  - `harvestAmount` in **`[2, 5]`**

The validator is applied in two places:

- When we **export UTT configs** from chromosomes (for safety/tools).
- In the **working evaluator** before writing the UTT JSON that Java uses:
  - `WorkingGAEvaluator._create_utt_file()` now calls `UTTValidator.validate_and_fix_utt(...)` on the generated config.

This guarantees that:

- Even if an old chromosome from a previous run had an out-of-range Worker cost or weird econ, the **actual UTT JSON used for matches is clamped back into safe bounds**.

#### 2.3 Structural safety: production graph

We keep the **production graph structure** fixed in the GA:

- `Base` **produces** `Worker`.
- `Worker` **produces** `Base` and `Barracks`.
- `Barracks` **produces** `Light`, `Heavy`, `Ranged`.

The GA is not allowed to break this graph (no “Base produces nothing” genomes). Combined with the numeric bounds above, this ensures:

- There is always a **path from starting resources → Worker(s) → Barracks → army units**.

---

### 3. Why we chose bounds (instead of only fitness penalties)

We could have tried to let the fitness function punish “no spawn” UTTs by giving them terrible scores. In practice:

- When **nothing spawns**, games often hit the **max steps** limit or have very few decisive battles.
- The fitness terms (balance, duration, diversity) then become noisy or misleading for these cases.

By adding **hard parameter bounds**:

- The GA **never explores obviously dead regions** of the search space (like Worker cost ≥ starting resources).
- The fitness function can focus on **interesting trade-offs** (rush speed, unit mix, time-to-kill, etc.) instead of constantly rejecting degenerate “no economy” UTTs.

---

### 4. How to recognize and debug a bad UTT in the future

If you ever see another UTT where “nothing seems to spawn”:

1. **Check Worker cost vs starting resources**
   - For 8×8 maps with 5 starting resources, Worker cost should be **≤ 4** under the new bounds.
2. **Check Worker econ parameters**
   - Very high `harvestTime` + `returnTime` with low `harvestAmount` can make the economy painfully slow.
3. **Verify the production graph**
   - `Base.produces` should include `"Worker"`.
   - `Worker.producedBy` should include `"Base"`.
   - `Barracks.producedBy` should include `"Worker"`.

If a UTT violates any of these, you can:

- Edit the JSON manually for experimentation, **or**
- Adjust the GA bounds / `SAFE_BOUNDS` so the GA can’t generate that pattern in future runs.

With the current bounds and validator wiring, new GA runs should **no longer produce “nothing spawns” UTTs**, and existing ones are automatically clamped before they are sent to the Java engine.

# Broken down for gen15_ind2.json
Starting resources on the map: 5.
Worker in that UTT:
cost = 5 → building one Worker already spends all starting resources.
Harvesting is slow (13 + 10 cycles per trip) → income comes in very slowly.
Light in that UTT:
cost = 8.
Barracks in that UTT:
cost = 23.
So in a Light vs Worker matchup:

At the start, the base can:
Build at most one new Worker (and maybe not even that, depending on timing), but
Cannot afford a Light or a Barracks until that Worker has done several long harvest-return trips.
If that early Worker dies before generating enough extra resources:
The player is stuck with no workers and not enough money to build another Worker, a Light, or a Barracks.
After that, nothing new gets produced → the game looks “dead.”
So it isn’t just that Workers are “too expensive” for Lights; it’s that Worker cost + slow econ + low starting resources combine so that the economy often never reaches the point where it can afford more Workers or any combat units.