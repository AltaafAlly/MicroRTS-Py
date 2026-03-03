# Balance Improvements Summary

## Quick Overview

Implemented four key improvements to increase balance scores in GA evolution:

1. ✅ **Increased balance weight** from 40% to 50% (alpha: 0.4 → 0.5)
2. ✅ **Exponential penalties** for very imbalanced matchups (>70-30 split)
3. ✅ **Geometric mean aggregation** (instead of arithmetic mean) - more sensitive to low scores
4. ✅ **Minimum balance threshold** with additional penalty for configurations with extremely imbalanced matchups

## Expected Impact

| Metric | Before | After (Target) | Improvement |
|--------|--------|----------------|-------------|
| **Balance Score** | 0.28 (28%) | 0.40-0.50+ (40-50%) | +43-79% |
| **One-sided Matchups** | Many 5-0-0, 0-5-0 | Very few or none | -80%+ |
| **Very Imbalanced** | Many 4-1-0, 1-4-0 | Rare | -60%+ |
| **Reasonably Balanced** | Some 3-2-0, 2-3-0 | Most matchups | +200%+ |

## Files Modified

1. **`core/ga_working_evaluator.py`**
   - Added `min_balance_threshold` and `use_strict_balance` parameters
   - Implemented exponential penalty calculation
   - Changed from arithmetic to geometric mean aggregation
   - Added minimum balance threshold penalty

2. **`core/ga_algorithm.py`**
   - Updated default fitness weights: alpha=0.5, beta=0.25, gamma=0.25

3. **Documentation**
   - Updated `cluster/README.md` with balance improvements section
   - Created `BALANCE_IMPROVEMENTS.md` with detailed explanation

## How to Use

**Default (Recommended)**: All improvements are enabled by default. Just run the GA as normal:

```bash
sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_ga.sbatch
```

**Custom Configuration**: To adjust balance strictness:

```python
# In your GA configuration
evaluator = WorkingGAEvaluator(
    alpha=0.6,  # Even higher balance weight (60%)
    beta=0.2,
    gamma=0.2,
    use_strict_balance=True,  # Enable exponential penalties (default)
    min_balance_threshold=0.3  # Stricter threshold (default: 0.2)
)
```

## Key Changes Explained

### 1. Exponential Penalties

**Problem**: A 4-1-0 matchup (imbalance=0.3) had balance score of 0.4, which wasn't penalized enough.

**Solution**: Apply exponential penalty:
- 4-1-0 matchup: balance = 0.4 * 0.465 ≈ **0.19** (reduced by 53%)
- 5-0-0 matchup: balance = **0.0** (still 0, but aggregation will penalize more)

### 2. Geometric Mean

**Problem**: Arithmetic mean of [0.0, 0.5, 0.5, 0.5, 0.5] = 0.4, which doesn't reflect that one matchup is terrible.

**Solution**: Geometric mean = **0.062** (6.5x lower!), making configurations with any very imbalanced matchups much less likely to survive.

### 3. Minimum Threshold

**Problem**: Configurations could have one extremely imbalanced matchup (balance=0.0) and still have decent average balance.

**Solution**: If any matchup has balance < 0.2, apply additional penalty: `penalty = (min_balance / 0.2)^0.5`. This further reduces overall balance.

## Testing Recommendations

1. **Run a comparison**: Compare old vs. new balance calculation on the same configuration
2. **Monitor convergence**: Watch if balance improves over generations (should see higher balance scores)
3. **Analyze matchups**: Check if one-sided matchups (5-0-0, 0-5-0) decrease over time
4. **Validate results**: Test evolved configurations in real matches to ensure they're actually more balanced

## Trade-offs

**Pros**:
- ✅ Higher balance scores (target: 0.4-0.5+ vs. current 0.28)
- ✅ Fewer one-sided matchups
- ✅ More fair configurations for real gameplay
- ✅ Better exploration of balanced configurations

**Cons**:
- ⚠️ May take more generations to converge (stricter fitness landscape)
- ⚠️ Could converge to local optima if balance constraints are too strict
- ⚠️ Might reduce strategy diversity if balance becomes the only concern

**Mitigation**: Monitor diversity scores - if they drop too low, consider reducing alpha back to 0.45 or adjusting the penalty thresholds.

## Next Steps

1. ✅ Code implemented and tested for syntax errors
2. ⏳ Run on cluster and compare results with previous run
3. ⏳ Monitor balance scores over generations
4. ⏳ Analyze matchup distribution to verify improvements
5. ⏳ If needed, fine-tune penalty thresholds or weights

## Questions?

- See `BALANCE_IMPROVEMENTS.md` for detailed mathematical explanations
- See `cluster/README.md` for configuration options
- Check the code in `core/ga_working_evaluator.py` for implementation details

