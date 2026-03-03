# AI Agent Expansion

## Summary

Expanded the AI agent set from **6 to 8 AIs** for more comprehensive balance evaluation.

## Changes

### Before (6 AIs)
- workerRushAI
- lightRushAI
- coacAI
- naiveMCTSAI
- passiveAI
- randomBiasedAI

**Pairs**: 15 (6 choose 2)  
**Games per chromosome**: 75 (15 pairs × 5 games/pair)

### After (8 AIs)
- workerRushAI
- lightRushAI
- coacAI
- naiveMCTSAI
- **droplet** (NEW - competition AI)
- **mixedBot** (NEW - mixed strategy)
- passiveAI
- randomBiasedAI

**Pairs**: 28 (8 choose 2)  
**Games per chromosome**: 140 (28 pairs × 5 games/pair)  
**Increase**: 87% more matchups, 87% more evaluation time

## Why These AIs?

### Added AIs

1. **droplet** - Strong competition AI
   - Strategic gameplay
   - Different approach from coacAI
   - Tests configuration against competitive strategies

2. **mixedBot** - Mixed strategy bot
   - Adaptive gameplay
   - Combines multiple strategies
   - Tests configuration robustness across different playstyles

### Why Not Others?

**Excluded to keep evaluation time manageable:**
- `randomAI` - Similar to randomBiasedAI, minimal added value
- `guidedRojoA3N` - Mentioned as causing issues in codebase
- `rojo`, `izanagi`, `tiamat`, `mayari` - Competition AIs, but would push to 10+ AIs (45+ pairs = 225+ games/chromosome = too slow)
- `POLightRush`, `POWorkerRush`, `POHeavyRush`, `PORangedRush` - Partial observability versions, likely similar strategies

## Impact on Evaluation Time

### Current Configuration
- **8 AIs** = 28 pairs
- **5 games per pair** = 140 games per chromosome
- **25 population** = 3,500 games per generation
- **20 generations** = 70,000 total games

### Time Estimate (per generation)
- With 6 AIs: ~10-13 hours per generation (from previous run)
- With 8 AIs: ~18-25 hours per generation (87% increase)
- **Total run time**: ~2.4-3.5 days (was ~2.4 days, now may need checkpoint resume)

## Benefits

1. **Better Balance Assessment**: More matchups means better statistical coverage of balance
2. **More Diverse Strategies**: droplet and mixedBot bring different strategic approaches
3. **Stronger Competitors**: droplet is a strong competition AI, better stress test for configurations
4. **More Comprehensive**: 28 matchups vs. 15 gives better insight into balance across different AI pairs

## Trade-offs

**Pros**:
- ✅ 87% more matchups (15 → 28)
- ✅ Better balance evaluation
- ✅ More diverse strategy coverage
- ✅ Stronger competition AIs

**Cons**:
- ⚠️ 87% longer evaluation time (140 games vs. 75 games per chromosome)
- ⚠️ May require checkpoint resume even with 3-day limit
- ⚠️ Some AIs might fail (error handling continues, but reduces matchups)

## Configuration Options

### Use Default (8 AIs)
No changes needed - just run as normal:
```bash
sbatch scripts/GA\ and\ MAP-Elites/cluster/submit_ga.sbatch
```

### Custom AI Set
You can specify a custom AI set when creating the evaluator:
```python
custom_ais = [
    "workerRushAI",
    "lightRushAI",
    "coacAI",
    "naiveMCTSAI",
    "droplet",
    "rojo",  # Add more competition AIs if needed
    "passiveAI",
    "randomBiasedAI"
]

evaluator = WorkingGAEvaluator(
    ai_agents=custom_ais,
    ...
)
```

### Reduce to 6 AIs (Original)
If evaluation time is too long, you can reduce:
```python
original_ais = [
    "workerRushAI",
    "lightRushAI",
    "coacAI",
    "naiveMCTSAI",
    "passiveAI",
    "randomBiasedAI"
]

evaluator = WorkingGAEvaluator(
    ai_agents=original_ais,
    ...
)
```

## Testing Recommendations

1. **Test on cluster**: Run a short test (1-2 generations) to verify all 8 AIs work correctly
2. **Monitor errors**: Check logs for any AI failures (droplet, mixedBot might need Java dependencies)
3. **Compare results**: Compare balance scores with 6 vs. 8 AIs to see improvement
4. **Adjust if needed**: If evaluation too slow or AIs fail, reduce to 7 or back to 6

## Future Considerations

If evaluation time allows, could expand to **10 AIs** (45 pairs):
- Add: `rojo`, `tiamat` (competition AIs)
- Would give 225 games per chromosome (3x original)
- Better balance coverage but 2x slower than current 8 AI setup

## Files Modified

1. **`core/ga_working_evaluator.py`** - Updated baseline_ais list to 8 AIs
2. **`cluster/submit_ga.sbatch`** - Updated pair calculation (now dynamic: 28 pairs)

