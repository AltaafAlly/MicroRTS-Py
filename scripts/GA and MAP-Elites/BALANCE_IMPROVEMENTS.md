# Balance Improvements for GA Evolution

## Problem Analysis

From the cluster run analysis (`cluster_long_run_1767718702`), we observed:
- **Low balance score**: 0.28 (28%) - far from ideal
- **Many one-sided matchups**: Numerous 5-0-0 or 0-5-0 results
- **coacAI dominance**: Consistently winning against multiple AIs
- **Averaging issue**: Balanced and imbalanced matchups were averaged, allowing poor individual matchups to persist

## Root Causes

1. **Linear balance penalty**: Original calculation used `balance = 1.0 - abs(win_ratio - 0.5) * 2`
   - A 5-0-0 matchup (0% balance) has the same impact as a 3-2-0 matchup (~50% balance) in terms of averaging
   - No extra penalty for extremely imbalanced matchups

2. **Arithmetic mean aggregation**: Simple average of balance scores
   - One very imbalanced matchup doesn't significantly hurt the overall score
   - Example: [0.0, 0.5, 0.5, 0.5, 0.5] → average = 0.4 (but has one terrible matchup)

3. **Balanced fitness weights**: Balance had only 40% weight (alpha=0.4)
   - Duration and diversity had 30% each
   - Not enough emphasis on achieving balance

## Implemented Solutions

### 1. Increased Balance Weight

**Change**: Default `alpha` increased from 0.4 to 0.5

```python
# Before
fitness_alpha: float = 0.4  # Balance weight
fitness_beta: float = 0.3   # Duration weight
fitness_gamma: float = 0.3  # Strategy diversity weight

# After
fitness_alpha: float = 0.5  # Balance weight (50% weight)
fitness_beta: float = 0.25  # Duration weight (25% weight)
fitness_gamma: float = 0.25 # Strategy diversity weight (25% weight)
```

**Impact**: Balance now has 50% influence on overall fitness, making it more important for survival.

### 2. Exponential Penalties for Imbalanced Matchups

**Change**: Added exponential penalty for very imbalanced matchups

```python
# Original (linear)
balance = 1.0 - abs(win_ratio - 0.5) * 2

# New (exponential for imbalanced)
imbalance = abs(win_ratio - 0.5)
base_balance = 1.0 - imbalance * 2

if imbalance > 0.35:  # More than 70-30 split
    # Exponential penalty: (imbalance/0.5)^2
    penalty_factor = (imbalance / 0.5) ** 2
    balance = base_balance * penalty_factor
elif imbalance > 0.25:  # Between 60-40 and 70-30
    # Quadratic penalty: (imbalance/0.5)^1.5
    penalty_factor = (imbalance / 0.5) ** 1.5
    balance = base_balance * penalty_factor
```

**Example Impact**:
- **5-0-0 matchup** (imbalance=0.5):
  - Original: balance = 0.0
  - New: balance = 0.0 * (0.5/0.5)^2 = 0.0 * 1.0 = 0.0 (still 0, but will be penalized more in aggregation)
  
- **4-1-0 matchup** (imbalance=0.4, win_ratio=0.8):
  - Original: balance = 1.0 - 0.4*2 = 0.2
  - New: balance = 0.2 * (0.4/0.5)^2 = 0.2 * 0.64 = 0.128 (reduced by 36%)
  
- **3-2-0 matchup** (imbalance=0.2, win_ratio=0.6):
  - Original: balance = 1.0 - 0.2*2 = 0.6
  - New: balance = 0.6 * (0.2/0.5)^1.5 = 0.6 * 0.36 = 0.216 (reduced by 64% - maybe too harsh?)

Wait, let me recalculate the 3-2-0 case more carefully:
- win_ratio = 3/5 = 0.6
- imbalance = |0.6 - 0.5| = 0.1 (not 0.2!)
- base_balance = 1.0 - 0.1*2 = 0.8
- Since 0.1 < 0.25, no penalty applied
- balance = 0.8 (actually better than before, which makes sense)

**Better example - 4-1-0**:
- win_ratio = 4/5 = 0.8
- imbalance = |0.8 - 0.5| = 0.3
- base_balance = 1.0 - 0.3*2 = 0.4
- Since 0.25 < 0.3 < 0.35, apply quadratic penalty
- penalty_factor = (0.3/0.5)^1.5 = 0.6^1.5 ≈ 0.465
- balance = 0.4 * 0.465 ≈ 0.186 (reduced from 0.4)

### 3. Geometric Mean Aggregation

**Change**: Use geometric mean instead of arithmetic mean for balance scores

```python
# Before (arithmetic mean)
balance = sum(balance_scores) / len(balance_scores)

# After (geometric mean)
import math
balanced_scores = [max(0.001, score) for score in balance_scores]  # Avoid log(0)
log_sum = sum(math.log(score) for score in balanced_scores)
geometric_mean = math.exp(log_sum / len(balanced_scores))
```

**Impact**: Geometric mean is more sensitive to very low scores. A single 0.0 balance score will significantly lower the overall balance.

**Example**:
- Scores: [0.0, 0.5, 0.5, 0.5, 0.5]
- Arithmetic mean: (0.0 + 0.5*4) / 5 = 0.4
- Geometric mean: exp((log(0.001) + log(0.5)*4) / 5) ≈ exp(-2.77) ≈ 0.062

The geometric mean is **6.5x lower** than the arithmetic mean, making it much harder for configurations with imbalanced matchups to survive.

### 4. Minimum Balance Threshold

**Change**: Added configurable minimum balance threshold with penalty

```python
min_balance_threshold: float = 0.2  # Minimum acceptable balance per matchup

# Check if any matchup is below threshold
min_balance = min(balance_scores)
if min_balance < self.min_balance_threshold:
    # Apply penalty: reduce overall balance
    penalty = (min_balance / self.min_balance_threshold) ** 0.5  # Square root penalty
    balance = geometric_mean * penalty
```

**Impact**: If any matchup has balance < 0.2, the overall balance is further reduced. This prevents configurations with extremely imbalanced matchups from surviving.

## Expected Results

### Before Improvements
- Balance score: ~0.28 (28%)
- Many matchups: 5-0-0, 0-5-0 (completely one-sided)
- Some matchups: 4-1-0, 1-4-0 (very imbalanced)
- Few matchups: 3-2-0, 2-3-0 (reasonably balanced)

### After Improvements
- **Target balance score**: 0.4-0.5+ (40-50%)
- **Fewer one-sided matchups**: Expect no or very few 5-0-0, 0-5-0 results
- **More balanced matchups**: Most matchups should be within 60-40 split or better
- **Fewer extreme imbalances**: 4-1-0 and 1-4-0 matchups should be rare

### Trade-offs

**Pros**:
- Higher overall balance scores
- More fair matchups across all AI pairs
- Better game configurations for real gameplay

**Cons**:
- May take more generations to converge (stricter fitness landscape)
- Might converge to local optima that favor certain AI strategies
- Could potentially reduce strategy diversity if balance constraints are too strict

## Configuration Options

All improvements are enabled by default. To disable strict balance:

```python
evaluator = WorkingGAEvaluator(
    alpha=0.5,  # Increased balance weight
    beta=0.25,
    gamma=0.25,
    use_strict_balance=False,  # Disable exponential penalties (uses linear)
    min_balance_threshold=0.2  # Minimum acceptable balance
)
```

## Testing Recommendations

1. **Run comparison**: Run the same GA configuration with old vs. new balance calculation
2. **Monitor convergence**: Watch if balance improves over generations
3. **Check matchup distribution**: Analyze if one-sided matchups decrease
4. **Validate gameplay**: Test evolved configurations in real matches to ensure they're actually more balanced

## Future Improvements

1. **Adaptive weights**: Adjust balance weight dynamically based on generation (start low, increase over time)
2. **AI-specific balance**: Consider that some AI pairs may naturally be imbalanced - weight balance by AI strength
3. **Matchup prioritization**: Give more weight to balance in common/relevant matchups
4. **Pareto optimization**: Use multi-objective optimization to explore balance vs. diversity trade-offs

