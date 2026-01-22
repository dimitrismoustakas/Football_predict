# Residual Analysis by Market Implied Probability Deciles

**Date**: January 22, 2026  
**Models Analyzed**:
- Result prediction (Home/Draw/Away) - `result_arch_tuned.pt`
- Over/Under 2.5 goals - `over_under_arch_tuned.pt`

## Summary

This analysis examines how our models perform relative to bookmaker odds across different probability ranges. We bin test set predictions by market implied probability deciles and compute:

1. **Residuals**: `model_prob - market_prob` (for the true outcome)
2. **Log Loss Delta**: `model_log_loss - market_log_loss` (positive = model worse)

---

## Result Model (Match Outcome) - MULTICLASS

### Key Findings

**Overall Pattern**: Strong asymmetric performance
- **Low market confidence (D1-D4)**: Model significantly OUTPERFORMS market
  - Mean Δ Log Loss: -0.576 to -0.118 (GREEN zones)
  - Model assigns 4-14% higher probability to true outcomes than market
  
- **High market confidence (D6-D10)**: Model UNDERPERFORMS market
  - Mean Δ Log Loss: +0.225 to +0.676 (RED zones)
  - Model assigns 8-36% lower probability to true outcomes than market

### Detailed Results

| Decile | Market Prob Range | N   | Mean Residual | ΔLog Loss | Interpretation |
|--------|------------------|-----|---------------|-----------|----------------|
| D1     | 0.086-0.225     | 56  | +0.142       | **-0.576** | Model is contrarian and correct on underdogs |
| D2     | 0.225-0.263     | 56  | +0.077       | **-0.260** | Model adds value over market |
| D3     | 0.263-0.288     | 56  | +0.049       | **-0.147** | Model adds value over market |
| D4     | 0.288-0.316     | 56  | +0.042       | **-0.118** | Model adds value over market |
| D5     | 0.316-0.380     | 56  | -0.010       | +0.052     | Near-neutral zone |
| D6     | 0.380-0.438     | 56  | -0.077       | +0.225     | Model underestimates favorites |
| D7     | 0.438-0.511     | 55  | -0.121       | +0.311     | Model underestimates favorites |
| D8     | 0.511-0.585     | 56  | -0.205       | +0.491     | Model underestimates favorites |
| D9     | 0.585-0.666     | 55  | -0.266       | +0.578     | Model underestimates favorites |
| D10    | 0.666-0.861     | 59  | -0.362       | +0.676     | Model underestimates heavy favorites |

### Interpretation

1. **Market inefficiency detected**: Bookmakers appear to underestimate underdogs (or overestimate favorites)
2. **Model captures this**: Model is more contrarian on low-probability outcomes and correct on average
3. **Calibration issue**: Model appears under-confident on high-probability outcomes
4. **Betting implications**: 
   - Potential value betting on underdogs (D1-D4)
   - Avoid betting against heavy favorites (D8-D10)

---

## Over/Under 2.5 Goals Model - BINARY

### Key Findings

**Overall Pattern**: Moderate asymmetric performance
- **Low-mid market confidence (D1-D7)**: Mixed performance, slight underperformance
  - Mean Δ Log Loss: -0.016 to +0.103
  - Model generally assigns 1-14% higher probability to Over than market
  
- **High market confidence (D8-D10)**: Model UNDERPERFORMS market
  - Mean Δ Log Loss: +0.044 to +0.085
  - Model assigns 4-14% lower probability to true outcomes

### Detailed Results

| Decile | Market Prob Range | N   | Mean Residual | ΔLog Loss | Interpretation |
|--------|------------------|-----|---------------|-----------|----------------|
| D1     | 0.281-0.413     | 51  | +0.143       | +0.007     | Neutral |
| D2     | 0.413-0.432     | 26  | +0.097       | +0.081     | Slight underperformance |
| D3     | 0.432-0.452     | 46  | +0.081       | +0.103     | Slight underperformance |
| D4     | 0.452-0.487     | 85  | +0.068       | +0.062     | Slight underperformance |
| D5     | 0.487-0.513     | 54  | +0.050       | +0.039     | Near-neutral zone |
| D6     | 0.513-0.548     | 64  | -0.009       | **-0.016** | Slight outperformance |
| D7     | 0.548-0.568     | 54  | +0.019       | **-0.011** | Slight outperformance |
| D8     | 0.568-0.603     | 66  | -0.038       | +0.044     | Underperformance |
| D9     | 0.603-0.637     | 46  | -0.109       | +0.068     | Underperformance |
| D10    | 0.637-0.781     | 69  | -0.141       | +0.085     | Underperformance |

### Interpretation

1. **Smaller edge than result model**: Over/Under model shows less clear advantage
2. **Sweet spot at D6-D7**: Model performs slightly better than market near 50-55% probability
3. **Calibration issue**: Similar under-confidence pattern on high-probability outcomes
4. **Betting implications**:
   - Limited value overall (most bins show small positive Δ log loss)
   - Potential small edge in D6-D7 range (moderate confidence bets)

---

## General Insights

### Common Patterns Across Both Models

1. **Systematic under-confidence on favorites**: Both models assign lower probabilities than market to high-confidence outcomes
2. **Better calibration on underdogs/uncertainty**: Models perform better when market confidence is lower
3. **Possible causes**:
   - Market efficiency higher for clear favorites (more liquidity, better info)
   - Model training may be regularized/conservative
   - Market odds may include favorite-longshot bias

### Recommendations

1. **Betting strategy**:
   - Focus on low-market-confidence bets for result model (D1-D4)
   - Be cautious betting against heavy favorites
   - Over/Under model shows limited edge overall

2. **Model improvements**:
   - Investigate calibration on high-probability outcomes
   - Consider temperature scaling or Platt scaling for better calibration
   - Explore whether favorite-longshot bias in market can be systematically exploited

3. **Further analysis**:
   - Examine residuals by league (some leagues may be more efficient)
   - Check if pattern holds across different seasons
   - Analyze profitability using Kelly criterion with estimated edge

---

## Files Generated

- Analysis script: `training/analyze_residuals_by_decile.py`
- Result model plot: `data/plots/residuals_by_decile_result.png`
- Over/Under model plot: `data/plots/residuals_by_decile_over_under.png`

## How to Reproduce

```bash
# Result model
uv run training/analyze_residuals_by_decile.py result

# Over/Under model
uv run training/analyze_residuals_by_decile.py over_under

# Custom number of bins (default 10)
uv run training/analyze_residuals_by_decile.py result 20
```
