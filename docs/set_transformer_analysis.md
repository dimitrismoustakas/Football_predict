# Set Transformer TODO

This file should only track unresolved work, live hypotheses, and worthwhile investigations.

---

## 1. Highest Priority: Loss Function

Main hypothesis:

- The remaining gap is mostly not simple calibration.
- The current Set Transformer still underperforms because plain cross-entropy does not shape its deviations from the market well enough.

Next experiments:

- Add GCE mixing.
- Add a small confidence penalty.
- Add a small Brier auxiliary term.

If those help:

- Add market target mixing with class weights.
- Add logit-delta regularization.
- Reintroduce gated residual / budget regularization only if the simpler loss additions show value.

Goal:

- Improve CV `log_loss` without pushing the model into low-quality anti-market bets.

---

## 2. Feature Gaps Worth Testing

Main hypothesis:

- Player-only inputs plus implied odds are still missing team context that the main model uses.

Priority additions:

- Elo / opponent-strength context
- rest / congestion features
- venue-conditioned team rolling stats
- promoted-team / league context

Cheap player-feature additions already available:

- `log_team_cumulative_minutes`
- `avg_minutes_r3`
- `minutes_last_match`
- `start_rate_r5`

Goal:

- Check whether the model is mainly limited by missing context rather than architecture.

---

## 3. Architecture Experiments

Only worth doing after loss and feature work unless a cheap change is available.

Candidates:

- increase position embedding dim from `4` to `8`
- multiple PMA seeds
- cross-team interaction / cross-attention blocks

Goal:

- Test whether the current team summary is too compressed or too independent across sides.

---

## 4. Low-Priority Cleanup

- Remove or properly use the ignored `player_rolling` argument in squad building.
- Keep the same-date `join_asof(..., allow_exact_matches=False)` edge case noted, but it is not an active priority.

---

## 5. Not A Priority Right Now

- Calibration-only fixes by themselves.
	- Post-hoc temperature scaling did not move held-out `log_loss` enough to matter.
- Gradient clipping.
	- Current gradient norms are well below a level that makes clipping urgent.
