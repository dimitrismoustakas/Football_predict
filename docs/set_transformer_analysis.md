# Set Transformer Model Analysis

Deep-dive analysis of the player-level Set Transformer model vs the production (main) model.
Covers preprocessing correctness, implementation review, gaps, and improvement paths.

---

## 1. Preprocessing & Data Leakage Audit

### Verdict: No leakage found

The anti-leakage design is solid. Two independent code paths compute player features:

| Path | Used by | Leak prevention |
|------|---------|-----------------|
| `compute_player_rolling_features` | standalone analysis | `shift(1)` before every rolling op |
| `_compute_player_state_history` (lineup_builder) | squad tensor assembly | No shift, but `join_asof(strategy="backward", allow_exact_matches=False)` achieves the same result |

Both yield identical pre-match values. Verified for:

- **Per-90 rolling stats**: shift+rolling vs rolling+asof — equivalent
- **`season_yellow_cards`**: `shift(1).cum_sum()` vs `cum_sum()` picked up from previous appearance via asof — equivalent
- **`red_card_prev_game`**: `shift(1)` check vs current-match value picked up from previous appearance — equivalent
- **Cumulative minutes for squad ranking**: excludes current match in both paths

### Minor edge case

`join_asof(allow_exact_matches=False)` joins on `date`. If a player had two appearances on the **same date** (extremely rare — rescheduled double-header), the second match would fail to pick up state from the first. Negligible in practice.

### Note: `player_rolling` parameter is unused

In `build_projected_squads` (line 299): `_ = player_rolling`. The function recomputes features internally via `_compute_player_state_history`. The call to `compute_player_rolling_features` in `train_player_model.py:423` is wasted work.

---

## 2. Set Transformer Implementation Review

### Architecture (correct)

- **Weight sharing**: Single `team_encoder` for both home/away — correct for permutation invariance
- **Attention masking**: `attn_mask = ~mask` correctly inverts (True = ignore padding)
- **PMA seeds**: `randn * 0.02` — standard small init
- **Post-norm transformer**: `norm(x + attn) -> norm(h + ff(h))` — standard

### Issues found

#### A. No gradient clipping during training (low-medium severity)

`train_one_epoch` has no gradient clipping, but `evaluate` clamps logits to `[-20, 20]`. After adding StandardScaler normalization, gradient norms are well-behaved (mean ~0.21, max spikes ~1.75), so aggressive clipping is not needed. A mild safety clip at 5.0 could be added but is not urgent.

#### B. No feature normalization (high impact) — DONE

Player features were fed raw. Features like `season_yellow_cards` (0-15) and `xg_per90_r10` (0-1) were on very different scales. This was starving the team encoder of gradient signal.

**Fix applied**: Per-fold `StandardScaler` fitted on training match player features, applied to all splits. See `fit_player_scaler` / `apply_player_scaler` / `scale_squad_tensors` in `train_player_model.py`.

**Impact measured** (10-epoch comparison on epoch-selection fold):
- team_encoder gradient norm: 0.055 (raw) -> 0.133 (scaled) — **2.4x more signal**
- Epoch 1 train loss: 1.068 (raw) -> 1.010 (scaled) — much faster convergence
- corr_with_implied: 0.97 (raw) -> 0.91 (scaled) — model learns to deviate more from market

The higher deviation from market causes higher CV log_loss with plain cross-entropy (0.983 vs 0.971), because the model makes more "bets" against the market without loss-function guidance. This is expected and motivates the loss function improvements below.

#### C. Position embedding dimension possibly too small

4 dimensions for 17 positions is compressed. Consider 8.

#### D. `shuffle_squad_features` defined but never used

Appears intended as a sanity check / ablation baseline. Could also serve as data augmentation.

---

## 3. Data & Features Left on the Table

### A. No team-level context beyond implied probabilities

The production model uses ~46 features. The set transformer sees none of:

| Feature category | Production model | Set transformer |
|-----------------|-----------------|-----------------|
| Elo ratings (diff, mean, rolling) | yes | no |
| Days since last match / fixture congestion | yes | no |
| Home/away venue rolling stats | yes | no |
| Opponent quality (rolling opponent Elo) | yes | no |
| Promoted team flags | yes | no |
| League embeddings | yes | no |
| Team rolling xG/npxG for/against | yes | no |
| Deep defensive play stats | yes | no |

Player stats alone cannot capture venue advantage, schedule fatigue, or team strength context.

### B. Auxiliary player features already computed but excluded

These are in `PLAYER_AUX_STATE_COLS` but not in `PLAYER_FEATURE_COLS`:

| Feature | Signal | Why useful |
|---------|--------|-----------|
| `log_team_cumulative_minutes` | Player tenure at club | Familiarity with team system |
| `avg_minutes_r3` | Very recent form | More responsive than r10 |
| `minutes_last_match` | Latest game time | Fatigue / match fitness |
| `start_rate_r5` | Starting likelihood | Squad hierarchy signal |

These are already computed — just not wired into the feature tensor.

### C. No cross-team player interactions

Each team is encoded independently, then concatenated as `[home, away, diff]`. There's no mechanism for the model to learn matchup-specific interactions (e.g., a fast striker vs a slow centre-back).

### D. Single PMA seed

One learnable seed compresses the entire team into a single vector. Multiple seeds (e.g., 2-4) could capture richer team structure (attack quality, defensive solidity, depth).

---

## 4. Loss Function — The Biggest Gap

### Current set transformer loss

```python
loss = F.cross_entropy(logits, labels)
```

Plain vanilla cross-entropy. No class weighting, no market awareness, no regularization.

### Production model loss (~150 lines, 8+ components)

| Component | Weight/Config | Purpose |
|-----------|--------------|---------|
| Market target mixing | mix=0.01125 | Soft labels blended with market probs |
| Class-specific weights | draw: 0.03x, away: 1.9x | Compensate for class difficulty |
| Market surprise scaling | scale=1.7, band mode | Upweight unexpected outcomes |
| GCE loss | 8.95%, q=1.25 | Noise-robust alternative to CE |
| Confidence penalty | 0.8% | Prevent overconfident predictions |
| Brier auxiliary | 3% | L2 calibration signal |
| Symmetric CE | 0.1% | Bidirectional KL divergence |
| Gate mean regularization | weight=0.161, budget=13.5% | Keep gate activation controlled |
| Repulsion from market | lambda=2.3 | Encourage meaningful deviation |
| Logit delta regularization | lambda=0.0325 | Prevent excessive deviation |
| Entropy curriculum | center_only, strength=0.15 | Focus on medium-entropy matches |

This is the single largest performance gap between the two models.

---

## 5. Epoch Selection — Diagnostic Results

### Problem

The val loss valley is very flat (~0.003 range over 10+ epochs). Picking the raw minimum epoch is chasing noise.

### Cross-fold stability (30 epochs each, no early stopping, with StandardScaler)

| Fold | n_train | Raw best | Smooth-3 best | Smooth-5 best |
|------|---------|----------|---------------|---------------|
| 1 (->2122) | 10,954 | 21 | 20 | 19 |
| 2 (->2223) | 12,532 | 19 | 20 | 21 |
| 3 (->2324) | 14,112 | 22 | 11 | 13 |
| 4 (->2425) | 15,622 | 11 | 20 | 21 |

Raw best epoch range: **11** (essentially random within the flat valley).

### Fix applied: Smoothed epoch selection

`_smoothed_best_epoch()` uses a rolling-5-epoch average of the val loss curve. Early stopping patience is computed relative to the smoothed best, not the raw best. This stabilizes the selected epoch count and avoids chasing noise.

Verified: on the epoch-selection fold, smoothed picks epoch 18 vs raw 16 — both sit in the flat valley, but the smoothed estimate is more robust to noise.

---

## 6. Experiment Results

### Prior runs (pre-normalization, from set_transformer_runs.tsv)

| Run | Description | CV log_loss | Delta | Best epoch | Status |
|-----|-------------|-------------|-------|------------|--------|
| 1 | baseline | 0.973022 | -- | 12 | keep |
| 2 | market4 classscale budget010 | 0.971651 | +0.001371 | 12 | keep |
| 3 | aux010 market4 classscale budget020 | 0.971872 | -0.000221 | 2 | discard |
| 4 | aux010 market4 classscale budget010 min8 | 0.971096 | +0.000555 | 18 | keep (best) |

Best run improves ~0.002 over baseline via gated residual head with market features. The model correlates very highly with implied (0.997-0.999), suggesting it's mostly learning to copy the market with minimal deviation.

### StandardScaler normalization run

| Metric | Best prior (no scaler) | With StandardScaler |
|--------|----------------------|---------------------|
| CV log_loss | 0.9711 | 0.9832 |
| corr_with_implied | 0.998 | 0.841 |
| team_encoder grad norm | 0.055 | 0.133 |

CV log_loss is worse because the model now deviates more from the market (corr dropped from 0.998 to 0.841) but plain cross-entropy can't guide those deviations. The encoder is finally learning (2.4x gradient improvement), but the loss function needs upgrading to channel that learning productively.

---

## 7. Remaining Work — TODO

### Priority 1: Hyperparameter sweep (run on GPU)

Training dynamics need tuning now that features are normalized. Run the sweep in `train_player_model.py` on the epoch-selection fold (35 epochs each, no early stopping). Configs to test:

**LR sweep** (cosine schedule, wd=1e-4):
- lr=5e-4, 1e-3 (current), 2e-3, 3e-3

**Weight decay sweep** (cosine, lr=1e-3):
- wd=5e-5, 1e-4 (current), 3e-4

**Schedule variants** (lr=1e-3, wd=1e-4):
- Cosine (current)
- Cosine + 10% linear warmup (use `SequentialLR` with `LinearLR` + `CosineAnnealingLR`)
- OneCycleLR (max_lr=3e-3, pct_start=0.3)
- OneCycleLR (max_lr=5e-3, pct_start=0.3)
- CosineAnnealingWarmRestarts (T_0=10, T_mult=2)

**Schedule + higher LR**:
- Cosine + 10% warmup at lr=2e-3

Compare on: smoothed best val loss, train-val gap at best epoch, val@15, val@20, val@25.

### Priority 2: Loss function improvements

After finding optimal training dynamics, add loss components incrementally:

**Step 1** — Start with these (low-risk, well-understood):
- GCE mixing (8-10%, q=1.25) — noise-robust, helps with label noise in football
- Confidence penalty (0.5-1%) — prevents overconfidence
- Brier auxiliary (2-3%) — calibration signal

**Step 2** — Market-aware components (need gated_residual head):
- Market target mixing with class weights
- Gate mean regularization
- Logit delta regularization

**Step 3** — Full production loss port (if step 2 helps):
- Market surprise scaling
- Entropy curriculum
- Repulsion term

### Priority 3: Additional features

- Include the 4 auxiliary player features (trivial change to `PLAYER_FEATURE_COLS`)
- Add team-level context to prediction head (Elo diff, rest days, etc.)
- Increase position embedding dim to 8

### Priority 4: Architecture experiments

- Multiple PMA seeds (K=2-4)
- Cross-attention between teams
- Player dropout augmentation (randomly mask 1-2 players, p=0.1)
