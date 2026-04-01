"""
Permutation test to check if player-level features carry genuine signal
or if the model is learning something it shouldn't.

Test: Shuffle player features ACROSS matches (destroying per-match identity)
while keeping the overall feature distribution the same.

If shuffled model ~= normal model: signal comes from team-level distribution
If shuffled model >> normal model: genuine match-specific player signal
If shuffled model ~= baseline: features add no signal when randomized
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preprocessing.lineup_builder import NUM_FEATURES
from training.models.set_transformer import PlayerMatchModel
from training.train_player_model import (
    PlayerMatchDataset,
    build_squad_data,
    compute_metrics,
    evaluate,
    prepare_match_data,
    set_seed,
    split_by_seasons,
)
from training.train_utils import (
    evaluate_implied_baseline,
    generate_rolling_cv_folds,
    get_sorted_seasons,
    load_frame,
)


def quick_train(model, train_loader, val_loader, max_epochs=35, patience=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            inputs = [t.to("cpu") for t in batch]
            home_feat, home_pos, home_mask, away_feat, away_pos, away_mask, implied, labels, _ = inputs
            optimizer.zero_grad()
            logits = model(home_feat, home_pos, home_mask, away_feat, away_pos, away_mask, implied)
            F.cross_entropy(logits, labels).backward()
            optimizer.step()
        vl, _ = evaluate(model, val_loader, "cpu")
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model


def main():
    set_seed(42)

    print("=" * 60)
    print("  LEAK INVESTIGATION: Permutation Test")
    print("=" * 60)

    df = load_frame(Path("data/training/understat_df.parquet"))
    df = prepare_match_data(df)
    squad_tensors, aligned_df = build_squad_data(df, top_n=16)

    test_season = get_sorted_seasons(aligned_df)[-1]
    folds = generate_rolling_cv_folds(aligned_df, n_folds=3, test_season=test_season)
    train_seasons, val_season = folds[-1]

    train_idx, val_idx, train_data, val_data = split_by_seasons(aligned_df, train_seasons, [val_season])
    _, test_idx, _, test_data = split_by_seasons(aligned_df, train_seasons + [val_season], [test_season])

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    test_baseline = evaluate_implied_baseline(test_data)
    print(f"\nImplied baseline test LL: {test_baseline['log_loss']:.5f}")

    # --- Normal model ---
    print("\n--- Training NORMAL model ---")
    set_seed(42)
    train_ds = PlayerMatchDataset(squad_tensors, train_data["y"], train_data["implied"], train_data["raw_margin"], train_idx)
    val_ds = PlayerMatchDataset(squad_tensors, val_data["y"], val_data["implied"], val_data["raw_margin"], val_idx)
    test_ds = PlayerMatchDataset(squad_tensors, test_data["y"], test_data["implied"], test_data["raw_margin"], test_idx)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = PlayerMatchModel(input_dim=NUM_FEATURES, team_encoder_type="deep_sets",
                             hidden_dim=64, team_output_dim=32, dropout=0.15, use_implied=True)
    model = quick_train(model, train_loader, val_loader)
    _, normal_probs = evaluate(model, test_loader, "cpu")
    normal_metrics = compute_metrics(normal_probs, test_data)
    print(f"  Normal test LL: {normal_metrics['log_loss']:.5f}  Acc: {normal_metrics['accuracy']:.4f}")

    # --- Shuffled model: randomize features across matches ---
    print("\n--- Training SHUFFLED model (features randomized across matches) ---")
    shuffled = {k: (v.copy() if isinstance(v, np.ndarray) else list(v)) for k, v in squad_tensors.items()}
    rng = np.random.RandomState(42)

    for side in ["home", "away"]:
        feats = shuffled[f"{side}_players"]  # (N, 16, D)
        mask = shuffled[f"{side}_mask"]       # (N, 16)
        # Shuffle each feature dimension independently across all valid player slots
        for d in range(feats.shape[2]):
            col = feats[:, :, d].copy()
            valid_vals = col[mask].copy()
            rng.shuffle(valid_vals)
            col[mask] = valid_vals
            feats[:, :, d] = col

    set_seed(42)
    shuf_train_ds = PlayerMatchDataset(shuffled, train_data["y"], train_data["implied"], train_data["raw_margin"], train_idx)
    shuf_val_ds = PlayerMatchDataset(shuffled, val_data["y"], val_data["implied"], val_data["raw_margin"], val_idx)
    shuf_test_ds = PlayerMatchDataset(shuffled, test_data["y"], test_data["implied"], test_data["raw_margin"], test_idx)

    shuf_train_loader = DataLoader(shuf_train_ds, batch_size=256, shuffle=True)
    shuf_val_loader = DataLoader(shuf_val_ds, batch_size=256, shuffle=False)
    shuf_test_loader = DataLoader(shuf_test_ds, batch_size=256, shuffle=False)

    model2 = PlayerMatchModel(input_dim=NUM_FEATURES, team_encoder_type="deep_sets",
                              hidden_dim=64, team_output_dim=32, dropout=0.15, use_implied=True)
    model2 = quick_train(model2, shuf_train_loader, shuf_val_loader)
    _, shuf_probs = evaluate(model2, shuf_test_loader, "cpu")
    shuf_metrics = compute_metrics(shuf_probs, test_data)
    print(f"  Shuffled test LL: {shuf_metrics['log_loss']:.5f}  Acc: {shuf_metrics['accuracy']:.4f}")

    # --- Labels-shuffled model: same features, random labels ---
    print("\n--- Training LABEL-SHUFFLED model (same features, random labels) ---")
    set_seed(42)
    fake_train_y = train_data["y"].copy()
    rng.shuffle(fake_train_y)

    fake_train_ds = PlayerMatchDataset(squad_tensors, fake_train_y, train_data["implied"], train_data["raw_margin"], train_idx)
    fake_train_loader = DataLoader(fake_train_ds, batch_size=256, shuffle=True)

    model3 = PlayerMatchModel(input_dim=NUM_FEATURES, team_encoder_type="deep_sets",
                              hidden_dim=64, team_output_dim=32, dropout=0.15, use_implied=True)
    model3 = quick_train(model3, fake_train_loader, val_loader)
    _, fake_probs = evaluate(model3, test_loader, "cpu")
    fake_metrics = compute_metrics(fake_probs, test_data)
    print(f"  Label-shuffled test LL: {fake_metrics['log_loss']:.5f}  Acc: {fake_metrics['accuracy']:.4f}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY:")
    print(f"  Implied baseline LL:   {test_baseline['log_loss']:.5f}")
    print(f"  Normal model LL:       {normal_metrics['log_loss']:.5f}  (delta: {normal_metrics['log_loss'] - test_baseline['log_loss']:+.5f})")
    print(f"  Shuffled features LL:  {shuf_metrics['log_loss']:.5f}  (delta: {shuf_metrics['log_loss'] - test_baseline['log_loss']:+.5f})")
    print(f"  Shuffled labels LL:    {fake_metrics['log_loss']:.5f}  (delta: {fake_metrics['log_loss'] - test_baseline['log_loss']:+.5f})")
    print()
    print("INTERPRETATION:")
    gap_normal = test_baseline["log_loss"] - normal_metrics["log_loss"]
    gap_shuffled = test_baseline["log_loss"] - shuf_metrics["log_loss"]
    if gap_shuffled > 0.5 * gap_normal:
        print("  WARNING: Shuffled model retains most of the gain.")
        print("  Signal likely comes from team-level distribution, not player-specific info.")
        print("  Or: model is mostly using implied probs and features add marginal value.")
    else:
        print("  Shuffled model loses most gain -> player-specific features carry real signal.")
    print(f"\n  Signal from player features: {gap_normal - gap_shuffled:.5f} LL")
    print(f"  Signal retained after shuffle: {gap_shuffled:.5f} LL")


if __name__ == "__main__":
    main()
