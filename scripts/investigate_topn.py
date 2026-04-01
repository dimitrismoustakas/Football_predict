"""
Test how top-N squad size affects model performance and lineup overlap.

If the model works well even with top-25 (where overlap is guaranteed 100%
but there's more noise), then it's learning from player quality distribution,
not from "guessing" the lineup.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preprocessing.lineup_builder import NUM_FEATURES, PLAYER_FEATURE_COLS
from preprocessing.player_feature_engineering import (
    compute_player_rolling_features,
    load_all_player_data,
    prepare_player_data,
)
from preprocessing.lineup_builder import (
    build_projected_squads,
    assemble_squad_tensors,
)
from training.models.set_transformer import PlayerMatchModel
from training.train_player_model import (
    PlayerMatchDataset,
    prepare_match_data,
    evaluate,
    compute_metrics,
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


def compute_overlap(raw, squads, season="2526"):
    prepared = prepare_player_data(raw)
    test_actual = prepared.filter(pl.col("season") == season)
    test_squads = squads.filter(pl.col("season") == season)

    overlaps = []
    for (gid, tid), group_df in test_actual.group_by(["game_id", "team_id"]):
        actual_pids = set(group_df["player_id"].to_list())
        proj = test_squads.filter((pl.col("game_id") == gid) & (pl.col("team_id") == tid))
        if len(proj) == 0:
            continue
        proj_pids = set(proj["player_id"].to_list())
        overlaps.append(len(actual_pids & proj_pids) / len(actual_pids))
    return np.array(overlaps)


def main():
    set_seed(42)

    print("=" * 60)
    print("  TOP-N SENSITIVITY ANALYSIS")
    print("=" * 60)

    # Load data once
    raw = load_all_player_data()
    rolling = compute_player_rolling_features(raw)
    df = load_frame(Path("data/training/understat_df.parquet"))
    df = prepare_match_data(df)

    results = []

    for top_n in [11, 16, 22, 28]:
        print(f"\n{'='*60}")
        print(f"  TOP-N = {top_n}")
        print(f"{'='*60}")

        squads = build_projected_squads(raw, rolling, top_n=top_n)
        squad_tensors = assemble_squad_tensors(squads, df, max_players=top_n)

        game_ids = squad_tensors["game_ids"]
        game_id_order = pl.DataFrame({"game_id": game_ids, "_tensor_idx": range(len(game_ids))})
        aligned_df = df.join(game_id_order, on="game_id", how="inner").sort("_tensor_idx")

        # Overlap
        overlap = compute_overlap(raw, squads)
        print(f"  Lineup overlap: mean={overlap.mean():.3f}, perfect={((overlap==1.0).mean()):.1%}")

        test_season = get_sorted_seasons(aligned_df)[-1]
        folds = generate_rolling_cv_folds(aligned_df, n_folds=3, test_season=test_season)
        train_seasons, val_season = folds[-1]

        train_idx, val_idx, train_data, val_data = split_by_seasons(aligned_df, train_seasons, [val_season])
        _, test_idx, _, test_data = split_by_seasons(aligned_df, train_seasons + [val_season], [test_season])

        test_baseline = evaluate_implied_baseline(test_data)

        # Train model
        set_seed(42)
        train_ds = PlayerMatchDataset(squad_tensors, train_data["y"], train_data["implied"], train_data["raw_margin"], train_idx)
        val_ds = PlayerMatchDataset(squad_tensors, val_data["y"], val_data["implied"], val_data["raw_margin"], val_idx)
        test_ds = PlayerMatchDataset(squad_tensors, test_data["y"], test_data["implied"], test_data["raw_margin"], test_idx)

        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        model = PlayerMatchModel(
            input_dim=NUM_FEATURES, team_encoder_type="deep_sets",
            hidden_dim=64, team_output_dim=32, dropout=0.15, use_implied=True,
        )
        model = quick_train(model, train_loader, val_loader)
        _, test_probs = evaluate(model, test_loader, "cpu")
        metrics = compute_metrics(test_probs, test_data)

        print(f"  Test LL: {metrics['log_loss']:.5f}  (baseline: {test_baseline['log_loss']:.5f}, delta: {metrics['log_loss'] - test_baseline['log_loss']:+.5f})")
        print(f"  Test Acc: {metrics['accuracy']:.4f}  (baseline: {test_baseline['accuracy']:.4f})")
        print(f"  Test RPS: {metrics['rps']:.5f}  (baseline: {test_baseline['rps']:.5f})")

        results.append({
            "top_n": top_n,
            "overlap_mean": overlap.mean(),
            "overlap_perfect": (overlap == 1.0).mean(),
            "test_ll": metrics["log_loss"],
            "test_acc": metrics["accuracy"],
            "test_rps": metrics["rps"],
            "baseline_ll": test_baseline["log_loss"],
        })

    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Top-N':>6} | {'Overlap':>8} | {'Perfect%':>8} | {'Test LL':>8} | {'Delta LL':>9} | {'Test Acc':>8}")
    print("-" * 65)
    for r in results:
        print(f"{r['top_n']:>6} | {r['overlap_mean']:>8.3f} | {r['overlap_perfect']:>7.1%} | {r['test_ll']:>8.5f} | {r['test_ll'] - r['baseline_ll']:>+9.5f} | {r['test_acc']:>8.4f}")


if __name__ == "__main__":
    main()
