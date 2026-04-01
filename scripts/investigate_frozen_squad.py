"""
Frozen squad test: use the SAME top-16 squad for every match of the season.

If the model works equally well with a frozen squad (computed once at season
start), then it's learning from squad composition/quality, not from
match-to-match lineup changes.

If performance drops significantly with frozen squads, the model is exploiting
the fact that cumulative minutes shift reveals who played recently (and thus
who is likely to play next), which IS available pre-match but may overfit to
short-term availability patterns.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preprocessing.lineup_builder import (
    NUM_FEATURES,
    PLAYER_FEATURE_COLS,
    POSITION_TO_IDX,
    assemble_squad_tensors,
    build_projected_squads,
)
from preprocessing.player_feature_engineering import (
    compute_player_rolling_features,
    load_all_player_data,
    prepare_player_data,
)
from training.models.set_transformer import PlayerMatchModel
from training.train_player_model import (
    PlayerMatchDataset,
    evaluate,
    compute_metrics,
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


def build_frozen_squads(raw, rolling, top_n=16):
    """
    Build squads frozen at the START of each season.

    For each team-season, pick the top-N players by cumulative minutes
    from the PREVIOUS season only. Use the same 16 players for every
    match in that season.
    """
    prepared = prepare_player_data(raw)

    # Get seasons in order
    seasons = sorted(prepared["season"].unique().to_list())

    all_frozen = []

    for i, season in enumerate(seasons):
        if i == 0:
            # No prior season — skip (these matches will be dropped anyway due to min_history)
            continue

        prev_season = seasons[i - 1]

        # Get cumulative minutes from previous season for each (team, player)
        prev_data = prepared.filter(pl.col("season") == prev_season)
        prev_minutes = prev_data.group_by(["league", "team_id", "player_id"]).agg(
            pl.col("minutes").sum().alias("prev_season_minutes")
        )

        # For each team, pick top-N players
        prev_minutes = prev_minutes.sort(
            ["league", "team_id", "prev_season_minutes"], descending=[False, False, True]
        )
        prev_minutes = prev_minutes.with_columns(
            pl.col("prev_season_minutes")
            .rank(method="ordinal", descending=True)
            .over(["league", "team_id"])
            .alias("rank")
        )
        top_players = prev_minutes.filter(pl.col("rank") <= top_n)

        # Get all matches for this season
        season_matches = prepared.filter(pl.col("season") == season)
        season_match_keys = season_matches.select(
            ["league", "team_id", "game_id", "date"]
        ).unique()

        # Cross join: every match gets the same frozen squad
        frozen = season_match_keys.join(
            top_players.select(["league", "team_id", "player_id", "rank"]),
            on=["league", "team_id"],
            how="inner",
        )
        frozen = frozen.rename({"rank": "squad_rank"})

        # Attach rolling features (these STILL update per match — only squad SELECTION is frozen)
        frozen = frozen.join(
            rolling,
            on=["league", "team_id", "player_id", "game_id"],
            how="inner",
        )

        all_frozen.append(frozen)

    return pl.concat(all_frozen, how="diagonal")


def main():
    set_seed(42)

    print("=" * 60)
    print("  FROZEN SQUAD TEST")
    print("  Same 16 players for every match (chosen from prev season)")
    print("=" * 60)

    raw = load_all_player_data()
    rolling = compute_player_rolling_features(raw)
    df = load_frame(Path("data/training/understat_df.parquet"))
    df = prepare_match_data(df)

    # --- Dynamic squads (original) ---
    print("\n--- Dynamic squads (original approach) ---")
    dyn_squads = build_projected_squads(raw, rolling, top_n=16)
    dyn_tensors = assemble_squad_tensors(dyn_squads, df, max_players=16)

    game_ids = dyn_tensors["game_ids"]
    gid_order = pl.DataFrame({"game_id": game_ids, "_tensor_idx": range(len(game_ids))})
    aligned_df = df.join(gid_order, on="game_id", how="inner").sort("_tensor_idx")

    test_season = get_sorted_seasons(aligned_df)[-1]
    folds = generate_rolling_cv_folds(aligned_df, n_folds=3, test_season=test_season)
    train_seasons, val_season = folds[-1]

    train_idx, val_idx, train_data, val_data = split_by_seasons(aligned_df, train_seasons, [val_season])
    _, test_idx, _, test_data = split_by_seasons(aligned_df, train_seasons + [val_season], [test_season])

    test_baseline = evaluate_implied_baseline(test_data)
    print(f"Baseline LL: {test_baseline['log_loss']:.5f}")

    set_seed(42)
    dyn_model = PlayerMatchModel(input_dim=NUM_FEATURES, team_encoder_type="deep_sets",
                                  hidden_dim=64, team_output_dim=32, dropout=0.15, use_implied=True)
    dyn_train_ds = PlayerMatchDataset(dyn_tensors, train_data["y"], train_data["implied"], train_data["raw_margin"], train_idx)
    dyn_val_ds = PlayerMatchDataset(dyn_tensors, val_data["y"], val_data["implied"], val_data["raw_margin"], val_idx)
    dyn_test_ds = PlayerMatchDataset(dyn_tensors, test_data["y"], test_data["implied"], test_data["raw_margin"], test_idx)
    dyn_model = quick_train(
        dyn_model,
        DataLoader(dyn_train_ds, batch_size=256, shuffle=True),
        DataLoader(dyn_val_ds, batch_size=256, shuffle=False),
    )
    _, dyn_probs = evaluate(dyn_model, DataLoader(dyn_test_ds, batch_size=256), "cpu")
    dyn_metrics = compute_metrics(dyn_probs, test_data)
    print(f"Dynamic LL: {dyn_metrics['log_loss']:.5f}  (delta: {dyn_metrics['log_loss'] - test_baseline['log_loss']:+.5f})")

    # --- Frozen squads ---
    print("\n--- Frozen squads (same 16 for whole season) ---")
    frozen_squads = build_frozen_squads(raw, rolling, top_n=16)
    print(f"Frozen squad entries: {len(frozen_squads)}")

    frozen_tensors = assemble_squad_tensors(frozen_squads, df, max_players=16)
    frozen_gids = frozen_tensors["game_ids"]
    frozen_gid_order = pl.DataFrame({"game_id": frozen_gids, "_tensor_idx": range(len(frozen_gids))})
    frozen_aligned = df.join(frozen_gid_order, on="game_id", how="inner").sort("_tensor_idx")

    # Use same season splits
    f_train_idx, f_val_idx, f_train_data, f_val_data = split_by_seasons(frozen_aligned, train_seasons, [val_season])
    _, f_test_idx, _, f_test_data = split_by_seasons(frozen_aligned, train_seasons + [val_season], [test_season])

    f_test_baseline = evaluate_implied_baseline(f_test_data)

    set_seed(42)
    frozen_model = PlayerMatchModel(input_dim=NUM_FEATURES, team_encoder_type="deep_sets",
                                     hidden_dim=64, team_output_dim=32, dropout=0.15, use_implied=True)
    f_train_ds = PlayerMatchDataset(frozen_tensors, f_train_data["y"], f_train_data["implied"], f_train_data["raw_margin"], f_train_idx)
    f_val_ds = PlayerMatchDataset(frozen_tensors, f_val_data["y"], f_val_data["implied"], f_val_data["raw_margin"], f_val_idx)
    f_test_ds = PlayerMatchDataset(frozen_tensors, f_test_data["y"], f_test_data["implied"], f_test_data["raw_margin"], f_test_idx)
    frozen_model = quick_train(
        frozen_model,
        DataLoader(f_train_ds, batch_size=256, shuffle=True),
        DataLoader(f_val_ds, batch_size=256, shuffle=False),
    )
    _, frozen_probs = evaluate(frozen_model, DataLoader(f_test_ds, batch_size=256), "cpu")
    frozen_metrics = compute_metrics(frozen_probs, f_test_data)
    print(f"Frozen LL:  {frozen_metrics['log_loss']:.5f}  (delta: {frozen_metrics['log_loss'] - f_test_baseline['log_loss']:+.5f})")

    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"  Implied baseline:  {test_baseline['log_loss']:.5f}")
    print(f"  Dynamic squads:    {dyn_metrics['log_loss']:.5f}  ({dyn_metrics['log_loss'] - test_baseline['log_loss']:+.5f})")
    print(f"  Frozen squads:     {frozen_metrics['log_loss']:.5f}  ({frozen_metrics['log_loss'] - f_test_baseline['log_loss']:+.5f})")
    print()
    print("If frozen ~= dynamic: model learns SQUAD QUALITY (production-safe)")
    print("If frozen << dynamic: model exploits match-to-match availability changes")


if __name__ == "__main__":
    main()
