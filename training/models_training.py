#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    f1_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
from joblib import dump

DEFAULT_PARQUET = Path("data/training/understat_df.parquet")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_frame(parquet_path: Path) -> pl.DataFrame:
    lf = pl.scan_parquet(str(parquet_path))
    df = lf.collect()
    return df

def select_feature_columns(df: pl.DataFrame) -> list[str]:
    cols = df.columns
    feat_cols = [
        c
        for c in cols
        if c.startswith("ovr__")
        and "__r5" in c
        and not "__sum__" in c
        and not c.startswith("ovr__games__")
        and (c.endswith("__h") or c.endswith("__a"))
    ]
    
    # Add Elo features if present
    elo_cols = ["elo_diff", "elo_sum", "elo_mean"]
    for ec in elo_cols:
        if ec in cols:
            feat_cols.append(ec)
            
    return sorted(feat_cols)

def filter_min_history(df: pl.DataFrame) -> pl.DataFrame:
    need_cols = ["ovr__games__r5__h", "ovr__games__r5__a"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for history filter: {missing}")
    return df.filter((pl.col("ovr__games__r5__h") >= 5) & (pl.col("ovr__games__r5__a") >= 5))

def train_test_season_splits(df: pl.DataFrame) -> tuple[list[str], str, str]:
    seasons = (
        df.select(pl.col("season").cast(pl.Utf8)).unique().sort(by="season").to_series().to_list()
        if "season" in df.columns
        else []
    )
    if len(seasons) < 3:
        raise ValueError("Need at least 3 seasons to create the requested splits.")
    current = seasons[-1]
    previous = seasons[-2]
    train = seasons[:-2]
    return train, previous, current

def make_targets(df: pl.DataFrame) -> pl.DataFrame:
    need = ["home_goals", "away_goals"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Required column missing for targets: {c}")
    df = df.with_columns(
        pl.when(pl.col("home_goals") > pl.col("away_goals"))
        .then(pl.lit("H"))
        .when(pl.col("home_goals") < pl.col("away_goals"))
        .then(pl.lit("A"))
        .otherwise(pl.lit("D"))
        .alias("match_result")
    )
    if "Over" not in df.columns:
        raise ValueError("'Over' column not found; ensure you used build_match_level().")
    return df

def prepare_matrices(df: pl.DataFrame, feature_cols: list[str], season_list: list[str], extra_cols: list[str] = None) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame | None]:
    part = df.filter(pl.col("season").cast(pl.Utf8).is_in(season_list))
    part = part.drop_nulls(subset=feature_cols)
    X = part.select(feature_cols).to_pandas()
    y_res = part.select("match_result").to_pandas().iloc[:, 0]
    y_over = part.select("Over").to_pandas().iloc[:, 0].astype(int)
    
    extras = None
    if extra_cols:
        extras = part.select(extra_cols).to_pandas()
        
    return X, y_res, y_over, extras

def evaluate_betting(proba, y_true, odds_df, model_type="result"):
    """
    Evaluates betting performance based on Expected Value (EV).
    Strategy: Bet 1 unit if EV > 0 on the outcome with highest probability.
    EV = prob * odds - 1
    """
    bets_placed = 0
    total_profit = 0.0
    
    if model_type == "result":
        # Classes: ['A', 'D', 'H'] (sorted)
        # Map indices to odds columns
        # 0 -> A -> odds_a
        # 1 -> D -> odds_d
        # 2 -> H -> odds_h
        
        # Ensure we have the odds columns
        if not all(col in odds_df.columns for col in ["odds_a", "odds_d", "odds_h"]):
            print("Missing odds columns for result betting evaluation.")
            return

        odds_values = odds_df[["odds_a", "odds_d", "odds_h"]].values
        outcomes_map = {0: "A", 1: "D", 2: "H"}
        
        # Get max prob index for each row
        pred_indices = np.argmax(proba, axis=1)
        max_probs = np.max(proba, axis=1)
        
        # Get corresponding odds
        selected_odds = odds_values[np.arange(len(odds_values)), pred_indices]
        
        # Calculate EV
        ev = (max_probs * selected_odds) - 1
        
        # Identify bets
        bet_mask = ev > 0
        bets_placed = np.sum(bet_mask)
        
        if bets_placed > 0:
            # Calculate profit
            # If won: profit = odds - 1
            # If lost: profit = -1
            
            actual_outcomes = y_true.values
            predicted_outcomes = np.array([outcomes_map[i] for i in pred_indices])
            
            # Filter for bets placed
            bet_outcomes = actual_outcomes[bet_mask]
            bet_predictions = predicted_outcomes[bet_mask]
            bet_odds = selected_odds[bet_mask]
            
            won_mask = (bet_outcomes == bet_predictions)
            
            # Profit calculation
            # Won: odds - 1
            # Lost: -1
            profits = np.where(won_mask, bet_odds - 1, -1.0)
            total_profit = np.sum(profits)
            
    else: # over
        # proba is prob of Over (class 1)
        # prob_under = 1 - proba
        
        if not all(col in odds_df.columns for col in ["odds_over", "odds_under"]):
            print("Missing odds columns for over/under betting evaluation.")
            return

        prob_over = proba
        prob_under = 1.0 - proba
        
        # Determine prediction (highest prob)
        pred_is_over = prob_over > prob_under # boolean
        
        max_probs = np.where(pred_is_over, prob_over, prob_under)
        
        # Get odds
        odds_over = odds_df["odds_over"].values
        odds_under = odds_df["odds_under"].values
        selected_odds = np.where(pred_is_over, odds_over, odds_under)
        
        # Calculate EV
        ev = (max_probs * selected_odds) - 1
        
        # Identify bets
        bet_mask = ev > 0
        bets_placed = np.sum(bet_mask)
        
        if bets_placed > 0:
            actual_over = y_true.values # 1 for Over, 0 for Under
            pred_values = pred_is_over.astype(int)
            
            bet_actual = actual_over[bet_mask]
            bet_pred = pred_values[bet_mask]
            bet_odds = selected_odds[bet_mask]
            
            won_mask = (bet_actual == bet_pred)
            
            profits = np.where(won_mask, bet_odds - 1, -1.0)
            total_profit = np.sum(profits)

    total_games = len(y_true)
    prop_bets = bets_placed / total_games if total_games > 0 else 0
    avg_return = total_profit / bets_placed if bets_placed > 0 else 0
    
    print(f"Betting Eval ({model_type}): Bets: {bets_placed}/{total_games} ({prop_bets:.1%}), Total Profit: {total_profit:.2f}, Avg Return/Bet: {avg_return:.4f}")
    return bets_placed, total_profit, avg_return

def evaluate_model(model, X_val, y_val, model_type="result"):
    if model_type == "result":
        pred = model.predict(X_val)
        proba = model.predict_proba(X_val)
        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, average="macro")
        print(f"Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
        return acc, f1, proba
    else: # over
        proba = model.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_val, pred)
        brier = brier_score_loss(y_val, proba)
        ll = log_loss(y_val, np.c_[1 - proba, proba])
        print(f"Accuracy: {acc:.4f}, Brier: {brier:.4f}, LogLoss: {ll:.4f}")
        return acc, brier, proba

def train_lgbm(X, y, objective="multiclass"):
    print(f"Training LGBM for {objective}...")
    params = {
        'colsample_bytree': 0.679486272613669,
        'learning_rate': 0.01110442342472048,
        'max_depth': 13,
        'min_child_samples': 26,
        'n_estimators': 491,
        'num_leaves': 54,
        'reg_alpha': 0.7901755405312056,
        'reg_lambda': 0.6059599747810114,
        'subsample': 0.9705203514053395,
        'n_jobs': -1,
        'verbosity': -1,
        'random_state': 42,
        'objective': objective
    }
    
    clf = LGBMClassifier(**params)
    clf.fit(X, y)
    return clf

def main(parquet_path: Path):
    print(f"Loading: {parquet_path}")
    df = load_frame(parquet_path)
    df = filter_min_history(df)
    df = make_targets(df)
    
    train_seasons, prev_season, curr_season = train_test_season_splits(df)
    print(f"Train: {train_seasons}, Val: {prev_season}, Excluded: {curr_season}")

    # Features
    base_feats = select_feature_columns(df)
    odds_res_cols = ["odds_h", "odds_d", "odds_a"]
    odds_over_cols = ["odds_over", "odds_under"]
    
    # Filter for comparison (must have odds)
    df_odds = df.drop_nulls(subset=odds_res_cols + odds_over_cols)
    print(f"Total rows: {len(df)}. Rows with odds: {len(df_odds)}")
    
    # --- Model A (No Odds) ---
    print("\n--- Model A (No Odds) ---")
    X_train_A, y_res_train, y_over_train, _ = prepare_matrices(df, base_feats, train_seasons)
    
    # We need odds for validation to evaluate betting
    odds_cols = odds_res_cols + odds_over_cols
    X_val_A, y_res_val, y_over_val, val_odds_A = prepare_matrices(df_odds, base_feats, [prev_season], extra_cols=odds_cols) 
    
    # Tune Result Model A
    print("Result Model A:")
    rf_A = train_lgbm(X_train_A, y_res_train, objective="multiclass")
    acc_res_A, _, proba_res_A = evaluate_model(rf_A, X_val_A, y_res_val, "result")
    bets_res_A, profit_res_A, roi_res_A = evaluate_betting(proba_res_A, y_res_val, val_odds_A, "result")
    
    # Tune Over Model A
    print("Over Model A:")
    lr_A = train_lgbm(X_train_A, y_over_train, objective="binary")
    acc_over_A, brier_over_A, proba_over_A = evaluate_model(lr_A, X_val_A, y_over_val, "over")
    bets_over_A, profit_over_A, roi_over_A = evaluate_betting(proba_over_A, y_over_val, val_odds_A, "over")
    
    # --- Model B (With Odds) ---
    print("\n--- Model B (With Odds) ---")
    # Train on df_odds only
    X_train_B_res, y_res_train_B, _, _ = prepare_matrices(df_odds, base_feats + odds_res_cols, train_seasons)
    X_val_B_res, _, _, val_odds_B_res = prepare_matrices(df_odds, base_feats + odds_res_cols, [prev_season], extra_cols=odds_cols)
    
    print("Result Model B:")
    rf_B = train_lgbm(X_train_B_res, y_res_train_B, objective="multiclass")
    acc_res_B, _, proba_res_B = evaluate_model(rf_B, X_val_B_res, y_res_val, "result")
    bets_res_B, profit_res_B, roi_res_B = evaluate_betting(proba_res_B, y_res_val, val_odds_B_res, "result")
    
    X_train_B_over, _, y_over_train_B, _ = prepare_matrices(df_odds, base_feats + odds_over_cols, train_seasons)
    X_val_B_over, _, _, val_odds_B_over = prepare_matrices(df_odds, base_feats + odds_over_cols, [prev_season], extra_cols=odds_cols)
    
    print("Over Model B:")
    lr_B = train_lgbm(X_train_B_over, y_over_train_B, objective="binary")
    acc_over_B, brier_over_B, proba_over_B = evaluate_model(lr_B, X_val_B_over, y_over_val, "over")
    bets_over_B, profit_over_B, roi_over_B = evaluate_betting(proba_over_B, y_over_val, val_odds_B_over, "over")
    
    # --- Comparison with Implied Odds ---
    print("\n--- Implied Odds Comparison ---")
    # Get odds for validation set
    val_odds = df_odds.filter(pl.col("season") == prev_season).select(odds_res_cols + odds_over_cols).to_pandas()
    
    # Implied probs (normalized)
    implied_h = 1 / val_odds["odds_h"]
    implied_d = 1 / val_odds["odds_d"]
    implied_a = 1 / val_odds["odds_a"]
    norm = implied_h + implied_d + implied_a
    implied_h /= norm
    implied_d /= norm
    implied_a /= norm
    
    implied_over = 1 / val_odds["odds_over"]
    implied_under = 1 / val_odds["odds_under"]
    norm_ou = implied_over + implied_under
    implied_over /= norm_ou
    
    # Correlation
    # Model A vs Implied
    # We need to match classes. rf.classes_ usually ['A', 'D', 'H'] sorted.
    classes = rf_A.classes_
    idx_h = np.where(classes == 'H')[0][0]
    
    corr_h_A = np.corrcoef(proba_res_A[:, idx_h], implied_h)[0, 1]
    corr_over_A = np.corrcoef(proba_over_A, implied_over)[0, 1]
    
    corr_h_B = np.corrcoef(proba_res_B[:, idx_h], implied_h)[0, 1]
    corr_over_B = np.corrcoef(proba_over_B, implied_over)[0, 1]
    
    print(f"Correlation (Home Prob vs Implied): Model A={corr_h_A:.4f}, Model B={corr_h_B:.4f}")
    print(f"Correlation (Over Prob vs Implied): Model A={corr_over_A:.4f}, Model B={corr_over_B:.4f}")
    
    # Save Model A (Baseline) as the production model for now
    dump(rf_A, MODELS_DIR / "result_model.joblib")
    dump(lr_A, MODELS_DIR / "over_model.joblib")
    
    # Save meta
    meta = {
        "parquet": str(parquet_path),
        "feature_cols": base_feats,
        "train_seasons": train_seasons,
        "validation_season": prev_season,
        "comparison": {
            "acc_res_A": acc_res_A, "acc_res_B": acc_res_B,
            "acc_over_A": acc_over_A, "acc_over_B": acc_over_B,
            "corr_h_A": corr_h_A, "corr_h_B": corr_h_B,
            "betting_res_A": {"bets": int(bets_res_A), "profit": float(profit_res_A), "roi": float(roi_res_A)},
            "betting_over_A": {"bets": int(bets_over_A), "profit": float(profit_over_A), "roi": float(roi_over_A)},
            "betting_res_B": {"bets": int(bets_res_B), "profit": float(profit_res_B), "roi": float(roi_res_B)},
            "betting_over_B": {"bets": int(bets_over_B), "profit": float(profit_over_B), "roi": float(roi_over_B)}
        }
    }
    (MODELS_DIR / "features_and_meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved Model A as production model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    args = parser.parse_args()
    main(args.parquet)
