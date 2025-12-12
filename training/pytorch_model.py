"""
Experimental over/under neural network with decorrelation loss.

This script mirrors the season-based split used by the tree models and
introduces a PyTorch MLP trained with a custom loss function that
penalizes similarity to bookmaker implied probabilities.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_PARQUET = Path("data/training/understat_df.parquet")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = Path("data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Check for GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Device Detection ---\nDevice: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print("------------------------\n")


def load_frame(parquet_path: Path) -> pl.DataFrame:
    lf = pl.scan_parquet(str(parquet_path))
    return lf.collect()


def select_feature_columns(df: pl.DataFrame) -> list[str]:
    cols = df.columns
    feat_cols = [
        c
        for c in cols
        if c.startswith("ovr__")
        and "__r5" in c
        and "__sum__" not in c
        and not c.startswith("ovr__games__")
        and (c.endswith("__h") or c.endswith("__a"))
    ]

    # Add Elo features if present
    for ec in ["elo_diff", "elo_sum", "elo_mean"]:
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


def add_targets_and_implied(df: pl.DataFrame) -> pl.DataFrame:
    if "Over" not in df.columns:
        raise ValueError("'Over' column not found; ensure you used build_match_level().")

    need_odds = ["odds_over", "odds_under"]
    missing = [c for c in need_odds if c not in df.columns]
    if missing:
        raise ValueError(f"Missing odds columns for implied probabilities: {missing}")

    df = df.with_columns(
        pl.when(pl.col("home_goals") > pl.col("away_goals"))
        .then(pl.lit("H"))
        .when(pl.col("home_goals") < pl.col("away_goals"))
        .then(pl.lit("A"))
        .otherwise(pl.lit("D"))
        .alias("match_result")
    )

    implied_over = 1 / pl.col("odds_over")
    implied_under = 1 / pl.col("odds_under")
    norm = implied_over + implied_under

    return df.with_columns((implied_over / norm).alias("implied_over_prob"))


def prepare_data(
    df: pl.DataFrame, 
    feature_cols: List[str], 
    season_list: List[str], 
    scaler: StandardScaler = None, 
    fit_scaler: bool = False
) -> Dict[str, np.ndarray]:
    """
    Selects data, scales features, and returns a dictionary of arrays.
    """
    part = df.filter(pl.col("season").cast(pl.Utf8).is_in(list(season_list)))
    
    # Filter out invalid odds (odds must be > 1.0 to be valid)
    # This handles the case where odds are 0.0 (causing Inf/NaN in implied probs)
    # and ensures we have valid data for the loss function.
    # We also drop nulls in features and targets.
    req_cols = list(set(feature_cols) | {"Over", "implied_over_prob", "odds_over", "odds_under", "date"})
    
    initial_count = len(part)
    part = part.filter(
        (pl.col("odds_over") > 1.0) & 
        (pl.col("odds_under") > 1.0) &
        pl.col("implied_over_prob").is_finite()
    ).drop_nulls(subset=req_cols)
    final_count = len(part)
    
    if initial_count != final_count:
        print(f"Dropped {initial_count - final_count} rows due to invalid odds/missing data in {season_list}")

    X = part.select(feature_cols).to_pandas().values
    
    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaler is not None:
        X = scaler.transform(X)
        
    y_over = part.select("Over").to_pandas().values.flatten().astype(int)
    implied = part.select("implied_over_prob").to_pandas().values.flatten()
    odds_over = part.select("odds_over").to_pandas().values.flatten()
    odds_under = part.select("odds_under").to_pandas().values.flatten()
    dates = part.select("date").to_pandas().values.flatten()
    
    print(
        f"Prepared seasons {season_list}: {X.shape[0]} rows, "
        f"{X.shape[1]} features after filtering"
    )

    return {
        "X": X,
        "y": y_over,
        "implied": implied,
        "odds_over": odds_over,
        "odds_under": odds_under,
        "dates": dates,
        "scaler": scaler
    }


def to_loader(data: Dict[str, np.ndarray], batch_size: int, shuffle: bool = True) -> DataLoader:
    tensor_x = torch.tensor(data["X"], dtype=torch.float32)
    tensor_implied = torch.tensor(data["implied"], dtype=torch.float32).unsqueeze(1)
    tensor_y = torch.tensor(data["y"], dtype=torch.float32).unsqueeze(1)
    
    ds = TensorDataset(tensor_x, tensor_implied, tensor_y)
    
    # Optimize loader for GPU
    kwargs = {'num_workers': 0, 'pin_memory': True} if DEVICE.type == 'cuda' else {}
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kwargs)


@dataclass
class TrainConfig:
    input_dim: int
    hidden_dim: int
    lambda_penalty: float
    epochs: int
    lr: float
    weight_decay: float
    batch_size: int
    dropout: float = 0.3
    patience: int = 15


class OverUnderNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 4),
            
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def decorrelated_mse_loss(
    logits: torch.Tensor, implied_prob: torch.Tensor, target: torch.Tensor, lambda_penalty: float
) -> torch.Tensor:
    pred_prob = torch.sigmoid(logits)
    base = F.mse_loss(pred_prob, target)
    if lambda_penalty == 0:
        return base
    penalty = -lambda_penalty * torch.mean((pred_prob - implied_prob) ** 2)
    return base + penalty


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0
            
    def load_best_weights(self, model):
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)


def train_model(config: TrainConfig, loader: DataLoader, val_loader: DataLoader = None) -> Tuple[OverUnderNet, Dict]:
    model = OverUnderNet(config.input_dim, config.hidden_dim, config.dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=config.patience, min_delta=1e-4)
    
    history = {"train_loss": [], "val_loss": []}

    model.train()
    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        for batch_x, batch_implied, batch_y in loader:
            batch_x, batch_implied, batch_y = batch_x.to(DEVICE), batch_implied.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = decorrelated_mse_loss(logits, batch_implied, batch_y, config.lambda_penalty)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        
        avg_loss = total_loss / len(loader.dataset)
        history["train_loss"].append(avg_loss)
        
        # Validation loss
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, bi, by in val_loader:
                    bx, bi, by = bx.to(DEVICE), bi.to(DEVICE), by.to(DEVICE)
                    logits = model(bx)
                    loss = decorrelated_mse_loss(logits, bi, by, config.lambda_penalty)
                    val_loss += loss.item() * len(bx)
            avg_val_loss = val_loss / len(val_loader.dataset)
            history["val_loss"].append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            early_stopping(avg_val_loss, model)
            
            model.train()
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
            
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                early_stopping.load_best_weights(model)
                break
        else:
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.5f}")
                
    if val_loader and not early_stopping.early_stop:
         early_stopping.load_best_weights(model)
                
    return model, history


def evaluate_profit(probs: np.ndarray, y_true: np.ndarray, odds_over: np.ndarray, odds_under: np.ndarray) -> Dict:
    """
    Evaluates profit based on value betting:
    Bet if Model_Prob * Odds - 1 > 0
    """
    # EV for Over bets
    ev_over = probs * odds_over - 1
    # EV for Under bets (Prob Under = 1 - Prob Over)
    ev_under = (1 - probs) * odds_under - 1
    
    # Identify bets
    bets_over = ev_over > 0
    bets_under = ev_under > 0
    
    # Calculate profit for each game if bet placed
    # Profit = Odds - 1 if win, -1 if loss
    profit_over_outcomes = np.where(y_true == 1, odds_over - 1, -1)
    profit_under_outcomes = np.where(y_true == 0, odds_under - 1, -1)
    
    # Filter for actual bets placed
    actual_profit_over = profit_over_outcomes[bets_over]
    actual_profit_under = profit_under_outcomes[bets_under]
    
    total_profit = np.sum(actual_profit_over) + np.sum(actual_profit_under)
    n_bets = len(actual_profit_over) + len(actual_profit_under)
    
    avg_profit = total_profit / n_bets if n_bets > 0 else 0.0
    percent_bets = (n_bets / len(y_true)) * 100
    
    return {
        "total_profit": total_profit,
        "avg_profit": avg_profit,
        "n_bets": n_bets,
        "percent_bets": percent_bets
    }


def evaluate_portfolio(
    probs: np.ndarray, 
    y_true: np.ndarray, 
    odds_over: np.ndarray, 
    odds_under: np.ndarray, 
    dates: np.ndarray,
    budget_per_day: float = 10.0,
    skip_negative_edge: bool = True
) -> Dict:
    
    df = pd.DataFrame({
        "date": dates,
        "prob_over": probs,
        "y_true": y_true,
        "odds_over": odds_over,
        "odds_under": odds_under
    })
    
    p = df["prob_over"]
    o_over = df["odds_over"]
    o_under = df["odds_under"]
    
    # Expected profit per 1 unit stake
    mu_over = p * o_over - 1
    mu_under = (1 - p) * o_under - 1
    
    # Second moment and variance
    e_x2_over = p * (o_over - 1)**2 + (1 - p) * 1
    var_over = e_x2_over - mu_over**2
    
    e_x2_under = (1 - p) * (o_under - 1)**2 + p * 1
    var_under = e_x2_under - mu_under**2
    
    # Choose side by value, not probability threshold
    better_is_over = mu_over >= mu_under
    mu_best = np.where(better_is_over, mu_over, mu_under)
    var_best = np.where(better_is_over, var_over, var_under)
    odds_best = np.where(better_is_over, o_over, o_under)
    
    df["bet_over"] = better_is_over
    df["mu"] = mu_best
    df["var"] = var_best
    df["odds"] = odds_best
    
    # Realized outcome
    df["won"] = np.where(df["bet_over"], df["y_true"] == 1, df["y_true"] == 0)
    
    if skip_negative_edge:
        df["eligible"] = df["mu"] > 0
    else:
        df["eligible"] = True
    
    daily_results_uniform = []
    daily_results_sharpe = []
    
    for date, group in df.groupby("date"):
        group = group[group["eligible"]]
        n_games = len(group)
        if n_games <= 3:
            # No positive-edge bets this day
            daily_results_uniform.append(0.0)
            daily_results_sharpe.append(0.0)
            continue
        
        # Uniform: equal stake per bet
        bet_amount_uniform = budget_per_day / n_games
        profit_uniform = np.sum(
            np.where(group["won"], bet_amount_uniform * (group["odds"] - 1), -bet_amount_uniform)
        )
        daily_results_uniform.append(profit_uniform)
        
        # Sharpe-style: weights ∝ μ / var (for μ>0)
        mus = group["mu"].values
        vars_ = group["var"].values + 1e-6
        
        raw_weights = mus / vars_
        raw_weights = np.maximum(0, raw_weights)  # just in case
        
        sum_weights = raw_weights.sum()
        if sum_weights > 0:
            norm_weights = raw_weights / sum_weights
            bets_sharpe = budget_per_day * norm_weights
            profits_sharpe = np.where(group["won"].values,
                                      bets_sharpe * (group["odds"].values - 1),
                                      -bets_sharpe)
            daily_results_sharpe.append(profits_sharpe.sum())
        else:
            daily_results_sharpe.append(0.0)
    
    uniform_daily = np.array(daily_results_uniform)
    sharpe_daily = np.array(daily_results_sharpe)
    
    def sharpe_ratio(x):
        if len(x) == 0:
            return 0.0
        std = x.std()
        return x.mean() / std if std > 0 else 0.0
    
    return {
        "uniform_total_profit": uniform_daily.sum(),
        "uniform_avg_daily_profit": uniform_daily.mean() if len(uniform_daily) > 0 else 0.0,
        "uniform_sharpe": sharpe_ratio(uniform_daily),
        "sharpe_total_profit": sharpe_daily.sum(),
        "sharpe_avg_daily_profit": sharpe_daily.mean() if len(sharpe_daily) > 0 else 0.0,
        "sharpe_sharpe": sharpe_ratio(sharpe_daily),
        "n_days": len(uniform_daily),
    }



def evaluate_model(model: OverUnderNet, data: Dict[str, np.ndarray]) -> dict:
    model.eval()
    X = torch.tensor(data["X"], dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        logits = model(X).squeeze(1)
        prob = torch.sigmoid(logits).cpu().numpy()

    y_true = data["y"]
    implied = data["implied"]
    
    preds = (prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    brier = brier_score_loss(y_true, prob)
    ll = log_loss(y_true, np.c_[1 - prob, prob], labels=[0, 1])
    corr = float(np.corrcoef(prob, implied)[0, 1])
    
    # Profit Eval
    profit_metrics = evaluate_profit(prob, y_true, data["odds_over"], data["odds_under"])
    
    # Portfolio Eval
    portfolio_metrics = evaluate_portfolio(
        prob, y_true, data["odds_over"], data["odds_under"], data["dates"]
    )

    print(f"Accuracy: {acc:.4f}, Brier: {brier:.4f}, LogLoss: {ll:.4f}, Corr: {corr:.4f}")
    print(f"Profit Eval: Bets: {profit_metrics['n_bets']} ({profit_metrics['percent_bets']:.1f}%), "
          f"Avg Profit: {profit_metrics['avg_profit']:.4f}, Total: {profit_metrics['total_profit']:.2f}")
    print(f"Portfolio Eval (Budget 10/day): Uniform Total: {portfolio_metrics['uniform_total_profit']:.2f}, "
          f"Sharpe Total: {portfolio_metrics['sharpe_total_profit']:.2f}")
          
    return {
        "accuracy": acc, 
        "brier": brier, 
        "log_loss": ll, 
        "corr": corr,
        **profit_metrics,
        **portfolio_metrics
    }


def plot_losses(history: Dict, title: str, filename: str):
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    if history["val_loss"]:
        plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"Loss Curve - {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / filename)
    plt.close()


def experiment(
    df: pl.DataFrame,
    base_feats: list[str],
    train_seasons: list[str],
    val_season: str,
    current_season: str,
    lambda_penalty: float,
    epochs: int,
    hidden_dim: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    patience: int,
):
    odds_over_cols = ["odds_over", "odds_under"]
    df_odds = df.drop_nulls(subset=odds_over_cols)
    print(f"Total rows: {len(df)}. Rows with odds: {len(df_odds)}")

    # --- Model A (No Odds) ---
    print("\n--- Preparing Model A (No Odds) ---")
    data_train_A = prepare_data(df_odds, base_feats, train_seasons, fit_scaler=True)
    data_val_A = prepare_data(df_odds, base_feats, [val_season], scaler=data_train_A["scaler"])
    
    loader_train_A = to_loader(data_train_A, batch_size)
    loader_val_A = to_loader(data_val_A, batch_size, shuffle=False)

    config_A = TrainConfig(
        input_dim=data_train_A["X"].shape[1],
        hidden_dim=hidden_dim,
        lambda_penalty=lambda_penalty,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        dropout=dropout,
        patience=patience,
    )
    
    print("Training Model A...")
    model_A, history_A = train_model(config_A, loader_train_A, loader_val_A)
    plot_losses(history_A, "Model A (No Odds)", "loss_model_A.png")
    
    print("Validation Model A:")
    metrics_A = evaluate_model(model_A, data_val_A)

    print("\n--- Evaluation on Current Season (Model A) ---")
    data_curr_A = prepare_data(df_odds, base_feats, [current_season], scaler=data_train_A["scaler"])
    metrics_curr_A = evaluate_model(model_A, data_curr_A)

    # --- Model B (With Odds) ---
    print("\n--- Preparing Model B (With Odds) ---")
    feat_with_odds = base_feats + odds_over_cols
    data_train_B = prepare_data(df_odds, feat_with_odds, train_seasons, fit_scaler=True)
    data_val_B = prepare_data(df_odds, feat_with_odds, [val_season], scaler=data_train_B["scaler"])
    
    loader_train_B = to_loader(data_train_B, batch_size)
    loader_val_B = to_loader(data_val_B, batch_size, shuffle=False)

    config_B = TrainConfig(
        input_dim=data_train_B["X"].shape[1],
        hidden_dim=hidden_dim,
        lambda_penalty=lambda_penalty,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        dropout=dropout,
        patience=patience,
    )
    
    print("Training Model B...")
    model_B, history_B = train_model(config_B, loader_train_B, loader_val_B)
    plot_losses(history_B, "Model B (With Odds)", "loss_model_B.png")
    
    print("Validation Model B:")
    metrics_B = evaluate_model(model_B, data_val_B)

    print("\n--- Evaluation on Current Season (Model B) ---")
    data_curr_B = prepare_data(df_odds, feat_with_odds, [current_season], scaler=data_train_B["scaler"])
    metrics_curr_B = evaluate_model(model_B, data_curr_B)

    # Save artifacts
    torch.save(model_A.state_dict(), MODELS_DIR / "over_under_nn_no_odds.pt")
    torch.save(model_B.state_dict(), MODELS_DIR / "over_under_nn_with_odds.pt")
    
    # Save scalers
    joblib.dump(data_train_A["scaler"], MODELS_DIR / "scaler_no_odds.joblib")
    joblib.dump(data_train_B["scaler"], MODELS_DIR / "scaler_with_odds.joblib")

    summary = {
        "lambda_penalty": lambda_penalty,
        "train_seasons": train_seasons,
        "validation_season": val_season,
        "current_season": current_season,
        "metrics": {
            "no_odds": {"val": metrics_A, "current": metrics_curr_A},
            "with_odds": {"val": metrics_B, "current": metrics_curr_B}
        },
        "feature_cols": {"no_odds": base_feats, "with_odds": feat_with_odds},
    }
    (MODELS_DIR / "over_under_nn_meta.json").write_text(json.dumps(summary, indent=2))
    print("Saved models, scalers, and metadata to data/models.")


def main():
    parser = argparse.ArgumentParser(description="Over/Under neural network experiment")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--lambda_penalty", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    print(f"Loading: {args.parquet}")
    df = load_frame(args.parquet)
    df = filter_min_history(df)
    df = add_targets_and_implied(df)

    train_seasons, prev_season, current_season = train_test_season_splits(df)
    print(f"Train: {train_seasons}, Val: {prev_season}, Current: {current_season}")

    base_feats = select_feature_columns(df)

    experiment(
        df,
        base_feats,
        train_seasons,
        prev_season,
        current_season,
        lambda_penalty=args.lambda_penalty,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
