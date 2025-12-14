"""
Over/Under Neural Network with Hyperparameter Optimization

This script:
1. Tunes architecture hyperparameters with lambda_penalty=0 using Optuna
2. Saves the best model (Model A without odds)
3. After finding best architecture, tunes lambda_penalty to maximize Sharpe ratio
4. Logs everything to MLflow for experiment tracking

To view MLflow results:
	cd to project root, then run: mlflow ui
	Open http://127.0.0.1:5000 in your browser
	
MLflow provides:
	- Run comparison tables with all metrics
	- Parameter vs metric scatter plots
	- Parallel coordinates plots for hyperparameter analysis
	- Artifact storage (models, plots, etc.)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import copy

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_PARQUET = Path("data/training/understat_df.parquet")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = Path("data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameter search settings
N_TRIALS_ARCHITECTURE = 50  # Number of trials for architecture search
N_TRIALS_LAMBDA = 50  # Number of trials for lambda tuning
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 128

# --- MANUAL OVERRIDE FOR PHASE 2 ---
# If you want to skip Phase 1 (Architecture Search) and run only Phase 2 (Lambda Tuning),
# fill this dictionary with the best parameters from a previous run.
# Set to None to run the full pipeline.
# MANUAL_ARCH_PARAMS = {
#     'n_layers': 3,
#     'layer_0_size': 128,
#     'layer_1_size': 512,
#     'layer_2_size': 128,
#     'dropout': 0.3179188503051007,
#     'norm': 'ln',
#     'lr': 0.0022539642014855073,
#     'weight_decay': 0.0005345906215353935
# }
MANUAL_ARCH_PARAMS = None  # Uncomment to run full pipeline

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Device Detection ---\nDevice: {DEVICE}")
if DEVICE.type == "cuda":
	print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print("------------------------\n")


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================


def load_frame(parquet_path: Path) -> pl.DataFrame:
	return pl.scan_parquet(str(parquet_path)).collect()


def select_feature_columns(df: pl.DataFrame) -> List[str]:
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
	return df.filter(
		(pl.col("ovr__games__r5__h") >= 5) & (pl.col("ovr__games__r5__a") >= 5)
	)


def train_test_season_splits(df: pl.DataFrame) -> Tuple[List[str], str, str]:
	seasons = (
		df.select(pl.col("season").cast(pl.Utf8))
		.unique()
		.sort(by="season")
		.to_series()
		.to_list()
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
	fit_scaler: bool = False,
) -> Dict[str, np.ndarray]:
	"""Selects data, scales features, and returns a dictionary of arrays."""
	part = df.filter(pl.col("season").cast(pl.Utf8).is_in(list(season_list)))

	req_cols = list(
		set(feature_cols)
		| {"Over", "implied_over_prob", "odds_over", "odds_under", "date"}
	)

	initial_count = len(part)
	part = part.filter(
		(pl.col("odds_over") > 1.0)
		& (pl.col("odds_under") > 1.0)
		& pl.col("implied_over_prob").is_finite()
	).drop_nulls(subset=req_cols)
	final_count = len(part)

	if initial_count != final_count:
		print(
			f"Dropped {initial_count - final_count} rows due to invalid odds/missing data in {season_list}"
		)

	X = part.select(feature_cols).to_pandas().values

	if fit_scaler:
		scaler = StandardScaler()
		X = scaler.fit_transform(X)
	elif scaler is not None:
		X = scaler.transform(X)

	return {
		"X": X,
		"y": part.select("Over").to_pandas().values.flatten().astype(int),
		"implied": part.select("implied_over_prob").to_pandas().values.flatten(),
		"odds_over": part.select("odds_over").to_pandas().values.flatten(),
		"odds_under": part.select("odds_under").to_pandas().values.flatten(),
		"dates": part.select("date").to_pandas().values.flatten(),
		"scaler": scaler,
	}


def to_loader(
	data: Dict[str, np.ndarray], batch_size: int, shuffle: bool = True
) -> DataLoader:
	tensor_x = torch.tensor(data["X"], dtype=torch.float32)
	tensor_implied = torch.tensor(data["implied"], dtype=torch.float32).unsqueeze(1)
	tensor_y = torch.tensor(data["y"], dtype=torch.float32).unsqueeze(1)
	ds = TensorDataset(tensor_x, tensor_implied, tensor_y)
	kwargs = {"num_workers": 0, "pin_memory": True} if DEVICE.type == "cuda" else {}
	return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kwargs)


# ============================================================================
# MODEL DEFINITION
# ============================================================================


class MLP(nn.Module):
	"""Flexible MLP with configurable layers, dropout, and normalization."""

	def __init__(
		self,
		input_dim: int,
		hidden_layers: List[int],
		dropout: float = 0.3,
		norm: str = "none",
	):
		super().__init__()
		layers = []
		prev = input_dim
		NormClass = {"none": None, "bn": nn.BatchNorm1d, "ln": nn.LayerNorm}.get(norm)

		for h in hidden_layers:
			layers.append(nn.Linear(prev, h))
			if NormClass is not None:
				layers.append(NormClass(h))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout))
			prev = h
		layers.append(nn.Linear(prev, 1))
		self.net = nn.Sequential(*layers)
		self.hidden_layers = hidden_layers
		self.dropout = dropout
		self.norm = norm

	def forward(self, x):
		return self.net(x)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def _logits(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	p = torch.clamp(p, eps, 1 - eps)
	return torch.log(p) - torch.log(1 - p)

def batch_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""Compute batch correlation between two tensors."""
	x = x - x.mean()
	y = y - y.mean()
	vx = x.var(unbiased=False) + eps
	vy = y.var(unbiased=False) + eps
	cov = (x * y).mean()
	return cov / torch.sqrt(vx * vy)

def residual_market_loss(
	residuals_logits: torch.Tensor,
	implied_prob: torch.Tensor,
	target: torch.Tensor,
	lambda_repulsion: float = 0.0,
) -> torch.Tensor:
	"""Loss function for residual market model."""
	implied_logit = _logits(implied_prob)
	pred_logits = residuals_logits + implied_logit
	base = F.binary_cross_entropy_with_logits(pred_logits, target)
	
	if lambda_repulsion <= 0:
		return base
	
	pred_prob = torch.sigmoid(pred_logits)
	repulsion = (pred_prob - implied_prob) ** 2
	repulsion = repulsion.mean()
	return base - lambda_repulsion * repulsion

def logits_conditional_corr(
	pred_logits: torch.Tensor,
	implied_logits: torch.Tensor,
	target: torch.Tensor,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Approximate Corr(pred_logits, implied_logits | y)"""
	pred = pred_logits.view(-1)
	impl = implied_logits.view(-1)
	y = target.view(-1)
	total_n = pred.shape[0]
	rho_weighted = pred.new_tensor(0.0)
	weighted_sum = 0.0
	
	for r in (0, 1):
		mask = (y == r)
		n_r = int(mask.sum().item())
		if n_r > 1:
			pred_r = pred[mask]
			impl_r = impl[mask]
			rho_r = batch_corr(pred_r, impl_r, eps=eps)
			w_r = n_r / float(total_n)
			rho_weighted = rho_weighted + w_r * rho_r
			weighted_sum += w_r
			
	if weighted_sum == 0:
		return batch_corr(pred, impl, eps=eps)
	
	return rho_weighted

def residual_market_loss_corr(
	residuals_logits: torch.Tensor,
	implied_prob: torch.Tensor,
	target: torch.Tensor,
	lambda_repulsion: float = 0.0,
	lambda_corr: float = 0.0,
	conditional_corr: bool = True,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Loss function for residual market model with correlation penalty."""
	implied_logit = _logits(implied_prob)
	pred_logits = residuals_logits + implied_logit
	base = F.binary_cross_entropy_with_logits(pred_logits, target)
	loss = base
	
	if lambda_repulsion > 0.0:
		pred_prob = torch.sigmoid(pred_logits)
		repulsion = (pred_prob - implied_prob) ** 2
		repulsion = repulsion.mean()
		loss = loss - lambda_repulsion * repulsion
	
	if lambda_corr > 0.0:
		if conditional_corr:
			rho = logits_conditional_corr(pred_logits, implied_logit, target, eps=eps)
		else:
			rho = batch_corr(pred_logits.view(-1), implied_logit.view(-1), eps=eps)

		corr_penalty = (rho + 1.0) ** 2 
		loss = loss + lambda_corr * corr_penalty

	return loss
# ============================================================================
# TRAINING UTILITIES
# ============================================================================


class EarlyStopping:
	def __init__(self, patience: int = 7, min_delta: float = 0.0):
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_loss = None
		self.early_stop = False
		self.best_model_state = None

	def __call__(self, val_loss: float, model: nn.Module):
		if self.best_loss is None:
			self.best_loss = val_loss
			self.best_model_state = copy.deepcopy(model.state_dict())
		elif val_loss > self.best_loss - self.min_delta:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_loss = val_loss
			self.best_model_state = copy.deepcopy(model.state_dict())
			self.counter = 0

	def load_best_weights(self, model: nn.Module):
		if self.best_model_state:
			model.load_state_dict(self.best_model_state)


@dataclass
class TrainConfig:
	input_dim: int
	hidden_layers: List[int]
	dropout: float
	norm: str
	lr: float
	weight_decay: float
	lambda_repulsion: float
	lambda_corr: float
	epochs: int = MAX_EPOCHS
	patience: int = PATIENCE
	batch_size: int = BATCH_SIZE


def train_model(
	config: TrainConfig,
	train_loader: DataLoader,
	val_loader: DataLoader,
	trial: optuna.Trial = None,
	verbose: bool = True,
) -> Tuple[MLP, Dict, float]:
	"""
	Train model with early stopping.
	Returns (model, history, best_val_loss)
	"""
	model = MLP(
		config.input_dim, config.hidden_layers, config.dropout, config.norm
	).to(DEVICE)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=config.lr, weight_decay=config.weight_decay
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode="min", factor=0.5, patience=5
	)
	early_stopping = EarlyStopping(patience=config.patience, min_delta=1e-4)

	history = {"train_loss": [], "val_loss": []}

	for epoch in range(1, config.epochs + 1):
		# Training phase
		model.train()
		total_loss = 0.0
		for batch_x, batch_implied, batch_y in train_loader:
			batch_x = batch_x.to(DEVICE)
			batch_implied = batch_implied.to(DEVICE)
			batch_y = batch_y.to(DEVICE)

			optimizer.zero_grad()
			residual_logits = model(batch_x)
			# loss = loss_with_decorrelation(
			# 	logits, batch_implied, batch_y, config.lambda_penalty)
			loss = residual_market_loss_corr(
				residual_logits, batch_implied, batch_y,
                lambda_repulsion=config.lambda_repulsion, lambda_corr=config.lambda_corr
            )
			loss.backward()
			optimizer.step()
			total_loss += loss.item() * len(batch_x)

		avg_train_loss = total_loss / len(train_loader.dataset)
		history["train_loss"].append(avg_train_loss)

		# Validation phase
		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for bx, bi, by in val_loader:
				bx, bi, by = bx.to(DEVICE), bi.to(DEVICE), by.to(DEVICE)
				residual_logits = model(bx)
				# loss = loss_with_decorrelation(logits, bi, by, config.lambda_penalty)
				loss = residual_market_loss_corr(
                    residual_logits, bi, by,
                    lambda_repulsion=config.lambda_repulsion, lambda_corr=config.lambda_corr
                )
				val_loss += loss.item() * len(bx)

		avg_val_loss = val_loss / len(val_loader.dataset)
		history["val_loss"].append(avg_val_loss)

		scheduler.step(avg_val_loss)
		early_stopping(avg_val_loss, model)

		# Log to MLflow if in an active run
		if mlflow.active_run():
			mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
			mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

		if verbose and (epoch % 10 == 0 or epoch == 1):
			print(
				f"Epoch {epoch:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}"
			)

		# Optuna pruning
		if trial is not None:
			trial.report(avg_val_loss, epoch)
			if trial.should_prune():
				raise optuna.TrialPruned()

		if early_stopping.early_stop:
			if verbose:
				print(f"Early stopping at epoch {epoch}")
			break

	early_stopping.load_best_weights(model)
	return model, history, early_stopping.best_loss


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate_profit(
	probs: np.ndarray,
	y_true: np.ndarray,
	odds_over: np.ndarray,
	odds_under: np.ndarray,
) -> Dict:
	"""Evaluates profit based on value betting: Bet if Model_Prob * Odds - 1 > 0"""
	ev_over = probs * odds_over - 1
	ev_under = (1 - probs) * odds_under - 1

	bets_over = ev_over > 0
	bets_under = ev_under > 0

	profit_over_outcomes = np.where(y_true == 1, odds_over - 1, -1)
	profit_under_outcomes = np.where(y_true == 0, odds_under - 1, -1)

	actual_profit_over = profit_over_outcomes[bets_over]
	actual_profit_under = profit_under_outcomes[bets_under]

	total_profit = np.sum(actual_profit_over) + np.sum(actual_profit_under)
	n_bets = len(actual_profit_over) + len(actual_profit_under)

	return {
		"total_profit": float(total_profit),
		"avg_profit": float(total_profit / n_bets) if n_bets > 0 else 0.0,
		"n_bets": int(n_bets),
		"percent_bets": float((n_bets / len(y_true)) * 100),
	}


def evaluate_portfolio(
	probs: np.ndarray,
	y_true: np.ndarray,
	odds_over: np.ndarray,
	odds_under: np.ndarray,
	dates: np.ndarray,
	budget_per_day: float = 10.0,
) -> Dict:
	"""Portfolio-style evaluation with Sharpe-weighted betting."""
	df = pd.DataFrame(
		{
			"date": dates,
			"prob_over": probs,
			"y_true": y_true,
			"odds_over": odds_over,
			"odds_under": odds_under,
		}
	)

	p = df["prob_over"]
	o_over = df["odds_over"]
	o_under = df["odds_under"]

	# Expected value calculations
	mu_over = p * o_over - 1
	mu_under = (1 - p) * o_under - 1

	# Variance calculations
	e_x2_over = p * (o_over - 1) ** 2 + (1 - p) * 1
	var_over = e_x2_over - mu_over**2
	e_x2_under = (1 - p) * (o_under - 1) ** 2 + p * 1
	var_under = e_x2_under - mu_under**2

	better_is_over = mu_over >= mu_under
	mu_best = np.where(better_is_over, mu_over, mu_under)
	var_best = np.where(better_is_over, var_over, var_under)
	odds_best = np.where(better_is_over, o_over, o_under)

	df["bet_over"] = better_is_over
	df["mu"] = mu_best
	df["var"] = var_best
	df["odds"] = odds_best
	df["won"] = np.where(df["bet_over"], df["y_true"] == 1, df["y_true"] == 0)
	df["eligible"] = df["mu"] > 0

	daily_results_sharpe = []

	for _, group in df.groupby("date"):
		group = group[group["eligible"]]
		if len(group) <= 1:
			daily_results_sharpe.append(0.0)
			continue

		mus = group["mu"].values
		vars_ = group["var"].values + 1e-6
		raw_weights = np.maximum(0, mus / vars_)
		sum_weights = raw_weights.sum()

		if sum_weights > 0:
			norm_weights = raw_weights / sum_weights
			bets_sharpe = budget_per_day * norm_weights
			profits = np.where(
				group["won"].values,
				bets_sharpe * (group["odds"].values - 1),
				-bets_sharpe,
			)
			daily_results_sharpe.append(profits.sum())
		else:
			daily_results_sharpe.append(0.0)

	daily = np.array(daily_results_sharpe)

	def sharpe_ratio(x):
		if len(x) == 0:
			return 0.0
		std = x.std()
		return float(x.mean() / std) if std > 0 else 0.0

	return {
		"sharpe_total_profit": float(daily.sum()),
		"sharpe_avg_daily_profit": float(daily.mean()) if len(daily) > 0 else 0.0,
		"sharpe_ratio": sharpe_ratio(daily),
		"n_days": int(len(daily)),
	}


def evaluate_model(model: MLP, data: Dict[str, np.ndarray], verbose: bool = True) -> Dict:
	"""Full evaluation of model on a dataset."""
	model.eval()
	X = torch.tensor(data["X"], dtype=torch.float32).to(DEVICE)
	implied = torch.tensor(data["implied"], dtype=torch.float32).unsqueeze(1).to(DEVICE)

	with torch.no_grad():
		residual_logits = model(X)
		logits = _logits(implied) + residual_logits
		prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()

	y_true = data["y"]
	implied_np = data["implied"]

	preds = (prob >= 0.5).astype(int)
	acc = accuracy_score(y_true, preds)
	brier = brier_score_loss(y_true, prob)
	ll = log_loss(y_true, np.c_[1 - prob, prob], labels=[0, 1])
	corr = float(np.corrcoef(prob, implied_np)[0, 1])

	profit_metrics = evaluate_profit(prob, y_true, data["odds_over"], data["odds_under"])
	portfolio_metrics = evaluate_portfolio(
		prob, y_true, data["odds_over"], data["odds_under"], data["dates"]
	)

	if verbose:
		print(f"Accuracy: {acc:.4f}, Brier: {brier:.4f}, LogLoss: {ll:.4f}, Corr: {corr:.4f}")
		print(
			f"Profit: {profit_metrics['n_bets']} bets ({profit_metrics['percent_bets']:.1f}%), "
			f"Total: {profit_metrics['total_profit']:.2f}"
		)
		print(
			f"Portfolio Sharpe: {portfolio_metrics['sharpe_ratio']:.4f}, "
			f"Total: {portfolio_metrics['sharpe_total_profit']:.2f}"
		)

	return {
		"accuracy": float(acc),
		"brier": float(brier),
		"log_loss": float(ll),
		"corr_with_implied": float(corr),
		**profit_metrics,
		**portfolio_metrics,
	}


def plot_losses(history: Dict, title: str, filepath: Path):
	plt.figure(figsize=(10, 6))
	plt.plot(history["train_loss"], label="Train Loss")
	if history["val_loss"]:
		plt.plot(history["val_loss"], label="Val Loss")
	plt.title(f"Loss Curve - {title}")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.grid(True)
	plt.savefig(filepath)
	plt.close()


# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================


def create_architecture_objective(
	df: pl.DataFrame,
	feature_cols: List[str],
	train_seasons: List[str],
	val_season: str,
):
	"""
	Creates an Optuna objective for architecture search.
	lambda_penalty is fixed at 0 during this phase.
	"""
	# Prepare data once
	data_train = prepare_data(df, feature_cols, train_seasons, fit_scaler=True)
	data_val = prepare_data(df, feature_cols, [val_season], scaler=data_train["scaler"])
	train_loader = to_loader(data_train, BATCH_SIZE)
	val_loader = to_loader(data_val, BATCH_SIZE, shuffle=False)
	input_dim = data_train["X"].shape[1]

	def objective(trial: optuna.Trial) -> float:
		# Sample architecture hyperparameters
		n_layers = trial.suggest_int("n_layers", 2, 5)
		layer_sizes = []
		for i in range(n_layers):
			size = trial.suggest_categorical(f"layer_{i}_size", [64, 128, 256, 512])
			layer_sizes.append(size)

		dropout = trial.suggest_float("dropout", 0.1, 0.5)
		norm = trial.suggest_categorical("norm", ["none", "ln"])
		lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
		weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

		config = TrainConfig(
			input_dim=input_dim,
			hidden_layers=layer_sizes,
			dropout=dropout,
			norm=norm,
			lr=lr,
			weight_decay=weight_decay,
			lambda_repulsion=0.0,  # Fixed at 0 for architecture search
			lambda_corr=0.0,  # Fixed at 0 for architecture search
		)

		_, _, best_val_loss = train_model(
			config, train_loader, val_loader, trial=trial, verbose=False
		)

		return best_val_loss

	return objective, data_train, data_val


def create_lambda_objective(
	df: pl.DataFrame,
	feature_cols: List[str],
	train_seasons: List[str],
	val_season: str,
	best_arch_params: Dict,
):
	"""
	Creates an Optuna objective for lambda tuning.
	Takes the best architecture from Phase 1 and retrains it with different
	lambda's values.	
	Architecture (layers, dropout, lr, etc.) is fixed from best_arch_params.
	Only lambda_repulsion and lambda_corr vary across trials.
	"""
	data_train = prepare_data(df, feature_cols, train_seasons, fit_scaler=True)
	data_val = prepare_data(df, feature_cols, [val_season], scaler=data_train["scaler"])
	train_loader = to_loader(data_train, BATCH_SIZE)
	val_loader = to_loader(data_val, BATCH_SIZE, shuffle=False)
	input_dim = data_train["X"].shape[1]

	# Extract architecture from best params (from Phase 1)
	n_layers = best_arch_params["n_layers"]
	hidden_layers = [best_arch_params[f"layer_{i}_size"] for i in range(n_layers)]

	def objective(trial: optuna.Trial) -> float:
		lambda_repulsion = trial.suggest_float("lambda_repulsion", 0.0, 0.5)
		lambda_corr = trial.suggest_float("lambda_corr", 0.0001, 0.3, log=True)

		# Same architecture, only lambda changes
		config = TrainConfig(
			input_dim=input_dim,
			hidden_layers=hidden_layers,
			dropout=best_arch_params["dropout"],
			norm=best_arch_params["norm"],
			lr=best_arch_params["lr"],
			weight_decay=best_arch_params["weight_decay"],
			lambda_repulsion=lambda_repulsion,
			lambda_corr=lambda_corr,
		)

		# Retrain model from scratch with this lambda value
		model, _, _ = train_model(config, train_loader, val_loader, verbose=False)
		metrics = evaluate_model(model, data_val, verbose=False)

		# Log all metrics to MLflow
		if mlflow.active_run():
			mlflow.log_metrics(
				{
					"val_accuracy": metrics["accuracy"],
					"val_brier": metrics["brier"],
					"val_log_loss": metrics["log_loss"],
					"val_corr": metrics["corr_with_implied"],
					"val_sharpe_ratio": metrics["sharpe_ratio"],
					"val_sharpe_profit": metrics["sharpe_total_profit"],
					"val_n_bets": metrics["n_bets"],
				}
			)

		# Maximize Sharpe ratio (return negative for minimization)
		return -metrics["sharpe_ratio"]

	return objective, data_train, data_val


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def run_pipeline():
	"""Main entry point for the training pipeline."""
	print("=" * 60)
	print("OVER/UNDER NEURAL NETWORK TRAINING PIPELINE")
	print("=" * 60)

	def set_seed(s=42):
		import random
		random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	set_seed(42)
	# Load and prepare data
	print(f"\nLoading data from {DEFAULT_PARQUET}")
	df = load_frame(DEFAULT_PARQUET)
	df = filter_min_history(df)
	df = add_targets_and_implied(df)

	train_seasons, val_season, test_season = train_test_season_splits(df)
	print(f"Train: {train_seasons}")
	print(f"Validation: {val_season}")
	print(f"Test (current): {test_season}")

	base_feats = select_feature_columns(df)
	print(f"Features: {len(base_feats)} columns")

	# Filter to rows with valid odds
	df = df.drop_nulls(subset=["odds_over", "odds_under"])
	print(f"Total rows with odds: {len(df)}")

	# Set up MLflow
	mlflow.set_experiment("over_under_nn")

	best_arch_params = None
	data_train = None
	data_val = None

	if MANUAL_ARCH_PARAMS is None:
		# ========================================================================
		# PHASE 1: Architecture Search (lambdas=0)
		# ========================================================================
		print("\n" + "=" * 60)
		print("PHASE 1: ARCHITECTURE SEARCH (lambdas=0)")
		print("=" * 60)

		with mlflow.start_run(run_name="architecture_search"):
			mlflow.log_param("phase", "architecture_search")
			mlflow.log_param("n_trials", N_TRIALS_ARCHITECTURE)
			mlflow.log_param("train_seasons", str(train_seasons))
			mlflow.log_param("val_season", val_season)

			objective_arch, data_train, data_val = create_architecture_objective(
				df, base_feats, train_seasons, val_season
			)

			study_arch = optuna.create_study(
				direction="minimize",
				pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
				study_name="architecture_search",
			)

			study_arch.optimize(
				objective_arch,
				n_trials=N_TRIALS_ARCHITECTURE,
				show_progress_bar=True,
			)

			best_arch_params = study_arch.best_params
			print(f"\nBest architecture params: {best_arch_params}")
			print(f"Best validation loss: {study_arch.best_value:.5f}")

			mlflow.log_params(best_arch_params)
			mlflow.log_metric("best_val_loss", study_arch.best_value)

		# ========================================================================
		# Train final model with best architecture (lambda=0)
		# ========================================================================
		print("\n" + "=" * 60)
		print("TRAINING FINAL MODEL WITH BEST ARCHITECTURE")
		print("=" * 60)

		n_layers = best_arch_params["n_layers"]
		best_hidden_layers = [best_arch_params[f"layer_{i}_size"] for i in range(n_layers)]

		with mlflow.start_run(run_name="best_model_lambda0"):
			mlflow.log_params(best_arch_params)
			mlflow.log_param("lambda_repulsion", 0.0)
			mlflow.log_param("lambda_corr", 0.0)

			config = TrainConfig(
				input_dim=data_train["X"].shape[1],
				hidden_layers=best_hidden_layers,
				dropout=best_arch_params["dropout"],
				norm=best_arch_params["norm"],
				lr=best_arch_params["lr"],
				weight_decay=best_arch_params["weight_decay"],
				lambda_repulsion=0.0,
				lambda_corr=0.0,
			)

			train_loader = to_loader(data_train, BATCH_SIZE)
			val_loader = to_loader(data_val, BATCH_SIZE, shuffle=False)

			print("Training best model...")
			best_model, history, _ = train_model(config, train_loader, val_loader, verbose=True)

			# Plot and log loss curves
			plot_path = PLOTS_DIR / "loss_best_model.png"
			plot_losses(history, "Best Model (lambdas=0)", plot_path)
			mlflow.log_artifact(str(plot_path))

			# Evaluate on validation
			print("\n--- Validation Set ---")
			val_metrics = evaluate_model(best_model, data_val)
			mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))})

			# Evaluate on test set
			print("\n--- Test Set (Current Season) ---")
			data_test = prepare_data(df, base_feats, [test_season], scaler=data_train["scaler"])
			test_metrics = evaluate_model(best_model, data_test)
			mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))})

			# Save model
			model_path = MODELS_DIR / "over_under_nn_no_odds.pt"
			torch.save(best_model.state_dict(), model_path)
			mlflow.log_artifact(str(model_path))

			# Save scaler
			scaler_path = MODELS_DIR / "scaler_no_odds.joblib"
			joblib.dump(data_train["scaler"], scaler_path)
			mlflow.log_artifact(str(scaler_path))

			# Save metadata
			meta = {
				"architecture": {
					"hidden_layers": best_hidden_layers,
					"dropout": best_arch_params["dropout"],
					"norm": best_arch_params["norm"],
				},
				"training": {
					"lr": best_arch_params["lr"],
					"weight_decay": best_arch_params["weight_decay"],
					"lambda_repulsion": 0.0,
					"lambda_corr": 0.0,
				},
				"features": base_feats,
				"train_seasons": train_seasons,
				"val_season": val_season,
				"test_season": test_season,
				"val_metrics": val_metrics,
				"test_metrics": test_metrics,
			}
			meta_path = MODELS_DIR / "over_under_nn_meta.json"
			meta_path.write_text(json.dumps(meta, indent=2))
			mlflow.log_artifact(str(meta_path))
	else:
		print("\n" + "=" * 60)
		print("SKIPPING PHASE 1: USING MANUAL ARCHITECTURE PARAMETERS")
		print("=" * 60)
		best_arch_params = MANUAL_ARCH_PARAMS
		print(f"Loaded params: {best_arch_params}")
		
		# Prepare data manually since Phase 1 didn't run
		data_train = prepare_data(df, base_feats, train_seasons, fit_scaler=True)
		data_val = prepare_data(df, base_feats, [val_season], scaler=data_train["scaler"])
		data_test = prepare_data(df, base_feats, [test_season], scaler=data_train["scaler"])
		train_loader = to_loader(data_train, BATCH_SIZE)
		val_loader = to_loader(data_val, BATCH_SIZE, shuffle=False)
		n_layers = best_arch_params["n_layers"]
		best_hidden_layers = [best_arch_params[f"layer_{i}_size"] for i in range(n_layers)]

	# ========================================================================
	# PHASE 2: Lambda Tuning (maximize Sharpe)
	# ========================================================================
	print("\n" + "=" * 60)
	print("PHASE 2: LAMBDA TUNING (maximize Sharpe ratio)")
	print("=" * 60)

	# End previous run before lambda tuning to avoid conflicts
	objective_lambda, _, _ = create_lambda_objective(
		df, base_feats, train_seasons, val_season, best_arch_params
	)

	study_lambda = optuna.create_study(
		direction="minimize",  # We return negative Sharpe
		study_name="lambda_tuning",
	)

	# Log each lambda trial as a separate MLflow run
	def log_trial_callback(study, trial):
		if trial.state == optuna.trial.TrialState.COMPLETE:
			with mlflow.start_run(run_name=f"lambda_trial_{trial.number}", nested=True):
				mlflow.log_params(trial.params)
				mlflow.log_param("trial_number", trial.number)
				mlflow.log_metric("neg_sharpe_ratio", trial.value)
				# The objective function already logged other metrics

	with mlflow.start_run(run_name="lambda_tuning_study"):
		mlflow.log_param("phase", "lambda_tuning")
		mlflow.log_param("n_trials", N_TRIALS_LAMBDA)
		mlflow.log_params(best_arch_params)

		study_lambda.optimize(
			objective_lambda,
			n_trials=N_TRIALS_LAMBDA,
			show_progress_bar=True,
			callbacks=[log_trial_callback],
		)

		best_lambda_repulsion = study_lambda.best_params["lambda_repulsion"]
		best_lambda_corr = study_lambda.best_params["lambda_corr"]
		best_sharpe = -study_lambda.best_value
		print(f"\nBest lambda_repulsion: {best_lambda_repulsion:.4f}")
		print(f"Best lambda_corr: {best_lambda_corr:.4f}")
		print(f"Best Sharpe ratio: {best_sharpe:.4f}")

		mlflow.log_param("best_lambda_repulsion", best_lambda_repulsion)
		mlflow.log_metric("best_sharpe_ratio", best_sharpe)

	# ========================================================================
	# Train final model with best lambdas (if lambda > 0)
	# ========================================================================
	if best_lambda_repulsion != 0.0 or best_lambda_corr != 0.0:
		print("\n" + "=" * 60)
		print(f"TRAINING FINAL MODEL WITH LAMBDA_REPULSION={best_lambda_repulsion:.4f} AND LAMBDA_CORR={best_lambda_corr:.4f}")
		print("=" * 60)

		with mlflow.start_run(run_name="best_model_with_lambda"):
			mlflow.log_params(best_arch_params)
			mlflow.log_param("lambda_repulsion", best_lambda_repulsion)
			mlflow.log_param("lambda_corr", best_lambda_corr)
			
			config = TrainConfig(
				input_dim=data_train["X"].shape[1],
				hidden_layers=best_hidden_layers,
				dropout=best_arch_params["dropout"],
				norm=best_arch_params["norm"],
				lr=best_arch_params["lr"],
				weight_decay=best_arch_params["weight_decay"],
				lambda_repulsion=best_lambda_repulsion,
				lambda_corr=best_lambda_corr,
			)

			best_model_lambda, history, _ = train_model(
				config, train_loader, val_loader, verbose=True
			)

			# Evaluate
			print("\n--- Validation Set ---")
			val_metrics = evaluate_model(best_model_lambda, data_val)
			mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))})

			print("\n--- Test Set ---")
			test_metrics = evaluate_model(best_model_lambda, data_test)
			mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))})

			# Save
			model_path = MODELS_DIR / "over_under_nn_sharpe_optimized.pt"
			torch.save(best_model_lambda.state_dict(), model_path)
			mlflow.log_artifact(str(model_path))

	print("\n" + "=" * 60)
	print("PIPELINE COMPLETE")
	print("=" * 60)
	print("\nTo view results, run: mlflow ui")
	print("Then open http://127.0.0.1:5000 in your browser")


if __name__ == "__main__":
	run_pipeline()