"""
Analyze test set predicted residuals and calibration.

This script performs two types of analysis:

1. RESIDUAL ANALYSIS: Bins by market probability of the REALIZED outcome
   - Shows where model agrees/disagrees with market on actual outcomes
   
2. CALIBRATION ANALYSIS: Bins by market PREDICTED probability for each class
   - Shows calibration curves (predicted vs empirical frequency)
   - Standard reliability diagram
"""

import json
import sys
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.train_utils import (
	load_frame,
	filter_min_history,
	add_targets_and_implied,
	add_targets_and_implied_result,
	select_feature_columns,
	prepare_data,
	prepare_data_result,
	get_test_season,
	get_num_leagues,
)
from training.models.neural_net import MLP, TaskType, CategoricalConfig
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log_softmax_from_implied(implied_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""Convert implied probabilities to log-softmax space."""
	implied_probs = implied_probs / (implied_probs.sum(dim=-1, keepdim=True) + eps)
	return torch.log(implied_probs + eps)


def get_model_predictions(
	model: torch.nn.Module,
	X: np.ndarray,
	cat_features: np.ndarray,
	device: torch.device,
	task_type: TaskType,
	use_cat_features: bool = True,
	implied_probs: np.ndarray = None,
	use_residual_market: bool = False,
) -> np.ndarray:
	"""
	Get model predictions.
	
	Args:
		model: The trained model
		X: Input features
		cat_features: Categorical features
		device: Torch device
		task_type: 'binary' or 'multiclass'
		use_cat_features: Whether to use categorical features
		implied_probs: Market implied probabilities (required if use_residual_market=True)
		use_residual_market: If True, model outputs residuals and we add log(p_mkt)
	"""
	model.eval()
	with torch.no_grad():
		X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
		cat_tensor = torch.tensor(cat_features, dtype=torch.long, device=device) if use_cat_features else None
		
		if task_type == "binary":
			logits = model(X_tensor, cat_tensor)
			if use_residual_market and implied_probs is not None:
				# r(x) + log(p_mkt / (1-p_mkt)) for binary
				implied_tensor = torch.tensor(implied_probs, dtype=torch.float32, device=device)
				log_odds_market = torch.log(implied_tensor / (1 - implied_tensor + 1e-6) + 1e-6)
				logits = logits.flatten() + log_odds_market
			probs = torch.sigmoid(logits).cpu().numpy().flatten()
		else:  # multiclass
			logits = model(X_tensor, cat_tensor)
			if use_residual_market and implied_probs is not None:
				# r(x) + log(p_mkt) for multiclass
				implied_tensor = torch.tensor(implied_probs, dtype=torch.float32, device=device)
				implied_log = _log_softmax_from_implied(implied_tensor)
				logits = logits + implied_log
			probs = torch.softmax(logits, dim=1).cpu().numpy()
	
	return probs


def compute_log_loss_per_sample(y_true: np.ndarray, y_pred: np.ndarray, task_type: TaskType) -> np.ndarray:
	"""Compute log loss for each sample."""
	eps = 1e-15
	
	if task_type == "binary":
		y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
		losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
	else:  # multiclass
		n = len(y_true)
		y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
		losses = -np.log(y_pred_clipped[np.arange(n), y_true.astype(int)])
	
	return losses


# ============================================================================
# RESIDUAL ANALYSIS (Binned by realized outcome probability)
# ============================================================================

def analyze_residuals_by_realized_outcome(
	y_true: np.ndarray,
	model_probs: np.ndarray,
	implied_probs: np.ndarray,
	task_type: TaskType,
	n_bins: int = 10,
) -> Dict:
	"""Bin by market probability of the TRUE outcome."""
	# Get market prob for the actual outcome
	if task_type == "binary":
		market_prob_for_true = implied_probs
		model_prob_for_true = model_probs
	else:
		n = len(y_true)
		market_prob_for_true = implied_probs[np.arange(n), y_true.astype(int)]
		model_prob_for_true = model_probs[np.arange(n), y_true.astype(int)]
	
	residuals = model_prob_for_true - market_prob_for_true
	
	# Bin by market prob of true outcome
	bin_edges = np.percentile(market_prob_for_true, np.linspace(0, 100, n_bins + 1))
	bin_edges[-1] += 0.001
	bin_indices = np.digitize(market_prob_for_true, bin_edges) - 1
	bin_indices = np.clip(bin_indices, 0, n_bins - 1)
	
	model_log_loss = compute_log_loss_per_sample(y_true, model_probs, task_type)
	market_log_loss = compute_log_loss_per_sample(y_true, implied_probs, task_type)
	log_loss_delta = model_log_loss - market_log_loss
	
	bins = []
	for i in range(n_bins):
		mask = bin_indices == i
		n_samples = mask.sum()
		if n_samples == 0:
			continue
		
		bins.append({
			"bin_idx": i,
			"bin_range": (bin_edges[i], bin_edges[i + 1]),
			"n_samples": n_samples,
			"mean_market_prob": market_prob_for_true[mask].mean(),
			"mean_model_prob": model_prob_for_true[mask].mean(),
			"mean_residual": residuals[mask].mean(),
			"std_residual": residuals[mask].std(),
			"mean_log_loss_delta": log_loss_delta[mask].mean(),
			"model_log_loss": model_log_loss[mask].mean(),
			"market_log_loss": market_log_loss[mask].mean(),
		})
	
	return {"bins": bins}


# ============================================================================
# CALIBRATION ANALYSIS (Binned by predicted probability)
# ============================================================================

def analyze_calibration_by_predicted_prob(
	y_true: np.ndarray,
	model_probs: np.ndarray,
	implied_probs: np.ndarray,
	task_type: TaskType,
	n_bins: int = 10,
) -> Dict:
	"""Bin by market PREDICTED probability for each class."""
	if task_type == "binary":
		# Bin by market prob of positive class (Over)
		market_pred = implied_probs
		model_pred = model_probs
		
		bin_edges = np.percentile(market_pred, np.linspace(0, 100, n_bins + 1))
		bin_edges[-1] += 0.001
		bin_indices = np.digitize(market_pred, bin_edges) - 1
		bin_indices = np.clip(bin_indices, 0, n_bins - 1)
		
		bins = []
		for i in range(n_bins):
			mask = bin_indices == i
			n_samples = mask.sum()
			if n_samples == 0:
				continue
			
			empirical_freq = y_true[mask].mean()
			mean_market_prob = market_pred[mask].mean()
			mean_model_prob = model_pred[mask].mean()
			
			# Binary log loss
			eps = 1e-15
			market_ll = -(y_true[mask] * np.log(np.clip(market_pred[mask], eps, 1-eps)) + 
						  (1 - y_true[mask]) * np.log(np.clip(1 - market_pred[mask], eps, 1-eps))).mean()
			model_ll = -(y_true[mask] * np.log(np.clip(model_pred[mask], eps, 1-eps)) + 
						 (1 - y_true[mask]) * np.log(np.clip(1 - model_pred[mask], eps, 1-eps))).mean()
			
			bins.append({
				"bin_idx": i,
				"bin_range": (bin_edges[i], bin_edges[i + 1]),
				"n_samples": n_samples,
				"mean_market_prob": mean_market_prob,
				"mean_model_prob": mean_model_prob,
				"empirical_freq": empirical_freq,
				"calibration_error_market": mean_market_prob - empirical_freq,
				"calibration_error_model": mean_model_prob - empirical_freq,
				"log_loss_delta": model_ll - market_ll,
			})
		
		return {"task_type": task_type, "bins": bins}
	
	else:  # multiclass - analyze each class separately
		results_by_class = {}
		class_names = ["Home", "Draw", "Away"]
		
		for class_idx, class_name in enumerate(class_names):
			market_pred = implied_probs[:, class_idx]
			model_pred = model_probs[:, class_idx]
			y_binary = (y_true == class_idx).astype(int)
			
			bin_edges = np.percentile(market_pred, np.linspace(0, 100, n_bins + 1))
			bin_edges[-1] += 0.001
			bin_indices = np.digitize(market_pred, bin_edges) - 1
			bin_indices = np.clip(bin_indices, 0, n_bins - 1)
			
			bins = []
			for i in range(n_bins):
				mask = bin_indices == i
				n_samples = mask.sum()
				if n_samples == 0:
					continue
				
				empirical_freq = y_binary[mask].mean()
				mean_market_prob = market_pred[mask].mean()
				mean_model_prob = model_pred[mask].mean()
				
				# Binary log loss for this class
				eps = 1e-15
				market_ll = -(y_binary[mask] * np.log(np.clip(market_pred[mask], eps, 1-eps)) + 
							  (1 - y_binary[mask]) * np.log(np.clip(1 - market_pred[mask], eps, 1-eps))).mean()
				model_ll = -(y_binary[mask] * np.log(np.clip(model_pred[mask], eps, 1-eps)) + 
							 (1 - y_binary[mask]) * np.log(np.clip(1 - model_pred[mask], eps, 1-eps))).mean()
				
				bins.append({
					"bin_idx": i,
					"bin_range": (bin_edges[i], bin_edges[i + 1]),
					"n_samples": n_samples,
					"mean_market_prob": mean_market_prob,
					"mean_model_prob": mean_model_prob,
					"empirical_freq": empirical_freq,
					"calibration_error_market": mean_market_prob - empirical_freq,
					"calibration_error_model": mean_model_prob - empirical_freq,
					"log_loss_delta": model_ll - market_ll,
				})
			
			results_by_class[class_name] = bins
		
		return {"task_type": task_type, "results_by_class": results_by_class}


# ============================================================================
# PRINTING FUNCTIONS
# ============================================================================

def print_residual_table(results: Dict, task_type: TaskType):
	"""Print residual analysis table."""
	print(f"\n{'='*100}")
	print(f"RESIDUAL ANALYSIS - BINNED BY REALIZED OUTCOME PROBABILITY ({task_type.upper()})")
	print(f"{'='*100}\n")
	
	header = f"{'Bin':^6} | {'Market Prob':^12} | {'N':^6} | {'Mean Resid':^12} | {'Std Resid':^10} | {'ΔLog Loss':^12} | {'Model LL':^10} | {'Market LL':^10}"
	print(header)
	print("-" * len(header))
	
	for b in results["bins"]:
		prob_range = f"{b['bin_range'][0]:.3f}-{b['bin_range'][1]:.3f}"
		row = (
			f"D{b['bin_idx']+1:2d}    | "
			f"{prob_range:^12} | "
			f"{b['n_samples']:6d} | "
			f"{b['mean_residual']:+11.4f} | "
			f"{b['std_residual']:10.4f} | "
			f"{b['mean_log_loss_delta']:+11.4f} | "
			f"{b['model_log_loss']:10.4f} | "
			f"{b['market_log_loss']:10.4f}"
		)
		print(row)
	
	print(f"\n{'='*100}\n")


def print_calibration_table(results: Dict, task_type: TaskType):
	"""Print calibration analysis table."""
	print(f"\n{'='*100}")
	print(f"CALIBRATION ANALYSIS - BINNED BY PREDICTED PROBABILITY ({task_type.upper()})")
	print(f"{'='*100}\n")
	
	if task_type == "binary":
		bins_data = results["bins"]
		
		header = f"{'Bin':^6} | {'Prob Range':^12} | {'N':^6} | {'Market P':^9} | {'Model P':^9} | {'Actual':^9} | {'Mkt CalErr':^11} | {'Mod CalErr':^11} | {'ΔLog Loss':^11}"
		print(header)
		print("-" * len(header))
		
		for b in bins_data:
			prob_range = f"{b['bin_range'][0]:.3f}-{b['bin_range'][1]:.3f}"
			row = (
				f"D{b['bin_idx']+1:2d}    | "
				f"{prob_range:^12} | "
				f"{b['n_samples']:6d} | "
				f"{b['mean_market_prob']:9.4f} | "
				f"{b['mean_model_prob']:9.4f} | "
				f"{b['empirical_freq']:9.4f} | "
				f"{b['calibration_error_market']:+10.4f} | "
				f"{b['calibration_error_model']:+10.4f} | "
				f"{b['log_loss_delta']:+10.4f}"
			)
			print(row)
		
		print(f"\n{'='*100}\n")
	
	else:  # multiclass
		results_by_class = results["results_by_class"]
		
		for class_name, bins_data in results_by_class.items():
			print(f"\n--- {class_name.upper()} Outcome ---")
			header = f"{'Bin':^6} | {'Prob Range':^12} | {'N':^6} | {'Market P':^9} | {'Model P':^9} | {'Actual':^9} | {'Mkt CalErr':^11} | {'Mod CalErr':^11} | {'ΔLog Loss':^11}"
			print(header)
			print("-" * len(header))
			
			for b in bins_data:
				prob_range = f"{b['bin_range'][0]:.3f}-{b['bin_range'][1]:.3f}"
				row = (
					f"D{b['bin_idx']+1:2d}    | "
					f"{prob_range:^12} | "
					f"{b['n_samples']:6d} | "
					f"{b['mean_market_prob']:9.4f} | "
					f"{b['mean_model_prob']:9.4f} | "
					f"{b['empirical_freq']:9.4f} | "
					f"{b['calibration_error_market']:+10.4f} | "
					f"{b['calibration_error_model']:+10.4f} | "
					f"{b['log_loss_delta']:+10.4f}"
				)
				print(row)
		
		print(f"\n{'='*100}\n")


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_residual_analysis(results: Dict, task_type: TaskType, save_path: Path):
	"""Plot residual analysis."""
	bins_data = results["bins"]
	if not bins_data:
		return
	
	bin_centers = [b["mean_market_prob"] for b in bins_data]
	mean_residuals = [b["mean_residual"] for b in bins_data]
	log_loss_deltas = [b["mean_log_loss_delta"] for b in bins_data]
	
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))
	fig.suptitle(f"Residual Analysis - {task_type.upper()}", fontsize=14, fontweight="bold")
	
	# Plot 1: Mean residual
	axes[0].plot(bin_centers, mean_residuals, 'o-', linewidth=2, markersize=8)
	axes[0].axhline(0, color='red', linestyle='--', alpha=0.7)
	axes[0].set_xlabel("Market Prob (True Outcome)")
	axes[0].set_ylabel("Mean Residual (Model - Market)")
	axes[0].grid(True, alpha=0.3)
	
	# Plot 2: Log loss delta
	colors = ['green' if d < 0 else 'red' for d in log_loss_deltas]
	axes[1].bar(range(len(log_loss_deltas)), log_loss_deltas, color=colors, alpha=0.7)
	axes[1].axhline(0, color='black', linestyle='--')
	axes[1].set_xlabel("Decile")
	axes[1].set_ylabel("ΔLog Loss (Model - Market)")
	axes[1].set_title("Green = Model Better")
	
	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.close()


def plot_calibration_analysis(results: Dict, task_type: TaskType, save_path: Path):
	"""Plot calibration analysis."""
	if task_type == "binary":
		bins_data = results["bins"]
		if not bins_data:
			return
		
		market_probs = [b["mean_market_prob"] for b in bins_data]
		model_probs = [b["mean_model_prob"] for b in bins_data]
		empirical_freqs = [b["empirical_freq"] for b in bins_data]
		
		fig, ax = plt.subplots(1, 1, figsize=(8, 8))
		ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect Calibration')
		ax.plot(market_probs, empirical_freqs, 'o-', linewidth=2, markersize=8, label='Market', color='blue')
		ax.plot(model_probs, empirical_freqs, 's-', linewidth=2, markersize=8, label='Model', color='red')
		ax.set_xlabel("Predicted Probability (Over)")
		ax.set_ylabel("Empirical Frequency (Actual Over)")
		ax.set_title("Calibration Curve - Over/Under 2.5")
		ax.legend()
		ax.grid(True, alpha=0.3)
		ax.set_xlim(0, 1)
		ax.set_ylim(0, 1)
		
	else:  # multiclass
		results_by_class = results["results_by_class"]
		class_names = ["Home", "Draw", "Away"]
		
		fig, axes = plt.subplots(1, 3, figsize=(18, 5))
		fig.suptitle("Calibration Curves by Outcome Class", fontsize=14, fontweight="bold")
		
		for col_idx, class_name in enumerate(class_names):
			bins_data = results_by_class[class_name]
			if not bins_data:
				continue
			
			market_probs = [b["mean_market_prob"] for b in bins_data]
			model_probs = [b["mean_model_prob"] for b in bins_data]
			empirical_freqs = [b["empirical_freq"] for b in bins_data]
			
			ax = axes[col_idx]
			ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect')
			ax.plot(market_probs, empirical_freqs, 'o-', linewidth=2, markersize=6, label='Market', color='blue')
			ax.plot(model_probs, empirical_freqs, 's-', linewidth=2, markersize=6, label='Model', color='red')
			ax.set_xlabel(f"Predicted P({class_name})")
			ax.set_ylabel(f"Actual Freq ({class_name})")
			ax.set_title(f"{class_name} Calibration")
			ax.legend()
			ax.grid(True, alpha=0.3)
			ax.set_xlim(0, 1)
			ax.set_ylim(0, 1)
	
	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(task: str = "result", n_bins: int = 10):
	"""Main analysis pipeline."""
	print(f"\n{'='*80}")
	print(f"ANALYZING {task.upper()} PREDICTION")
	print(f"{'='*80}\n")
	
	# Paths
	project_root = Path(__file__).parent.parent
	data_path = project_root / "data" / "training" / "understat_df.parquet"
	models_dir = project_root / "data" / "models"
	plots_dir = project_root / "data" / "plots"
	plots_dir.mkdir(exist_ok=True, parents=True)
	
	# Task-specific configuration
	if task == "result":
		task_type = "multiclass"
		model_path = models_dir / "result_arch_tuned.pt"
		scaler_path = models_dir / "result_scaler_arch_tuned.joblib"
		config_path = models_dir / "result_architecture_config.json"
		prepare_fn = prepare_data_result
		add_targets_fn = add_targets_and_implied_result
		residual_plot_path = plots_dir / "residuals_by_realized_outcome_result.png"
		calib_plot_path = plots_dir / "calibration_by_predicted_prob_result.png"
	else:  # over_under
		task_type = "binary"
		model_path = models_dir / "over_under_arch_tuned.pt"
		scaler_path = models_dir / "scaler_arch_tuned.joblib"
		config_path = models_dir / "architecture_config.json"
		prepare_fn = prepare_data
		add_targets_fn = add_targets_and_implied
		residual_plot_path = plots_dir / "residuals_by_realized_outcome_over_under.png"
		calib_plot_path = plots_dir / "calibration_by_predicted_prob_over_under.png"
	
	# Load data
	print("Loading data...")
	df = load_frame(data_path)
	df = filter_min_history(df)
	df = add_targets_fn(df)
	
	test_season = get_test_season(df)
	print(f"Test season: {test_season}\n")
	
	# Load config
	with open(config_path, 'r') as f:
		config = json.load(f)
	
	feature_cols = config["feature_cols"]
	print(f"Features: {len(feature_cols)}\n")
	
	# Load scaler and prepare test data
	scaler = joblib.load(scaler_path)
	data_test = prepare_fn(df, feature_cols, [test_season], scaler=scaler)
	print(f"Test samples: {len(data_test['y'])}\n")
	
	# Load model
	use_cat_features = "num_leagues" in config or "cat_config" in config
	cat_config = None
	
	if use_cat_features:
		num_leagues = get_num_leagues(df)
		cat_config = CategoricalConfig(
			num_leagues=num_leagues,
			league_embed_dim=config.get("league_embed_dim", 3),
		)
	
	model = MLP(
		input_dim=len(feature_cols),
		hidden_layers=config["hidden_layers"],
		dropout=config["dropout"],
		norm=config.get("norm", "none"),
		activation=config.get("activation", "relu"),
		output_dim=3 if task_type == "multiclass" else 1,
		cat_config=cat_config,
	).to(DEVICE)
	
	model.load_state_dict(torch.load(model_path, map_location=DEVICE))
	print("Model loaded.\n")
	
	# Check if model was trained with residual_market approach
	# The result model uses residual_market=True (outputs r(x), combined with log p_mkt)
	use_residual_market = (task == "result")
	
	# Get predictions (with proper residual market handling)
	print(f"Using residual_market approach: {use_residual_market}")
	model_probs = get_model_predictions(
		model, 
		data_test["X"], 
		data_test["cat_features"],
		DEVICE, 
		task_type,
		use_cat_features,
		implied_probs=data_test["implied"],
		use_residual_market=use_residual_market,
	)
	
	# Also get raw model predictions (without market prior) for comparison
	model_probs_raw = get_model_predictions(
		model, 
		data_test["X"], 
		data_test["cat_features"],
		DEVICE, 
		task_type,
		use_cat_features,
		implied_probs=None,
		use_residual_market=False,
	)
	
	# ========== RAW RESIDUAL OUTPUT ANALYSIS ==========
	if use_residual_market:
		print(f"\n{'='*80}")
		print("0. RAW RESIDUAL OUTPUT ANALYSIS (What the MLP outputs directly)")
		print(f"{'='*80}\n")
		
		# The raw model outputs are r(x) passed through softmax - this shows what the MLP learned
		# These should be close to uniform (1/3, 1/3, 1/3) if the model learned small corrections
		print("Raw MLP output (softmax of r(x)) statistics:")
		print(f"  Home: mean={model_probs_raw[:, 0].mean():.4f}, std={model_probs_raw[:, 0].std():.4f}")
		print(f"  Draw: mean={model_probs_raw[:, 1].mean():.4f}, std={model_probs_raw[:, 1].std():.4f}")
		print(f"  Away: mean={model_probs_raw[:, 2].mean():.4f}, std={model_probs_raw[:, 2].std():.4f}")
		
		# Show residual magnitudes
		# r(x) = logits, so we can look at logit statistics
		model.eval()
		with torch.no_grad():
			X_tensor = torch.tensor(data_test["X"], dtype=torch.float32, device=DEVICE)
			cat_tensor = torch.tensor(data_test["cat_features"], dtype=torch.long, device=DEVICE)
			raw_logits = model(X_tensor, cat_tensor).cpu().numpy()
		
		print(f"\nRaw residual logits r(x) statistics:")
		print(f"  Home: mean={raw_logits[:, 0].mean():.4f}, std={raw_logits[:, 0].std():.4f}, range=[{raw_logits[:, 0].min():.3f}, {raw_logits[:, 0].max():.3f}]")
		print(f"  Draw: mean={raw_logits[:, 1].mean():.4f}, std={raw_logits[:, 1].std():.4f}, range=[{raw_logits[:, 1].min():.3f}, {raw_logits[:, 1].max():.3f}]")
		print(f"  Away: mean={raw_logits[:, 2].mean():.4f}, std={raw_logits[:, 2].std():.4f}, range=[{raw_logits[:, 2].min():.3f}, {raw_logits[:, 2].max():.3f}]")
	
	# ========== RESIDUAL ANALYSIS ==========
	print(f"\n{'='*80}")
	print("1. RESIDUAL ANALYSIS (Binned by Realized Outcome Probability)")
	print(f"{'='*80}")
	
	residual_results = analyze_residuals_by_realized_outcome(
		y_true=data_test["y"],
		model_probs=model_probs,
		implied_probs=data_test["implied"],
		task_type=task_type,
		n_bins=n_bins,
	)
	
	print_residual_table(residual_results, task_type)
	plot_residual_analysis(residual_results, task_type, residual_plot_path)
	print(f"Plot saved: {residual_plot_path}")
	
	# ========== CALIBRATION ANALYSIS ==========
	print(f"\n{'='*80}")
	print("2. CALIBRATION ANALYSIS (Binned by Predicted Probability)")
	print(f"{'='*80}")
	
	calibration_results = analyze_calibration_by_predicted_prob(
		y_true=data_test["y"],
		model_probs=model_probs,
		implied_probs=data_test["implied"],
		task_type=task_type,
		n_bins=n_bins,
	)
	
	print_calibration_table(calibration_results, task_type)
	plot_calibration_analysis(calibration_results, task_type, calib_plot_path)
	print(f"Plot saved: {calib_plot_path}")
	
	print(f"\n{'='*80}")
	print("ANALYSIS COMPLETE")
	print(f"{'='*80}\n")


if __name__ == "__main__":
	import sys
	
	task = sys.argv[1] if len(sys.argv) > 1 else "result"
	n_bins = int(sys.argv[2]) if len(sys.argv) > 2 else 10
	
	if task not in ["result", "over_under"]:
		print("Usage: python analyze_residuals_by_decile.py [result|over_under] [n_bins]")
		sys.exit(1)
	
	main(task=task, n_bins=n_bins)
