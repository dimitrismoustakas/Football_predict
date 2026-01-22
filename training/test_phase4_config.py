"""
Quick test script to evaluate a specific configuration with the new Phase 4 approach.
Tests whether the improved early stopping method reduces overfitting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import torch
from training.models import TrainConfig, CategoricalConfig
from training.train_utils import (
	add_targets_and_implied_result,
	build_hidden_layers,
	filter_min_history,
	generate_rolling_cv_folds,
	get_test_season,
	get_num_leagues,
	load_frame,
	prepare_data_result,
	select_feature_columns,
	to_loader,
	train_model,
	evaluate_implied_baseline,
)
from training.evaluation import evaluate_model

# Configuration to test
TEST_CONFIG = {
	'base_width': 128,
	'n_layers': 5,
	'shape': 'pyramid',
	'activation': 'silu',
	'norm': 'ln',
	'lr': 0.0008158793058505103,
	'weight_decay': 9.440685529111033e-06,
	'dropout': 0.4333529892406824,
	'batch_size': 128,
	'scheduler_type': 'onecycle'
}

# Training parameters (from refinement phase)
REFINE_EPOCHS = 80
REFINE_PATIENCE = 20
N_CV_FOLDS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASK_TYPE = "multiclass"


def main():
	print("=" * 60)
	print("TESTING PHASE 4 WITH NEW EARLY STOPPING APPROACH")
	print("=" * 60)
	
	print(f"\nDevice: {DEVICE}")
	print(f"\nConfiguration to test:")
	for k, v in TEST_CONFIG.items():
		print(f"  {k}: {v}")
	
	# Load and prepare data
	print("\n--- Loading Data ---")
	data_path = Path("data/training/understat_df.parquet")
	df = load_frame(data_path)
	df = filter_min_history(df)
	df = add_targets_and_implied_result(df)
	df = df.drop_nulls(subset=["odds_home", "odds_draw", "odds_away"])
	print(f"Total rows with result odds: {len(df)}")
	
	feature_cols = select_feature_columns(df)
	print(f"Features: {len(feature_cols)} columns")
	
	# Create categorical config
	num_leagues = get_num_leagues(df)
	cat_config = CategoricalConfig(
		num_leagues=num_leagues,
		league_embed_dim=3,
	)
	print(f"Categorical: {num_leagues} leagues, embed_dim=3")
	
	# Generate folds and test season
	print(f"\nGenerating {N_CV_FOLDS}-fold rolling CV splits...")
	folds = generate_rolling_cv_folds(df, n_folds=N_CV_FOLDS)
	test_season = get_test_season(df)
	print(f"Test season (held out): {test_season}")
	
	# Get all CV seasons
	all_cv_seasons = set()
	for train_seasons, val_season in folds:
		all_cv_seasons.update(train_seasons)
		all_cv_seasons.add(val_season)
	all_cv_seasons = sorted(all_cv_seasons)
	
	# Use last CV season for early stopping
	final_val_season = all_cv_seasons[-1]
	initial_train_seasons = all_cv_seasons[:-1]
	
	print(f"\n--- Phase 4 Training Strategy ---")
	print(f"Step 1: Train on {initial_train_seasons[0]}..{initial_train_seasons[-1]} ({len(initial_train_seasons)} seasons)")
	print(f"        Validate on {final_val_season} (early stopping)")
	print(f"Step 2: Retrain on {all_cv_seasons[0]}..{all_cv_seasons[-1]} ({len(all_cv_seasons)} seasons) for best_epoch")
	
	# Build config
	hidden_layers = build_hidden_layers(
		TEST_CONFIG['base_width'],
		TEST_CONFIG['n_layers'],
		TEST_CONFIG['shape']
	)
	print(f"\nArchitecture: {hidden_layers}")
	
	# === STEP 1: Train with early stopping ===
	print("\n" + "=" * 60)
	print("STEP 1: FINDING BEST EPOCH WITH EARLY STOPPING")
	print("=" * 60)
	
	data_initial_train = prepare_data_result(df, feature_cols, initial_train_seasons, fit_scaler=True)
	data_final_val = prepare_data_result(df, feature_cols, [final_val_season], scaler=data_initial_train["scaler"])
	
	initial_train_loader = to_loader(data_initial_train, TEST_CONFIG['batch_size'], device=DEVICE, task_type=TASK_TYPE)
	final_val_loader = to_loader(data_final_val, TEST_CONFIG['batch_size'], shuffle=False, device=DEVICE, task_type=TASK_TYPE)
	
	early_stop_config = TrainConfig(
		input_dim=data_initial_train["X"].shape[1],
		hidden_layers=hidden_layers,
		dropout=TEST_CONFIG['dropout'],
		norm=TEST_CONFIG['norm'],
		lr=TEST_CONFIG['lr'],
		weight_decay=TEST_CONFIG['weight_decay'],
		lambda_repulsion=0.0,
		lambda_corr=0.0,
		activation=TEST_CONFIG['activation'],
		scheduler_type=TEST_CONFIG['scheduler_type'],
		epochs=REFINE_EPOCHS,
		patience=REFINE_PATIENCE,
		batch_size=TEST_CONFIG['batch_size'],
		task_type=TASK_TYPE,
		cat_config=cat_config,
	)
	
	_, early_stop_history, best_val_loss = train_model(
		early_stop_config, initial_train_loader, final_val_loader, device=DEVICE, verbose=True
	)
	
	# Find the epoch with the best validation loss (not just when early stopping triggered)
	best_epoch = early_stop_history["val_loss"].index(min(early_stop_history["val_loss"])) + 1
	total_epochs_trained = len(early_stop_history["val_loss"])
	print(f"\n✓ Early stopping: trained for {total_epochs_trained} epochs, best was epoch {best_epoch} (val_loss = {best_val_loss:.5f})")
	
	# === STEP 2: Retrain on all data for best_epoch ===
	print("\n" + "=" * 60)
	print(f"STEP 2: RETRAINING ON ALL DATA FOR {best_epoch} EPOCHS")
	print("=" * 60)
	
	data_train = prepare_data_result(df, feature_cols, all_cv_seasons, fit_scaler=True)
	data_test = prepare_data_result(df, feature_cols, [test_season], scaler=data_train["scaler"])
	
	train_loader = to_loader(data_train, TEST_CONFIG['batch_size'], device=DEVICE, task_type=TASK_TYPE)
	dummy_val_loader = to_loader(data_train, TEST_CONFIG['batch_size'], shuffle=False, device=DEVICE, task_type=TASK_TYPE)
	
	final_config = TrainConfig(
		input_dim=data_train["X"].shape[1],
		hidden_layers=hidden_layers,
		dropout=TEST_CONFIG['dropout'],
		norm=TEST_CONFIG['norm'],
		lr=TEST_CONFIG['lr'],
		weight_decay=TEST_CONFIG['weight_decay'],
		lambda_repulsion=0.0,
		lambda_corr=0.0,
		activation=TEST_CONFIG['activation'],
		scheduler_type=TEST_CONFIG['scheduler_type'],
		epochs=best_epoch,
		patience=best_epoch + 1,  # Disable early stopping
		batch_size=TEST_CONFIG['batch_size'],
		task_type=TASK_TYPE,
		cat_config=cat_config,
	)
	
	model, history, _ = train_model(
		final_config, train_loader, dummy_val_loader, device=DEVICE, verbose=True
	)
	
	# === EVALUATION ===
	print("\n" + "=" * 60)
	print("EVALUATION ON TEST SET")
	print("=" * 60)
	
	# Baseline (bookmaker implied probabilities)
	print("\n--- Baseline (Bookmaker Implied Probabilities) ---")
	baseline_metrics = evaluate_implied_baseline(data_test, task_type=TASK_TYPE)
	print(f"Accuracy: {baseline_metrics['accuracy']:.4f}")
	print(f"Brier:    {baseline_metrics['brier']:.4f}")
	print(f"RPS:      {baseline_metrics['rps']:.4f}")
	print(f"LogLoss:  {baseline_metrics['log_loss']:.4f}")
	
	# New model
	print("\n--- New Model (with improved Phase 4) ---")
	metrics = evaluate_model(model, data_test, device=DEVICE, verbose=True, task_type=TASK_TYPE)
	
	# Comparison
	print("\n" + "=" * 60)
	print("COMPARISON: NEW MODEL vs BASELINE")
	print("=" * 60)
	print(f"{'Metric':<15} {'Baseline':>12} {'New Model':>12} {'Diff':>12} {'Better?':>10}")
	print("-" * 68)
	
	for metric in ["accuracy", "brier", "rps", "log_loss"]:
		baseline_val = baseline_metrics[metric]
		model_val = metrics[metric]
		diff = model_val - baseline_val
		
		# For accuracy, higher is better; for others, lower is better
		if metric == "accuracy":
			is_better = "✓" if diff > 0 else "✗"
		else:
			is_better = "✓" if diff < 0 else "✗"
		
		sign = "+" if diff > 0 else ""
		print(f"{metric:<15} {baseline_val:>12.4f} {model_val:>12.4f} {sign}{diff:>11.4f} {is_better:>10}")
	
	# Try to compare with existing model
	print("\n" + "=" * 60)
	print("COMPARISON WITH EXISTING SAVED MODEL")
	print("=" * 60)
	
	try:
		from training.train_utils import load_existing_model
		import joblib
		
		model_path = Path("data/models/result_arch_tuned.pt")
		config_path = Path("data/models/result_architecture_config.json")
		scaler_path = Path("data/models/result_scaler_arch_tuned.joblib")
		
		if model_path.exists():
			existing_model, existing_config = load_existing_model(config_path, model_path, DEVICE, task_type=TASK_TYPE)
			
			# Check if features match
			old_feature_cols = existing_config.get("feature_cols") if existing_config else None
			if old_feature_cols and set(old_feature_cols) != set(feature_cols):
				print(f"Note: Feature sets differ ({len(old_feature_cols)} vs {len(feature_cols)} features)")
				old_scaler = joblib.load(scaler_path)
				data_test_old = prepare_data_result(df, old_feature_cols, [test_season], scaler=old_scaler)
			else:
				data_test_old = data_test
			
			existing_metrics = evaluate_model(existing_model, data_test_old, device=DEVICE, verbose=False, task_type=TASK_TYPE)
			
			print(f"\n{'Metric':<15} {'Existing':>12} {'New Model':>12} {'Diff':>12} {'Better?':>10}")
			print("-" * 68)
			
			for metric in ["accuracy", "brier", "rps", "log_loss"]:
				old_val = existing_metrics[metric]
				new_val = metrics[metric]
				diff = new_val - old_val
				
				if metric == "accuracy":
					is_better = "✓" if diff > 0 else "✗"
				else:
					is_better = "✓" if diff < 0 else "✗"
				
				sign = "+" if diff > 0 else ""
				print(f"{metric:<15} {old_val:>12.4f} {new_val:>12.4f} {sign}{diff:>11.4f} {is_better:>10}")
			
			# Overall assessment
			print("\n--- Overall Assessment ---")
			log_loss_improvement = (existing_metrics["log_loss"] - metrics["log_loss"]) / existing_metrics["log_loss"] * 100
			if metrics["log_loss"] < existing_metrics["log_loss"]:
				print(f"✓ NEW MODEL IS BETTER: {log_loss_improvement:.2f}% improvement in LogLoss")
			else:
				print(f"✗ Existing model is still better by {-log_loss_improvement:.2f}%")
		else:
			print("No existing model found at data/models/result_arch_tuned.pt")
	
	except Exception as e:
		print(f"Could not compare with existing model: {e}")
	
	print("\n" + "=" * 60)
	print("TEST COMPLETE")
	print("=" * 60)
	print(f"\nBest epoch found: {best_epoch}")
	print(f"Final model trained for exactly {best_epoch} epochs on all CV data")


if __name__ == "__main__":
	main()
