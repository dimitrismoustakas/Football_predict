import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from training.models.neural_net import TrainConfig
from training.training_loop import (
	_entropy_curriculum_weights,
	build_model,
	create_optimizer,
	run_train_epoch,
	train_fixed_epochs,
	train_with_early_stopping,
)


class TrainingLoopTests(unittest.TestCase):
	def fit_synthetic_epochs(self, epochs, val_losses=None):
		config = TrainConfig(input_dim=1, lr=0.0016, weight_decay=0.0, epochs=epochs, patience=3)
		model = torch.nn.Linear(1, 1, bias=False)
		learning_rates = []

		def train_epoch(model, loader, optimizer, device, config):
			learning_rates.append(optimizer.param_groups[0]["lr"])
			optimizer.step()
			with torch.no_grad():
				model.weight.fill_(len(learning_rates))
			return 1.0

		validation = [(loss, [0.0] * 3, [0.0] * 3) for loss in val_losses or []]
		with (
			patch("training.training_loop.build_model", return_value=model),
			patch("training.training_loop.run_train_epoch", side_effect=train_epoch),
			patch("training.training_loop.run_validation_epoch", side_effect=validation),
		):
			if val_losses is None:
				model, history, best_loss = train_fixed_epochs(config, None, device=torch.device("cpu"), verbose=False)
			else:
				model, history, best_loss = train_with_early_stopping(
					config, None, object(), device=torch.device("cpu"), verbose=False,
				)
		return model.weight.item(), history, best_loss, learning_rates

	def test_learning_rate_is_constant_for_selection_and_fixed_epoch_budgets(self):
		for epochs in (1, 4):
			for use_validation in (False, True):
				with self.subTest(epochs=epochs, use_validation=use_validation):
					losses = [1.0 - 0.01 * epoch for epoch in range(epochs)] if use_validation else None
					_, _, _, learning_rates = self.fit_synthetic_epochs(epochs, losses)
					self.assertEqual(learning_rates, [0.0016] * epochs)

	def test_tiny_improvement_updates_checkpoint_but_not_patience(self):
		losses = [0.970000, 0.969950, 0.969950, 0.970200, 0.5]
		checkpoint_epoch, history, best_loss, _ = self.fit_synthetic_epochs(10, losses)
		best_epoch = history["val_loss"].index(min(history["val_loss"])) + 1

		self.assertEqual(history["val_loss"], losses[:4])
		self.assertEqual(best_loss, 0.969950)
		self.assertEqual(best_epoch, 2)
		self.assertEqual(checkpoint_epoch, best_epoch)

	def test_patience_compares_against_last_meaningful_improvement(self):
		losses = [1.0, 0.99994, 0.99988, 0.99987, 0.99989, 0.99991, 0.5]
		checkpoint_epoch, history, best_loss, _ = self.fit_synthetic_epochs(10, losses)

		self.assertEqual(history["val_loss"], losses[:6])
		self.assertEqual(checkpoint_epoch, 4)
		self.assertEqual(best_loss, 0.99987)

	def test_run_train_epoch_updates_weights(self):
		config = TrainConfig(
			input_dim=2,
			lr=1e-3,
			weight_decay=0.0,
			model_kwargs={
				"hidden_layers": [4],
				"num_leagues": 1,
			},
			batch_size=2,
			gate_mean_weight=0.0,
			gate_sat_weight=0.0,
		)
		device = torch.device("cpu")
		model = build_model(config, device)
		optimizer = create_optimizer(model, config)

		x = torch.tensor([[0.1, 0.2], [0.2, 0.1], [0.8, 0.4], [0.6, 0.3]], dtype=torch.float32)
		cat = torch.zeros(4, 3, dtype=torch.long)
		implied = torch.tensor(
			[
				[0.50, 0.30, 0.20],
				[0.45, 0.35, 0.20],
				[0.30, 0.25, 0.45],
				[0.35, 0.25, 0.40],
			],
			dtype=torch.float32,
		)
		y = torch.tensor([0, 1, 2, 2], dtype=torch.long)
		raw_margin = torch.ones(4, 1, dtype=torch.float32)
		loader = DataLoader(TensorDataset(x, cat, implied, y, raw_margin), batch_size=2, shuffle=False)

		before = model.backbone.final_layer.weight.detach().clone()
		loss = run_train_epoch(model, loader, optimizer, device, config)
		after = model.backbone.final_layer.weight.detach()

		self.assertTrue(torch.isfinite(torch.tensor(loss)))
		self.assertFalse(torch.allclose(before, after))

	def test_entropy_curriculum_center_only_favors_mid_entropy_markets(self):
		config = TrainConfig(
			input_dim=2,
			lr=1e-3,
			weight_decay=0.0,
			entropy_curriculum_mode="center_only",
			entropy_curriculum_strength=1.0,
		)
		implied_probs = torch.tensor(
			[
				[0.95, 0.03, 0.02],
				[0.80, 0.10, 0.10],
				[0.34, 0.33, 0.33],
			],
			dtype=torch.float32,
		)

		weights = _entropy_curriculum_weights(implied_probs, batch_idx=0, total_batches=1, config=config)

		self.assertIsNotNone(weights)
		self.assertAlmostEqual(float(weights.mean().item()), 1.0, places=6)
		self.assertGreater(float(weights[1].item()), float(weights[0].item()))
		self.assertGreater(float(weights[1].item()), float(weights[2].item()))


if __name__ == "__main__":
	unittest.main()
