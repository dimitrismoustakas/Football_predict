import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from training.models.neural_net import TrainConfig
from training.training_loop import _entropy_curriculum_weights, build_model, create_optimizer, run_train_epoch


class TrainingLoopTests(unittest.TestCase):
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
