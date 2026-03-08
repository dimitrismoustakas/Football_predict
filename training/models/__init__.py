"""Neural network models and loss functions."""

from typing import Callable

from training.models.neural_net import (
	CategoricalConfig,
	CategoricalEmbedder,
	GatedResidualModel,
	MLPWithHiddenAccess,
	TrainConfig,
	_log_softmax_from_implied,
	gated_loss,
)

MODEL_BUILDERS: dict[str, Callable[..., object]] = {
	"gated_residual": GatedResidualModel,
}


def build_model(model_name: str, **kwargs):
	"""Build a registered model by name."""

	if model_name not in MODEL_BUILDERS:
		raise ValueError(f"Unsupported model_name={model_name}")
	return MODEL_BUILDERS[model_name](**kwargs)

__all__ = [
	"CategoricalConfig",
	"CategoricalEmbedder",
	"build_model",
	"GatedResidualModel",
	"MLPWithHiddenAccess",
	"TrainConfig",
	"_log_softmax_from_implied",
	"gated_loss",
]
