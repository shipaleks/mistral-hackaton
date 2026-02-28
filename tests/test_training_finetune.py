from __future__ import annotations

from argparse import Namespace

from training.finetune import _build_hyperparameters


def test_build_hyperparameters_defaults() -> None:
    args = Namespace(
        training_steps=120,
        learning_rate=0.0,
        epochs=0.0,
        seq_len=0,
        fim_ratio=-1.0,
    )
    params = _build_hyperparameters(args)
    assert params == {"training_steps": 120}


def test_build_hyperparameters_with_overrides() -> None:
    args = Namespace(
        training_steps=100,
        learning_rate=0.0002,
        epochs=2.0,
        seq_len=4096,
        fim_ratio=0.5,
    )
    params = _build_hyperparameters(args)
    assert params["training_steps"] == 100
    assert params["learning_rate"] == 0.0002
    assert params["epochs"] == 2.0
    assert params["seq_len"] == 4096
    assert params["fim_ratio"] == 0.5
