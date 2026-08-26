from __future__ import annotations

from typing import Tuple

import numpy as np


def generate_sequences(
    data: np.ndarray,
    history_steps: int,
    horizon_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sliding-window forecasting sequences for all sensors.

    Args:
        data: Array with shape (timesteps, nodes), already split and transformed.
        history_steps: Number of past steps used as model input.
        horizon_steps: Number of future steps used as prediction target.

    Returns:
        X: (samples, history_steps, nodes, 1)
        Y: (samples, horizon_steps, nodes, 1)
    """
    if data.ndim != 2:
        raise ValueError(f"Expected data shape (timesteps, nodes), got {data.shape}")

    timesteps, nodes = data.shape
    required = history_steps + horizon_steps
    if timesteps < required:
        raise ValueError(
            f"Not enough timesteps ({timesteps}) for history+horizon ({required})"
        )

    sample_count = timesteps - required + 1

    x = np.zeros((sample_count, history_steps, nodes), dtype=np.float32)
    y = np.zeros((sample_count, horizon_steps, nodes), dtype=np.float32)

    for i in range(sample_count):
        start = i
        hist_end = i + history_steps
        fut_end = hist_end + horizon_steps

        x[i] = data[start:hist_end]
        y[i] = data[hist_end:fut_end]

    # Add a feature channel for single-feature traffic input.
    return x[..., np.newaxis], y[..., np.newaxis]


def verify_sequence_shapes(
    x: np.ndarray,
    y: np.ndarray,
    history_steps: int,
    horizon_steps: int,
    expected_nodes: int,
) -> None:
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError(f"Expected 4D tensors, got x={x.ndim}D, y={y.ndim}D")

    if x.shape[1] != history_steps:
        raise ValueError(f"Expected history_steps={history_steps}, got {x.shape[1]}")
    if y.shape[1] != horizon_steps:
        raise ValueError(f"Expected horizon_steps={horizon_steps}, got {y.shape[1]}")

    if x.shape[2] != expected_nodes or y.shape[2] != expected_nodes:
        raise ValueError(
            f"Expected nodes={expected_nodes}, got x={x.shape[2]}, y={y.shape[2]}"
        )

    if x.shape[3] != 1 or y.shape[3] != 1:
        raise ValueError(
            f"Expected feature dimension 1, got x={x.shape[3]}, y={y.shape[3]}"
        )

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Sample count mismatch: x={x.shape[0]}, y={y.shape[0]}")
