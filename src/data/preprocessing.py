from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass(frozen=True)
class PreprocessingConfig:
    dataset_name: str = "METR-LA"
    sampling_interval_minutes: int = 5
    history_steps: int = 12
    horizon_steps: int = 3
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    scaling_method: str = "standard"


def load_metr_la_h5(h5_path: Path) -> pd.DataFrame:
    """Load METR-LA HDF5 data into a timestamp-indexed DataFrame."""
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        df_group = f["df"]
        sensor_ids = [sensor.decode("utf-8") for sensor in df_group["axis0"][:]]
        timestamps = pd.to_datetime(df_group["axis1"][:])
        values = df_group["block0_values"][:]

    traffic_df = pd.DataFrame(values, index=timestamps, columns=sensor_ids)
    traffic_df.index.name = "timestamp"
    return traffic_df


def chronological_split(
    traffic_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, pd.DataFrame]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    n = len(traffic_df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = traffic_df.iloc[:train_end].copy()
    val_df = traffic_df.iloc[train_end:val_end].copy()
    test_df = traffic_df.iloc[val_end:].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("One or more temporal splits are empty.")

    return {"train": train_df, "validation": val_df, "test": test_df}


def build_scaler(method: str) -> StandardScaler | MinMaxScaler:
    method = method.lower().strip()
    if method == "standard":
        return StandardScaler(with_mean=True, with_std=True)
    if method == "minmax":
        return MinMaxScaler(feature_range=(0.0, 1.0))
    raise ValueError(f"Unsupported scaling method: {method}")


def fit_scaler(train_df: pd.DataFrame, method: str) -> StandardScaler | MinMaxScaler:
    scaler = build_scaler(method)
    scaler.fit(train_df.values)
    return scaler


def transform_split(
    split_frames: dict[str, pd.DataFrame],
    scaler: StandardScaler | MinMaxScaler,
) -> dict[str, np.ndarray]:
    transformed: dict[str, np.ndarray] = {}
    for split_name, frame in split_frames.items():
        transformed[split_name] = scaler.transform(frame.values).astype(np.float32)
    return transformed


def verify_array_health(name: str, arr: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
        "is_chronological": True,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def scaler_to_metadata(scaler: StandardScaler | MinMaxScaler) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "scaler_class": scaler.__class__.__name__,
        "n_features_in": int(scaler.n_features_in_),
    }

    for attr in ("mean_", "scale_", "var_", "data_min_", "data_max_", "data_range_"):
        if hasattr(scaler, attr):
            value = getattr(scaler, attr)
            metadata[attr] = np.asarray(value, dtype=np.float64).tolist()

    return metadata


def build_metadata(
    config: PreprocessingConfig,
    traffic_df: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    sequence_shapes: dict[str, dict[str, list[int]]],
    scaling_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": config.dataset_name,
        "source": "data/raw/METR-LA/METR-LA.h5",
        "number_of_nodes": int(traffic_df.shape[1]),
        "number_of_timestamps": int(traffic_df.shape[0]),
        "sampling_interval_minutes": config.sampling_interval_minutes,
        "feature_count": 1,
        "train_split": {
            "ratio": config.train_ratio,
            "start": split_frames["train"].index.min().isoformat(),
            "end": split_frames["train"].index.max().isoformat(),
            "timesteps": int(len(split_frames["train"])),
        },
        "validation_split": {
            "ratio": config.val_ratio,
            "start": split_frames["validation"].index.min().isoformat(),
            "end": split_frames["validation"].index.max().isoformat(),
            "timesteps": int(len(split_frames["validation"])),
        },
        "test_split": {
            "ratio": config.test_ratio,
            "start": split_frames["test"].index.min().isoformat(),
            "end": split_frames["test"].index.max().isoformat(),
            "timesteps": int(len(split_frames["test"])),
        },
        "history_steps": config.history_steps,
        "horizon_steps": config.horizon_steps,
        "scaling_method": config.scaling_method,
        "sequence_shapes": sequence_shapes,
        "scaler": scaling_info,
        "preprocessing_date_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing_version": "phase-2-v1",
    }


def save_config(path: Path, config: PreprocessingConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
