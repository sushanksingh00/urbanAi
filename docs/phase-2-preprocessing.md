# Phase 2 — Data Preprocessing

## 1. Objective

The objective of Phase 2 is to transform the raw METR-LA traffic data into reproducible training, validation, and testing sequences suitable for spatio-temporal graph neural network experiments.

The preprocessing pipeline performs:

1. Configuration loading
2. Chronological splitting
3. Training-only scaler fitting
4. Scaling
5. Sequence generation
6. Data-quality verification
7. Metadata generation
8. Processed-data storage

## 2. Input Data

Raw traffic data:

`data/raw/METR-LA/METR-LA.h5`

Dataset:

- 34,272 timestamps
- 207 sensors
- 5-minute sampling interval

## 3. Preprocessing Configuration

Configuration file:

`configs/preprocessing_config.json`

Current configuration:

```json
{
  "dataset_name": "METR-LA",
  "sampling_interval_minutes": 5,
  "history_steps": 12,
  "horizon_steps": 3,
  "train_ratio": 0.7,
  "val_ratio": 0.15,
  "test_ratio": 0.15,
  "scaling_method": "standard"
}
```

The model input contains 12 × 5 = 60 minutes of historical observations.

The prediction horizon is 3 × 5 = 15 minutes.

## 4. Chronological Data Split

The dataset is split chronologically rather than randomly.

### Training

- Samples: 23,990
- Start: 2012-03-01 00:00:00
- End: 2012-05-23 07:05:00

### Validation

- Samples: 5,140
- Start: 2012-05-23 07:10:00
- End: 2012-06-10 03:25:00

### Test

- Samples: 5,142
- Start: 2012-06-10 03:30:00
- End: 2012-06-27 23:55:00

The split sizes sum to the complete dataset:

23,990 + 5,140 + 5,142 = 34,272.

The split boundaries are strictly chronological.

## 5. Zero-Value Handling

Phase 1 identified substantial zero-valued observations:

- Total zero values: 575,302
- Overall zero percentage: 8.109%
- Completely-zero timestamps: 2,148
- Completely-zero periods: 97
- Longest completely-zero period: 31 hours 45 minutes
- Highest sensor-level zero percentage: 20.1039% for sensor 718076

The baseline preprocessing pipeline preserves zero values exactly as provided by the raw dataset.

Zero values are not automatically:

- Converted to NaN
- Imputed
- Interpolated
- Removed

This avoids introducing an unsupported imputation assumption into the baseline experiment.

The substantial presence of zero-valued observations is documented as a dataset-quality limitation. Future experiments may investigate missing-data-aware preprocessing.

## 6. Scaling

The pipeline uses `StandardScaler`.

The scaler is fitted **only on the training split**.

The same fitted scaler is then used to transform:

- Training data
- Validation data
- Test data

This prevents validation/test information from influencing scaling parameters.

The scaler operates across the 207 sensor features.

## 7. Sequence Generation

Configuration:

- History: 12 timesteps
- Horizon: 3 timesteps

Therefore:

- X contains 12 historical observations
- Y contains 3 future observations

Each timestep contains measurements from 207 sensors.

### Training

```text
X: (23976, 12, 207, 1)
Y: (23976, 3, 207, 1)
```

### Validation

```text
X: (5126, 12, 207, 1)
Y: (5126, 3, 207, 1)
```

### Test

```text
X: (5128, 12, 207, 1)
Y: (5128, 3, 207, 1)
```

Sequence counts follow:

`number_of_sequences = number_of_timesteps - history_steps - horizon_steps + 1`

## 8. Data Integrity Verification

The preprocessing pipeline verifies:

- Shape
- Data type
- NaN count
- Infinite-value count
- Chronological ordering
- Sequence shapes

Results:

```text
Train:
NaN = 0
Inf  = 0

Validation:
NaN = 0
Inf  = 0

Test:
NaN = 0
Inf  = 0
```

All three splits passed the chronological-order check.

## 9. Processed Artifacts

Processed data is stored under:

`data/processed/METR-LA/`

Sequence files:

```text
train/sequences.npz
validation/sequences.npz
test/sequences.npz
```

Metadata:

```text
metadata/metadata.json
metadata/preprocessing_config_used.json
metadata/run_summary.json
metadata/verification.json
```

## 10. Source Code

Preprocessing implementation:

```text
src/data/preprocessing.py
src/data/sequence.py
```

Main notebook:

```text
notebooks/02_preprocessing.ipynb
```

## 11. Reproducibility

The preprocessing configuration is stored separately in:

`configs/preprocessing_config.json`

The intended reproducible pipeline is:

```text
Raw data
   ↓
Configuration
   ↓
Chronological split
   ↓
Training-only scaling
   ↓
Sequence generation
   ↓
Verification
   ↓
Processed artifacts
```

The raw dataset is not modified.

## 12. Known Limitations

The raw dataset contains substantial zero-valued observations. Although the baseline pipeline preserves these values, some may represent missing or unavailable sensor measurements.

This issue is explicitly documented rather than silently corrected.

Future experiments may investigate:

- Missing-value-aware preprocessing
- Imputation strategies
- Sensor-specific missingness
- Robust training under missing observations

These are outside the baseline Phase 2 pipeline.

## 13. Phase Status

**Phase 2 — PREPROCESSING IMPLEMENTED AND VERIFIED**

The preprocessing pipeline has generated train, validation, and test sequences and passed the current data-integrity checks.

Final project closure requires the final notebook execution, repository cleanup, documentation review, and Git checkpoint.
