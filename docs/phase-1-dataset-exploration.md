# Phase 1 — Dataset Exploration

## 1. Objective

The objective of Phase 1 was to understand the METR-LA traffic dataset before implementing the preprocessing and forecasting pipeline.

The exploration focused on:

- Dataset dimensions
- Sensor count
- Temporal coverage
- Sampling frequency
- Missing values
- Zero-valued observations
- Descriptive statistics
- Temporal patterns
- Sensor relationships
- Road-network adjacency information

## 2. Dataset

Dataset: METR-LA

Raw traffic data:
`data/raw/METR-LA/METR-LA.h5`

Adjacency information:
`data/raw/METR-LA/adj_METR-LA.pkl`

### Dataset dimensions

- Number of timestamps: 34,272
- Number of sensors: 207
- Total observations: 7,094,304
- Raw data type: float64
- Sampling interval: 5 minutes

## 3. Temporal Coverage

- Start: 2012-03-01 00:00:00
- End: 2012-06-27 23:55:00

All consecutive timestamps are separated by exactly five minutes.

## 4. Missing Values

- Missing values: 0
- Missing percentage: 0.0%

No explicit NaN values are present in the raw dataset. Zero-valued observations were investigated separately.

## 5. Descriptive Statistics

Sensor measurements vary substantially across the network. For example, sensor `773869` has:

- Mean: approximately 54.63
- Standard deviation: approximately 22.62
- Minimum: 0
- Median: approximately 64.89
- Maximum: 70

## 6. Zero-Value Analysis

The dataset contains substantial zero-valued observations:

- Total zero values: 575,302
- Overall zero percentage: 8.109%
- Completely-zero timestamps: 2,148
- Completely-zero periods: 97

A completely-zero timestamp means all 207 sensors simultaneously contain a value of zero.

The longest completely-zero period was:

- Start: 2012-03-26 08:25:00
- End: 2012-03-27 16:05:00
- Duration: 1,905 minutes
- Timesteps: 381

This is approximately 31 hours and 45 minutes.

The sensor with the highest observed zero percentage was:

- Sensor: 718076
- Zero count: 6,890
- Zero percentage: 20.1039%

These observations indicate that zero-valued measurements are a significant characteristic of the raw dataset.

For the baseline preprocessing pipeline, zero values are preserved as provided in the raw dataset. They are not automatically converted to NaN or imputed. Their presence is documented as a dataset-quality limitation.

## 7. Temporal Exploration

Temporal analysis investigated:

- Time-of-day patterns
- Day-of-week patterns
- Daily traffic behaviour
- Weekly traffic behaviour

The corresponding visualizations and analyses are contained in `notebooks/01_dataset_exploration.ipynb`.

## 8. Sensor Correlation

Correlation analysis was performed across the 207 sensors to investigate relationships between traffic measurements at different locations.

This provides exploratory context for later graph construction. Correlation analysis does not replace physical road-network adjacency.

## 9. Graph Data

The METR-LA adjacency file was inspected:

`data/raw/METR-LA/adj_METR-LA.pkl`

The final graph representation will be constructed formally in Phase 3.

## 10. Main Findings

1. METR-LA contains 207 sensors.
2. The dataset contains 34,272 timestamps.
3. Measurements are sampled every five minutes.
4. There are no explicit NaN values.
5. Zero-valued measurements are common.
6. Some timestamps contain zero values across all 207 sensors.
7. The longest completely-zero period lasts approximately 31 hours and 45 minutes.
8. Sensor-level zero rates vary considerably.
9. Temporal patterns exist across the dataset.
10. Sensor relationships provide useful exploratory context for graph-based modeling.

## 11. Output

Primary Phase 1 notebook:

`notebooks/01_dataset_exploration.ipynb`

The raw dataset remains unchanged.

## 12. Phase Status

**Phase 1 — COMPLETE**
