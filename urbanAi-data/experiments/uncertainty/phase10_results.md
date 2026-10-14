# Phase 10 — Uncertainty Analysis

## 1. Objective

This phase evaluates predictive uncertainty for the proposed Multi-Scale STGCN using Monte Carlo Dropout and examines whether estimated uncertainty is associated with prediction error.

The analysis uses the verified permanent checkpoint:

`experiments/proposed/best_model.pt`

The checkpoint was reconstructed and verified against the permanently saved test predictions with zero numerical difference.

---

## 2. Experimental Setup

- Dataset: METR-LA
- Nodes: 207
- Historical input: 12 time steps
- Forecast horizon: 3 time steps
- Forecast intervals: 5, 10, and 15 minutes
- MC-Dropout samples: 30
- Dropout probability: 0.2
- Test samples: 6841
- Uncertainty measure: predictive standard deviation across MC-Dropout samples

---

## 3. Checkpoint Verification

The reconstructed Multi-Scale STGCN produced exactly the same deterministic predictions as the permanently saved proposed-model predictions.

- Maximum absolute difference: 0.0
- Mean absolute difference: 0.0
- RMSE difference: 0.0

This confirms that the uncertainty analysis uses the same trained model associated with the original proposed-model experiment.

---

## 4. Raw MC-Dropout Uncertainty

### Overall uncertainty

| Statistic | Value |
|---|---:|
| Mean | 0.053441 |
| Median | 0.029699 |
| Std | 0.050516 |
| Minimum | 0.004953 |
| P90 | 0.131407 |
| P95 | 0.156093 |
| P99 | 0.217845 |
| Maximum | 0.651836 |

---

## 5. Error–Uncertainty Relationship

The relationship between predictive uncertainty and absolute prediction error was evaluated using Pearson and Spearman correlation.

| Measure | Result |
|---|---:|
| Pearson correlation | 0.406993 |
| Spearman correlation | 0.452587 |

The positive correlations indicate that higher estimated uncertainty tends to be associated with larger prediction errors.

This should be interpreted as an association rather than evidence that uncertainty directly causes prediction error.

---

## 6. Uncertainty by Forecast Horizon

| Horizon | Mean uncertainty | Mean absolute error | Pearson | Spearman |
|---|---:|---:|---:|---:|
| 5 min | 0.055520 | 0.317099 | 0.417797 | 0.440943 |\n| 10 min | 0.052898 | 0.348004 | 0.413653 | 0.456531 |\n| 15 min | 0.051905 | 0.369878 | 0.402942 | 0.462979 |\n
---

## 7. Raw 95% Predictive Intervals

Raw intervals were constructed as:

`predictive mean ± 1.96 × predictive standard deviation`

The nominal coverage was 95%.

Observed test coverage:

**19.3860%**

### Coverage by horizon

| Horizon | Coverage |
|---|---:|
| 5 min | 21.8874% |\n| 10 min | 18.8236% |\n| 15 min | 17.4470% |\n
The raw MC-Dropout intervals substantially under-cover the nominal 95% target.

---

## 8. Raw Interval Width

| Statistic | Value |
|---|---:|
| Mean | 0.209489 |
| Median | 0.116419 |
| Std | 0.198023 |
| P90 | 0.515113 |
| P95 | 0.611886 |
| P99 | 0.853954 |
| Maximum | 2.555196 |

---

## 9. Validation-Based Calibration

A scalar uncertainty multiplier was selected using the validation set only.

The calibration target was 95% interval coverage.

- Calibration factor: **22.50**
- Validation coverage: **94.9998%**

The calibration factor was frozen before evaluating the test set.

---

## 10. Calibrated Test Coverage

Overall calibrated test coverage:

**94.9462%**

### Coverage by horizon

| Horizon | Coverage |
|---|---:|
| 5 min | 95.7234% |\n| 10 min | 94.9322% |\n| 15 min | 94.1829% |\n
---

## 11. Calibrated Interval Width

| Statistic | Value |
|---|---:|
| Mean | 2.404850 |
| Median | 1.336439 |
| Std | 2.273218 |
| P90 | 5.913292 |
| P95 | 7.024194 |
| P99 | 9.803045 |
| Maximum | 29.332605 |

---

## 12. Main Findings

1. MC-Dropout uncertainty showed a positive association with absolute prediction error.
2. Prediction error increased across the 5-, 10-, and 15-minute horizons.
3. Raw MC-Dropout intervals constructed using ±1.96 standard deviations were substantially under-covering.
4. A calibration factor selected exclusively from the validation set increased test-set coverage to approximately the nominal 95% level.
5. Calibration substantially increased interval width, demonstrating a coverage–sharpness trade-off.
6. The calibrated result should be described specifically for this METR-LA experiment and should not be generalized as universal calibration.

---

## 13. Limitations

- The uncertainty analysis uses MC Dropout with a fixed dropout probability of 0.2.
- The raw predictive intervals were not calibrated.
- The scalar calibration factor was estimated on the validation split of METR-LA.
- Calibration was evaluated on one held-out METR-LA test split.
- The analysis does not establish that the intervals would achieve 95% coverage on another dataset or deployment environment.
- The calibrated intervals are substantially wider than the raw MC-Dropout intervals.

---

## 14. Reproducibility

All uncertainty outputs are stored under:

`experiments/uncertainty/`

including MC predictions, predictive statistics, uncertainty/error tables, coverage results, calibration results, interval widths, metadata, and figures.

