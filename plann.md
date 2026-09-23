# UrbanFlow AI — Revised Development Plan

## Goal

UrbanFlow AI is an intelligent traffic forecasting research system using Spatio-Temporal Graph Neural Networks (STGNNs).

**Immediate priority:** build and verify the AI/ML system for academic evaluation, project marks, research, and resume presentation.

**Current status:** the core ML/research experiments have been implemented and evaluated on METR-LA. The remaining research-core work is literature consolidation, error analysis, reproducibility cleanup, and the inference engine.

**Later:** build the full application around the validated ML system.

The application layer remains postponed:
- FastAPI
- PostgreSQL
- React
- Docker
- Deployment

---

# Development Order

```text
Dataset
  ↓
Exploration
  ↓
Preprocessing
  ↓
Baselines
  ↓
STGCN
  ↓
Evaluation
  ↓
Literature Review
  ↓
Proposed Research Model
  ↓
Ablation + Error Analysis
  ↓
Inference Engine
  ↓
========== AI/RESEARCH CORE COMPLETE ==========
  ↓
FastAPI
  ↓
PostgreSQL
  ↓
React Dashboard
  ↓
Docker
  ↓
Deployment
  ↓
Paper + Presentation
```

---

# 1. Mac + GitHub + Colab Setup

## Source of truth

The permanent source of truth is the local Mac project and GitHub repository.

Example:

```text
/Users/sushank/Documents/GitHub/urbanAi
```

Antigravity edits the local project.

GitHub stores the source.

Colab is a temporary GPU execution environment.

```text
Mac / Antigravity
       ↓
     GitHub
       ↓
     Colab
       ↓
   GPU execution
```

If Colab disconnects or resets:

```text
New Colab runtime
      ↓
git clone / git pull
      ↓
install requirements
      ↓
restore/download dataset
      ↓
continue
```

Do not depend on manually uploaded source files surviving a Colab reset.

Never hardcode Mac paths inside notebooks or source code.

Use centralized project paths.

---

# 2. Technology Scope

## Current ML/Research Stack

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter
- Google Colab GPU
- Git
- GitHub

## Later Application Stack

- FastAPI
- Pydantic
- PostgreSQL
- React
- TypeScript
- Tailwind CSS
- Recharts/Plotly
- Map visualization
- Docker
- Docker Compose
- Cloud deployment

Do not implement the application layer during the initial ML/research phase.

---

# 3. Dataset

Primary dataset:

```text
METR-LA
```

Expected graph size:

```text
207 traffic sensors/nodes
```

Raw traffic:

```text
(time, nodes)
```

Model representation:

```text
(time, nodes, features)
```

For this project:

```text
features = 1
```

The project uses the METR-LA adjacency graph with 207 sensors. The graph is represented by a 207 × 207 adjacency matrix and is row-normalized before graph convolution.

---

# 4. Canonical Prediction Task

This is a **hard project invariant**.

Do NOT change the base task to 12 → 12.

Initial task:

```text
Previous 60 minutes
        ↓
      Model
        ↓
Next 15 minutes
```

With 5-minute intervals:

```text
history_steps = 12
horizon_steps = 3
```

## Canonical tensor contract

Raw:

```text
(T, 207)
```

Model-ready:

```text
(T, 207, 1)
```

Input:

```text
X = (N, 12, 207, 1)
```

Target:

```text
Y = (N, 3, 207, 1)
```

DataLoader:

```text
batch_x = (B, 12, 207, 1)
batch_y = (B, 3, 207, 1)
```

Every baseline and neural model follows this contract.

Output:

```text
prediction = (B, 3, 207, 1)
```

Sequence generation:

```python
X[i] = traffic[t : t + 12]
Y[i] = traffic[t + 12 : t + 15]
```

Therefore:

```text
X[i] = (12, 207, 1)
Y[i] = (3, 207, 1)
```

Do NOT:
- change horizon to 12 just to fix errors
- add notebook-specific unsqueeze/reshape hacks
- squeeze away the feature dimension
- modify models merely to accept broken preprocessing output

Fix the source preprocessing pipeline if the contract is violated.

---

# 5. Project Structure

```text
urbanAi/
├── README.md
├── PLAN.md
├── requirements.txt
├── .gitignore
├── .env.example
├── project_paths.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_stgcn.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_proposed_model.ipynb
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── sequence.py
│   │   ├── dataset.py
│   │   ├── graph.py
│   │   └── dataloader.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── lstm.py
│   │   ├── stgcn.py
│   │   └── proposed.py
│   ├── training/
│   │   ├── train.py
│   │   ├── config.py
│   │   └── checkpoints.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluate.py
│   │   └── visualization.py
│   └── inference/
│       └── predictor.py
│
├── experiments/
│   ├── baseline/
│   ├── stgcn/
│   ├── proposed/
│   └── comparison/
│
└── docs/
    ├── research/
    ├── experiments/
    └── architecture/
```

Later:

```text
backend/
frontend/
deployment/
```

---

# 6. Phase 0 — Environment + Reproducibility

Tasks:

- [x] Clean/reinitialize repository
- [ ] Push/verify repository on GitHub
- [x] Create requirements.txt
- [x] Configure .gitignore
- [x] Configure centralized project paths
- [x] Test local Python
- [x] Test Colab runtime
- [x] Test PyTorch
- [x] Verify CUDA/GPU
- [ ] Verify fresh Colab → GitHub → execution workflow
- [ ] Verify reproducibility from a clean Colab runtime

Definition of done:

A fresh Colab runtime can obtain the project from GitHub, install dependencies, obtain/restore the dataset, and execute the core pipeline without manually reconstructing source files.

---

# 7. Phase 1 — Dataset Exploration

Notebook:

```text
notebooks/01_dataset_exploration.ipynb
```

Status: **Completed**

Tasks:

- [x] Obtain METR-LA
- [x] Store raw data correctly
- [x] Document dataset source
- [x] Identify sensors
- [x] Identify sampling frequency
- [x] Identify traffic variable
- [x] Inspect timestamps
- [x] Inspect missing values
- [x] Inspect abnormal values
- [x] Inspect sensor metadata
- [x] Inspect adjacency matrix
- [x] Visualize traffic distributions
- [x] Visualize traffic over time
- [x] Document dataset statistics

Do not start FastAPI, React, PostgreSQL, Docker, or research extensions here.

---

# 8. Phase 2 — Data Preprocessing

Goal: convert raw METR-LA into model-ready sequences.

Notebook:

```text
notebooks/02_data_preprocessing.ipynb
```

Status: **Completed**

Files:

```text
src/data/preprocessing.py
src/data/sequence.py
src/data/dataset.py
src/data/graph.py
```

Implemented:

- [x] Load raw traffic
- [x] Validate data
- [x] Chronological train/validation/test split
- [x] Fit scaler on training data only
- [x] Normalize values
- [x] Process adjacency matrix
- [x] Create sliding windows
- [x] Create model-ready tensors
- [x] Verify tensor dimensions
- [x] Save processed sequences
- [x] Save scaler
- [x] Save preprocessing metadata
- [x] Copy processed artifacts to persistent Drive storage

Saved processed data:

```text
data/processed/METR-LA/
├── train.npz
├── val.npz
├── test.npz
├── scaler.npz
├── scaler_metadata.json
└── preprocessing_metadata.json
```

Verified training shapes:

```text
X = (23976, 12, 207, 1)
Y = (23976, 3, 207, 1)
```

The test set contains 6,841 prediction samples.

Definition of done:

```text
X = (N, 12, 207, 1)
Y = (N, 3, 207, 1)
```

and DataLoader batches follow:

```text
batch_x = (B, 12, 207, 1)
batch_y = (B, 3, 207, 1)
```

---

# 9. Phase 3 — Baseline Models

Status: **Completed**

Implemented/evaluated:

1. Historical Average
2. Last Value
3. Linear Regression
4. LSTM

Notebook:

```text
notebooks/03_baselines.ipynb
```

All models use the canonical 12 → 3 prediction task.

## Overall original-scale test results

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Historical Average | 15.198922 | 22.213470 | 27.072224% |
| Last Value | 3.511726 | 8.586182 | 8.180886% |
| Linear Regression | 3.918365 | 8.305223 | 8.260439% |
| LSTM | 6.240199 | 11.601593 | 12.581139% |

Artifacts:

```text
experiments/baseline/
├── baseline_results.csv
├── last_value_predictions.npy
├── historical_average_predictions.npy
├── linear_regression_predictions.npy
├── lstm_predictions.npy
└── test_targets.npy
```

Important research note:

The measured results do **not** show the neural models outperforming every simpler baseline on every metric. The final paper must report this directly rather than selecting or modifying results to create a favorable narrative.

---

# 10. Phase 4 — STGCN

Status: **Completed**

Goal: implement an established Spatio-Temporal Graph Neural Network.

Architecture:

```text
Input
  ↓
Temporal Convolution
  ↓
Graph Convolution
  ↓
Temporal Convolution
  ↓
STGCN Block
  ↓
STGCN Block
  ↓
Forecasting Head
  ↓
3-step future prediction
```

Graph:

```text
207 × 207
```

The supplied METR-LA graph is asymmetric, so row normalization is used.

The graph was verified to have:

```text
shape: (207, 207)
dtype: float32
finite values: True
non-zero entries: 1722
density: ~0.04019
diagonal self-connections: present
symmetric: False
```

Do not add another identity matrix to this graph during preprocessing.

Training configuration:

```text
epochs = 20
learning rate = 1e-3
weight decay = 1e-4
loss = MSE
gradient clipping = 5
hidden channels = 32
```

Verified forward contract:

```text
Input  = (64, 12, 207, 1)
Output = (64, 3, 207, 1)
```

Overall original-scale test results:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| STGCN | 6.263297 | 11.017338 | 13.161572% |

Horizon results:

| Horizon | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| 5 min | 5.875838 | 10.208804 | 12.432792% |
| 10 min | 6.270047 | 11.027397 | 13.169551% |
| 15 min | 6.644026 | 11.761041 | 13.882375% |

Artifacts:

```text
experiments/stgcn/
├── test_predictions.npy
├── test_targets.npy
├── overall_results.csv
├── horizon_results.csv
├── sensor_mae.csv
├── best_model.pt
└── figures/
```

---

# 11. Phase 5 — Unified Evaluation

Status: **Mostly completed**

Notebook:

```text
notebooks/05_evaluation.ipynb
```

Implemented:

- [x] MAE
- [x] RMSE
- [x] MAPE
- [x] Original-scale inverse transformation
- [x] Overall model comparison
- [x] Horizon-wise evaluation
- [x] Sensor-level MAE analysis
- [x] Prediction artifact storage
- [x] Comparison CSVs
- [ ] WAPE
- [ ] R²
- [ ] Final unified visualization set
- [ ] Complete error-distribution analysis
- [ ] Complete difficult-condition analysis

Important:

MAPE is already implemented with an epsilon mask. Because traffic values can be close to zero, MAPE should be interpreted carefully.

Current main comparison artifacts:

```text
experiments/comparison/
├── model_comparison_original_scale.csv
├── horizon_comparison_original_scale.csv
├── final_model_comparison.csv
└── final_horizon_comparison.csv
```

---

# 12. Phase 6 — Literature Review

Status: **Partially completed / needs final consolidation**

This phase must support the research motivation and proposed contribution.

Study:

- STGCN
- DCRNN
- Graph WaveNet
- ASTGCN
- MTGNN
- Dynamic graph methods
- Weather-aware forecasting
- Event-aware forecasting
- Accident-aware forecasting
- Explainable GNNs
- Lightweight GNNs

Tasks:

- [ ] Collect 15–25 relevant papers
- [ ] Read abstracts
- [ ] Read methodology
- [ ] Record datasets
- [ ] Record models
- [ ] Record metrics
- [ ] Record limitations
- [ ] Record future work
- [ ] Create literature comparison table
- [ ] Identify recurring limitations
- [ ] Finalize research direction
- [ ] Document research gap with citations

Deliverable:

```text
docs/research/literature_review.md
```

Do not claim broad novelty until this phase is finalized.

---

# 13. Phase 7 — Proposed Research Model

Status: **Implemented and evaluated**

The current proposed architecture is a **Multi-Scale STGCN** using parallel temporal convolutions with kernel sizes:

```text
(3, 5)
```

Architecture:

```text
Input
  ↓
Multi-Scale Temporal Convolution
  ├── Kernel 3
  └── Kernel 5
  ↓
Graph Convolution
  ↓
Multi-Scale Temporal Convolution
  ├── Kernel 3
  └── Kernel 5
  ↓
Second Proposed Block
  ↓
Forecasting Head
  ↓
3-step prediction
```

Configuration:

```text
input_steps = 12
output_steps = 3
nodes = 207
input_channels = 1
hidden_channels = 32
temporal_kernels = [3, 5]
dropout = 0.2
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-4
epochs = 20
gradient clipping = 5
```

Verified forward contract:

```text
Input  = (64, 12, 207, 1)
Output = (64, 3, 207, 1)
```

Overall original-scale test results from the main proposed-model run:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Proposed MultiScale STGCN | 6.314953 | 10.947598 | 13.102770% |

Horizon results:

| Horizon | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| 5 min | 5.831080 | 10.124292 | 12.247165% |
| 10 min | 6.360851 | 10.973485 | 13.156385% |
| 15 min | 6.752928 | 11.688925 | 13.904788% |

Interpretation rule:

Do not describe the proposed model as universally superior. Its measured overall MAE is slightly higher than the original STGCN run, while its RMSE and MAPE are slightly lower. These differences should be reported as measured.

Artifacts:

```text
experiments/proposed/
├── test_predictions.npy
├── test_targets.npy
├── history.csv
├── best_model.pt
└── ablation/
```

---

# 14. Phase 8 — Ablation Study

Status: **Completed**

Purpose:

Test whether multiple temporal receptive fields contribute useful performance while keeping the rest of the architecture and training protocol controlled.

Configurations:

```text
STGCN_3x3       → kernels (3,)
STGCN_5x5       → kernels (5,)
MultiScale_3_5  → kernels (3,5)
```

Controlled ablation overall original-scale results:

| Configuration | Kernels | MAE | RMSE | MAPE |
|---|---|---:|---:|---:|
| STGCN_3x3 | (3,) | 6.385260 | 10.942251 | 13.104069% |
| STGCN_5x5 | (5,) | 6.277655 | 11.018918 | 13.299939% |
| MultiScale_3_5 | (3,5) | 6.224726 | 10.991349 | 12.920200% |

Controlled horizon results:

| Configuration | Horizon | MAE | RMSE | MAPE |
|---|---|---:|---:|---:|
| STGCN_3x3 | 5 min | 6.029098 | 10.131438 | 12.424157% |
| STGCN_3x3 | 10 min | 6.364127 | 10.973769 | 13.034259% |
| STGCN_3x3 | 15 min | 6.762556 | 11.667436 | 13.853786% |
| STGCN_5x5 | 5 min | 5.846546 | 10.209141 | 12.503773% |
| STGCN_5x5 | 10 min | 6.291490 | 11.038233 | 13.322553% |
| STGCN_5x5 | 15 min | 6.694927 | 11.755021 | 14.073512% |
| MultiScale_3_5 | 5 min | 5.804359 | 10.144118 | 12.180582% |
| MultiScale_3_5 | 10 min | 6.276794 | 11.036702 | 12.948381% |
| MultiScale_3_5 | 15 min | 6.593026 | 11.735311 | 13.631645% |

The controlled multi-scale configuration has the lowest MAE and MAPE among these three ablation configurations. The `(3,)` configuration has the lowest RMSE.

Important experimental integrity note:

The controlled `(3,5)` ablation run is separate from the earlier main proposed-model training run. The main proposed-model result and controlled ablation result must not be silently treated as the same training run.

Artifacts:

```text
experiments/proposed/ablation/
├── ablation_plan.csv
├── ablation_results.csv
├── ablation_horizon_results.csv
├── STGCN_3x3/
├── STGCN_5x5/
├── MultiScale_3_5/
└── temporal_kernel_ablation.png
```

---

# 15. Phase 9 — Error Analysis

Status: **Next research task**

Goal:

Understand where the models succeed and fail instead of only reporting aggregate metrics.

Analyze:

- [ ] Rush hours
- [ ] High congestion
- [ ] Low congestion
- [x] Different sensors
- [x] Different prediction horizons
- [ ] Sudden traffic changes
- [ ] Missing/noisy-data sensitivity

Current evidence already available:

- Sensor-level MAE for STGCN
- Horizon-wise MAE/RMSE/MAPE
- Proposed-model horizon metrics
- Controlled temporal-kernel ablation

Required next analyses:

1. Error by traffic regime
2. Error around sudden traffic changes
3. Sensor difficulty comparison
4. Forecast error distribution
5. Actual vs predicted examples
6. Comparison of STGCN vs proposed model under difficult cases

Questions:

- Where are errors larger?
- Which sensors are difficult?
- Does error increase with horizon?
- Does the proposed method change performance under difficult conditions?
- Are there systematic failure patterns?

Do not invent explanations unsupported by experiments.

Deliverable:

```text
docs/experiments/error_analysis.md
```

---

# 16. Phase 10 — ML Inference Engine

Status: **Not started**

File:

```text
src/inference/predictor.py
```

Interface:

```python
predictor.predict(input_data)
```

Responsibilities:

- [ ] Load trained model
- [ ] Load preprocessing configuration
- [ ] Load graph structure
- [ ] Accept traffic input
- [ ] Apply preprocessing
- [ ] Generate prediction
- [ ] Inverse-transform prediction
- [ ] Return structured result
- [ ] Validate input tensor shape
- [ ] Add inference-time tests
- [ ] Measure inference latency

This remains pure ML. No FastAPI required yet.

---

# 17. AI/Research Core Completion Milestone

Current path:

```text
METR-LA
   ↓
Exploration                    [COMPLETE]
   ↓
Preprocessing                  [COMPLETE]
   ↓
DataLoader                     [COMPLETE]
   ↓
HA / LV / LR / LSTM            [COMPLETE]
   ↓
STGCN                          [COMPLETE]
   ↓
Unified Evaluation             [MOSTLY COMPLETE]
   ↓
Literature Review              [PARTIALLY COMPLETE]
   ↓
Proposed MultiScale STGCN      [COMPLETE]
   ↓
Ablation Study                 [COMPLETE]
   ↓
Error Analysis                 [NEXT]
   ↓
ML Inference                   [NEXT]
```

The AI/research core is considered complete only after:

```text
Literature Review
        +
Error Analysis
        +
Inference Engine
        +
Reproducibility Verification
```

are completed.

At that point the AI component is ready for academic demonstration, project evaluation, and resume presentation.

---

# 18. Later Application Layer

Only after the AI/research core is stable.

## Phase 11 — FastAPI

Endpoints:

```text
GET  /health
GET  /roads
GET  /traffic/current
GET  /traffic/history
GET  /predictions
POST /predict
GET  /analytics
```

Connect the inference engine to the API.

## Phase 12 — PostgreSQL

Tables:

```text
roads
traffic_measurements
predictions
```

Optional:

```text
weather
events
accidents
```

Only if required by the final research model or application.

## Phase 13 — React Dashboard

Sections:

- Overview
- Road Network
- Prediction
- Analytics

Features:

- Current traffic
- Predicted congestion
- Average speed
- Alerts
- Sensor/road selection
- Prediction horizon
- Historical traffic
- Prediction accuracy
- Model performance
- Forecast charts
- Road network visualization

The dashboard should demonstrate the ML system, not be a generic admin panel.

## Phase 14 — Docker

Later:

```text
frontend
backend
postgres
```

Only introduce a separate ML service if genuinely necessary.

## Phase 15 — Testing

ML:

- Dataset tests
- Preprocessing tests
- Tensor shape tests
- Forward-pass tests
- Evaluation tests
- Inference tests

Later backend:

- API tests
- Database tests
- Prediction endpoint tests
- Error handling

Later frontend:

- Component tests
- API integration tests
- Responsive testing

## Phase 16 — Deployment

- Build Docker images
- Configure production environment
- Deploy backend
- Deploy frontend
- Deploy database
- Configure environment variables
- Configure CORS
- Configure logging
- Test production API
- Test production dashboard

---

# 19. Research Paper

Status: **Research evidence substantially available; final paper still needs literature/error-analysis integration**

Final structure:

1. Abstract
2. Introduction
3. Related Work
4. Problem Formulation
5. Methodology
6. Proposed Model
7. Experimental Setup
8. Results
9. Ablation Study
10. Error Analysis
11. Limitations
12. Conclusion
13. Future Work

Current paper-ready evidence:

- Dataset description
- Preprocessing methodology
- Canonical prediction task
- Baseline results
- STGCN results
- Proposed MultiScale STGCN results
- Horizon-wise results
- Controlled temporal-kernel ablation

Still required before final claims:

- Final literature review/gap
- Error analysis
- Reproducibility verification
- Final interpretation of the proposed contribution

Do not claim universal superiority of the proposed model.
Do not claim novelty without literature support.
Do not fabricate or alter experimental results.

---

# 20. Experiment Tracking

Every experiment should record:

```text
Model
Dataset
Input length
Prediction horizon
Learning rate
Batch size
Epochs
Optimizer
Hidden dimensions
Temporal kernels
Number of layers
Parameter count
Training time
Inference time
MAE
RMSE
MAPE
Notes
Random seed
Checkpoint
```

Do not rely on memory for results.

The controlled ablation uses:

```text
seed = 42
```

The main proposed-model run was trained separately before the controlled ablation seed was established; keep those runs distinguishable.

---

# 21. Project Rules

1. ML comes before UI.
2. GitHub/local project is the source of truth.
3. Colab is a temporary GPU runtime.
4. Never hardcode local Mac paths into notebooks or source code.
5. Never change 12 → 3 into 12 → 12 just to fix an error.
6. Never hide preprocessing bugs with notebook-specific reshape/unsqueeze hacks.
7. Fix source code rather than patching symptoms.
8. Verify saved data before training.
9. Do not claim novelty before literature review.
10. Do not fabricate experimental results.
11. Track experiments.
12. Keep research code separate from later production code.
13. Every major result must be reproducible.
14. Do not add external features without a clear research purpose.
15. The proposed contribution must be experimentally measurable.
16. Negative or mixed results are valid; do not manipulate results to appear positive.
17. Keep main model runs and controlled ablation runs clearly separated.
18. Report original-scale evaluation metrics for final interpretation.
19. Preserve the canonical tensor contract throughout the pipeline.
20. Prefer root-cause fixes over notebook-specific workarounds.

---

# 22. Current Status

```text
Phase 0  — Environment + Reproducibility    [~] Mostly Complete
Phase 1  — Dataset Exploration              [x] Completed
Phase 2  — Preprocessing                    [x] Completed
Phase 3  — Baselines                        [x] Completed
Phase 4  — STGCN                            [x] Completed
Phase 5  — Unified Evaluation               [~] Mostly Complete
Phase 6  — Literature Review                [~] Partially Complete
Phase 7  — Proposed Model                   [x] Completed
Phase 8  — Ablation                         [x] Completed
Phase 9  — Error Analysis                   [ ] Next
Phase 10 — ML Inference                     [ ] Next

========== AI/RESEARCH CORE COMPLETE ==========

Phase 11 — FastAPI                          [ ] Later
Phase 12 — PostgreSQL                       [ ] Later
Phase 13 — React Dashboard                  [ ] Later
Phase 14 — Docker                           [ ] Later
Phase 15 — Testing                          [ ] Later
Phase 16 — Deployment                       [ ] Later
Phase 17 — Research Paper                   [~] Evidence available; final writing later
Phase 18 — Presentation                     [ ] Later
```

---

# 23. Immediate Next Steps

Do these in order:

### Research core

1. Finish literature-review consolidation.
2. Create the literature comparison table.
3. Finalize the documented research gap.
4. Run error analysis.
5. Generate final research figures.
6. Consolidate final experiment CSVs.
7. Implement `src/inference/predictor.py`.
8. Test inference against saved test samples.
9. Verify reproducibility from a fresh Colab runtime.
10. Commit and push the complete ML/research core.

### Then

```text
========== AI/RESEARCH CORE COMPLETE ==========
```

Only after that:

11. FastAPI
12. PostgreSQL
13. React dashboard
14. Docker
15. Deployment
16. Final paper
17. Presentation

Do NOT start:

- FastAPI
- PostgreSQL
- React
- Docker
- Deployment
- Live traffic APIs
- Weather integration
- Event integration
- Accident integration

until the AI/research core is stable.

---

# 24. Antigravity Agent Implementation Rules

When asked to implement a task, the agent must actually implement it.

Do NOT merely:
- Read files
- Create a verification-only script
- Explain what should be changed
- Produce an implementation plan
- Claim something is fixed without executing it

Required workflow:

```text
Inspect
  ↓
EDIT ACTUAL SOURCE FILES
  ↓
RUN IMPLEMENTATION
  ↓
TEST
  ↓
FIX FAILURES
  ↓
RUN AGAIN
  ↓
SHOW ACTUAL CHANGES + RESULTS
```

For preprocessing:

```text
Source code
    ↓
Regenerate sequences.npz
    ↓
Inspect saved shapes
    ↓
DataLoader test
    ↓
Model forward-pass test
```

Never consider a task complete merely because a verification script was created.

For the canonical preprocessing contract, completion requires actual generated data:

```text
X = (N, 12, 207, 1)
Y = (N, 3, 207, 1)
```

and actual DataLoader/model verification.

For model changes, completion requires:

```text
Source implementation
      ↓
Shape test
      ↓
Training run
      ↓
Validation/test evaluation
      ↓
Saved checkpoint/artifacts
      ↓
Recorded result
```

---

# 25. Final Target Architecture

```text
                         URBANFLOW AI
                              |
                    ┌─────────┴─────────┐
                    |                   |
                 RESEARCH          APPLICATION
                    |                   |
              Traffic Dataset       FastAPI
                    |                   |
              Preprocessing        PostgreSQL
                    |                   |
               Baselines           React
                    |                   |
                  STGCN              Docker
                    |
             Research Model
                    |
              Experiments
                    |
             Ablation Study
                    |
             Error Analysis
                    |
             ML Inference
```

Final objective:

A reproducible traffic forecasting research system with a validated STGNN model, a measurable research contribution, controlled experiments, documented error analysis, a reusable ML inference engine, and a later production-style software platform built around the model.

**AI/research first. Application second.**
