# UrbanFlow AI — Revised Development Plan

## Goal

UrbanFlow AI is an intelligent traffic forecasting research system using Spatio-Temporal Graph Neural Networks (STGNNs).

**Immediate priority:** build and verify the AI/ML system for academic evaluation, project marks, research, and resume presentation.

**Later:** build the full application around the validated ML system.

The application layer is intentionally postponed:
- FastAPI
- PostgreSQL
- React
- Docker
- Deployment

---

## Development Order

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

Do not implement the application layer during the initial ML phase.

---

# 3. Dataset

Primary dataset:

```text
METR-LA
```

Initial goal is reproducible research using a standard traffic forecasting benchmark.

Expected graph size:

```text
207 traffic sensors/nodes
```

Raw traffic may be represented as:

```text
(time, nodes)
```

Model representation must preserve a feature dimension:

```text
(time, nodes, features)
```

For this project:

```text
features = 1
```

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

Every baseline and neural model must follow this contract.

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
│   └── proposed/
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
- [x] Configure Git
- [ ] Push to GitHub
- [x] Create requirements.txt
- [x] Configure .gitignore
- [x] Configure centralized project paths
- [x] Test local Python
- [ ] Test Colab
- [ ] Test PyTorch
- [ ] Test CUDA/GPU
- [ ] Verify GitHub → Colab workflow

Definition of done:

A fresh Colab runtime can obtain the project from GitHub and execute it without manually reconstructing source files.

---

# 7. Phase 1 — Dataset Exploration

Notebook:

```text
notebooks/01_dataset_exploration.ipynb
```

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

Files:

```text
src/data/preprocessing.py
src/data/sequence.py
src/data/dataset.py
src/data/graph.py
```

Notebook:

```text
notebooks/02_data_preprocessing.ipynb
```

Tasks:

- [ ] Load raw traffic
- [ ] Clean data
- [ ] Handle missing values
- [ ] Normalize values
- [ ] Process adjacency matrix
- [ ] Chronological train/validation/test split
- [ ] Prevent temporal leakage
- [ ] Create sliding windows
- [ ] Create PyTorch Dataset
- [ ] Create DataLoader
- [ ] Verify tensor dimensions
- [ ] Save processed sequences

## Definition of done

Saved files:

```text
train:
X = (N, 12, 207, 1)
Y = (N, 3, 207, 1)

validation:
X = (N, 12, 207, 1)
Y = (N, 3, 207, 1)

test:
X = (N, 12, 207, 1)
Y = (N, 3, 207, 1)
```

DataLoader:

```text
batch_x = (B, 12, 207, 1)
batch_y = (B, 3, 207, 1)
```

Do not proceed if these shapes are wrong.

---

# 9. Phase 3 — Baseline Models

Mandatory:

1. Historical Average
2. Last Value
3. Linear Regression
4. LSTM

Input:

```text
(B, 12, 207, 1)
```

Output:

```text
(B, 3, 207, 1)
```

Tasks:

- [ ] Implement Historical Average
- [ ] Implement Last Value
- [ ] Implement Linear Regression
- [ ] Implement LSTM
- [ ] Train applicable models
- [ ] Evaluate
- [ ] Save results
- [ ] Save LSTM checkpoint

Notebook:

```text
notebooks/03_baselines.ipynb
```

Deliverable:

```text
experiments/baseline/results.csv
```

Example:

```text
model,MAE,RMSE,MAPE
HistoricalAverage,...
LastValue,...
LinearRegression,...
LSTM,...
```

---

# 10. Phase 4 — STGCN

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
Graph Convolution
  ↓
Prediction Layer
  ↓
3-step future prediction
```

Tasks:

- [ ] Study STGCN
- [ ] Understand graph convolution
- [ ] Understand temporal convolution
- [ ] Implement graph processing
- [ ] Implement STGCN
- [ ] Verify forward-pass shapes
- [ ] Train on Colab GPU
- [ ] Track training loss
- [ ] Track validation loss
- [ ] Tune hyperparameters
- [ ] Save best checkpoint
- [ ] Evaluate on test set

Files:

```text
src/models/stgcn.py
src/data/graph.py
src/training/train.py
```

Notebook:

```text
notebooks/04_stgcn.ipynb
```

Deliverables:

```text
experiments/stgcn/
├── checkpoints/
├── metrics.csv
└── plots/
```

---

# 11. Phase 5 — Unified Evaluation

Models:

```text
Historical Average
Last Value
Linear Regression
LSTM
STGCN
```

Metrics:

- [ ] MAE
- [ ] RMSE
- [ ] MAPE
- [ ] WAPE
- [ ] R²
- [ ] MAE by prediction horizon

Visualizations:

- [ ] Actual vs predicted
- [ ] Training loss
- [ ] Validation loss
- [ ] Error distribution
- [ ] Error by horizon
- [ ] Node-level performance

Files:

```text
src/evaluation/metrics.py
src/evaluation/evaluate.py
src/evaluation/visualization.py
```

Notebook:

```text
notebooks/05_evaluation.ipynb
```

Never hardcode or fabricate experimental results.

---

# 12. Phase 6 — Literature Review

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
- [ ] Select one research direction

Deliverable:

```text
docs/research/literature_review.md
```

Do not claim novelty before completing this phase.

---

# 13. Phase 7 — Proposed Research Model

Possible directions:

- Dynamic/adaptive graph
- Weather-aware STGNN
- Event-aware STGNN
- Accident-aware STGNN
- Explainable STGNN
- Lightweight STGNN

Do not automatically select one before literature review.

Tasks after selection:

- [ ] Define hypothesis
- [ ] Define architecture
- [ ] Define additional inputs/components
- [ ] Implement proposed model
- [ ] Train
- [ ] Tune
- [ ] Evaluate
- [ ] Compare against STGCN
- [ ] Save checkpoint
- [ ] Record experiments

Notebook:

```text
notebooks/06_proposed_model.ipynb
```

---

# 14. Phase 8 — Ablation Study

Example:

```text
STGCN
STGCN + Component A
STGCN + Component B
STGCN + A + B
```

Tasks:

- [ ] Define experiments
- [ ] Run experiments
- [ ] Record metrics
- [ ] Compare
- [ ] Analyze failures

Deliverable:

```text
docs/experiments/ablation_study.md
```

---

# 15. Phase 9 — Error Analysis

Analyze:

- [ ] Rush hours
- [ ] High congestion
- [ ] Low congestion
- [ ] Different sensors
- [ ] Different prediction horizons
- [ ] Sudden traffic changes
- [ ] Missing/noisy data

Questions:

- Where are errors larger?
- Which sensors are difficult?
- Does error increase with horizon?
- Does the proposed method change performance under difficult conditions?

Do not invent explanations unsupported by experiments.

---

# 16. Phase 10 — ML Inference Engine

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

This remains pure ML. No FastAPI required yet.

---

# 17. AI/Research Core Completion Milestone

The first major target is:

```text
METR-LA
   ↓
Exploration
   ↓
Preprocessing
   ↓
DataLoader
   ↓
HA / LV / LR / LSTM
   ↓
STGCN
   ↓
Evaluation
   ↓
Literature Review
   ↓
Proposed Model
   ↓
Ablation Study
   ↓
Error Analysis
   ↓
Inference Engine
```

At this point the AI/research component is ready for academic demonstration, project evaluation, and resume presentation.

---

# 18. Later Application Layer

Only after the AI core is stable.

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

Only if needed by the final research model.

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

Do not write final research claims before experiments are complete.

---

# 20. Experiment Tracking

Every experiment records:

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
Number of layers
Parameter count
Training time
Inference time
MAE
RMSE
MAPE
Notes
```

Do not rely on memory for results.

---

# 21. Project Rules

1. ML comes before UI.
2. GitHub/local project is the source of truth.
3. Colab is a temporary GPU runtime.
4. Never hardcode local Mac paths into notebooks.
5. Never change 12→3 into 12→12 just to fix an error.
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
16. Negative results are valid; do not manipulate results to appear positive.

---

# 22. Reset Status

```text
Phase 0  — Environment + Git + Colab       [x] Partially Complete (Pending GitHub/Colab tests)
Phase 1  — Dataset Exploration             [x] Completed
Phase 2  — Preprocessing                   [ ] Not Started
Phase 3  — Baselines                       [ ] Not Started
Phase 4  — STGCN                           [ ] Not Started
Phase 5  — Evaluation                      [ ] Not Started
Phase 6  — Literature Review               [ ] Not Started
Phase 7  — Proposed Model                  [ ] Not Started
Phase 8  — Ablation                        [ ] Not Started
Phase 9  — Error Analysis                  [ ] Not Started
Phase 10 — ML Inference                    [ ] Not Started

========== AI/RESEARCH CORE COMPLETE ==========

Phase 11 — FastAPI                         [ ] Later
Phase 12 — PostgreSQL                      [ ] Later
Phase 13 — React Dashboard                 [ ] Later
Phase 14 — Docker                          [ ] Later
Phase 15 — Testing                         [ ] Later
Phase 16 — Deployment                      [ ] Later
Phase 17 — Research Paper                  [ ] Later
Phase 18 — Presentation                    [ ] Later
```

---

# 23. Immediate Next Steps

ONLY work on these first:

1. Clean/reinitialize repository
2. Configure GitHub
3. Configure requirements.txt
4. Configure .gitignore
5. Configure project_paths.py
6. Set up Colab Git workflow
7. Verify Colab GPU
8. Obtain METR-LA
9. Run dataset exploration
10. Build preprocessing
11. Verify exact 4D tensor contract
12. Build DataLoader
13. Run baseline models
14. Implement/train STGCN
15. Evaluate models

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

A reproducible traffic forecasting research system with a validated STGNN model, a measurable research contribution, and a later production-style software platform built around the model.

**AI/research first. Application second.**
