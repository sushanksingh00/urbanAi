
# UrbanFlow AI — Development Plan

## Intelligent Traffic Prediction using Spatio-Temporal Graph Neural Networks

---

# 1. Project Overview

UrbanFlow AI is an end-to-end intelligent traffic prediction platform using Spatio-Temporal Graph Neural Networks (STGNNs).

The project has two main goals:

### Software Goal

Build a complete traffic prediction platform with:

- ML prediction engine
- FastAPI backend
- PostgreSQL database
- React dashboard
- REST APIs
- Docker deployment

### Research Goal

- Implement established traffic forecasting models
- Compare baseline models
- Implement an STGNN
- Perform a literature review
- Identify a research gap
- Propose one improvement
- Evaluate the improvement experimentally
- Prepare a research paper

---

# 2. Development Philosophy

Do NOT start by building the dashboard.

The ML pipeline is the core of the project.

Development order:

```text
Research
    ↓
Dataset
    ↓
Data Processing
    ↓
Baseline Models
    ↓
STGNN
    ↓
Research Improvement
    ↓
Experiments
    ↓
ML Inference API
    ↓
Backend
    ↓
Database
    ↓
Dashboard
    ↓
Docker
    ↓
Deployment
    ↓
Research Paper
````

The first objective is simple:

```text
Past Traffic
     ↓
STGNN
     ↓
Future Traffic
```

Only after this works should the software platform be built.

---

# 3. Technology Stack

## Machine Learning

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Jupyter Notebook

## Backend

* FastAPI
* Pydantic
* Uvicorn

## Database

* PostgreSQL

## Frontend

* React
* TypeScript
* Tailwind CSS
* Recharts or Plotly
* Map visualization library

## Infrastructure

* Git
* GitHub
* Docker
* Docker Compose

---

# 4. Dataset Strategy

Start with an established traffic forecasting benchmark.

Preferred datasets:

1. METR-LA
2. PEMS-BAY
3. PEMS04
4. PEMS08

Initial recommendation:

```text
METR-LA or PEMS-BAY
```

Do not start with live traffic APIs.

Do not start with Indian traffic data.

The first objective is reproducible research using a standard benchmark dataset.

---

# 5. Initial Prediction Task

Start with:

```text
Input:
Previous 60 minutes of traffic

Output:
Next 15 minutes of traffic
```

If the dataset uses 5-minute intervals:

```text
Input  = 12 time steps
Output = 3 time steps
```

After the initial model works, extend to:

```text
12 → 3     # 15 minutes
12 → 6     # 30 minutes
12 → 12    # 60 minutes
```

Do not try to solve every prediction horizon immediately.

---

# 6. Project Structure

```text
urbanflow-ai/
│
├── README.md
├── PLAN.md
├── requirements.txt
├── .gitignore
├── .env.example
├── docker-compose.yml
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
│   └── 04_stgnn_experiments.ipynb
│
├── ml/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   ├── graph.py
│   │   └── dataloader.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── lstm.py
│   │   ├── stgcn.py
│   │   └── proposed_model.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── config.py
│   │   └── checkpoints.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── evaluate.py
│   │   └── visualization.py
│   │
│   └── inference/
│       ├── __init__.py
│       └── predictor.py
│
├── experiments/
│   ├── baseline/
│   ├── stgcn/
│   └── proposed/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── api/
│   │   │   ├── traffic.py
│   │   │   ├── predictions.py
│   │   │   └── analytics.py
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── database/
│   │
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── hooks/
│   │   └── types/
│   └── Dockerfile
│
├── docs/
│   ├── research/
│   ├── architecture/
│   ├── experiments/
│   └── api/
│
└── deployment/
    ├── docker/
    └── scripts/
```

---

# 7. Phase 0 — Environment Setup

## Objective

Create a clean and reproducible development environment.

## Tasks

* [ ] Install Python
* [ ] Create virtual environment
* [ ] Install PyTorch
* [ ] Install NumPy
* [ ] Install Pandas
* [ ] Install Scikit-learn
* [ ] Install Matplotlib
* [ ] Install Jupyter
* [ ] Install FastAPI
* [ ] Install Uvicorn
* [ ] Configure Git
* [ ] Create GitHub repository
* [ ] Create `.gitignore`
* [ ] Create `requirements.txt`
* [ ] Create project folder structure

## Definition of Done

The following commands work:

```bash
python --version
```

```bash
python -c "import torch; print(torch.__version__)"
```

---

# 8. Phase 1 — Dataset Exploration

## Objective

Understand the dataset before building models.

## Tasks

* [ ] Download dataset
* [ ] Store raw data in `data/raw/`
* [ ] Document dataset source
* [ ] Identify number of sensors
* [ ] Identify sampling frequency
* [ ] Identify target variable
* [ ] Inspect missing values
* [ ] Inspect timestamps
* [ ] Inspect sensor metadata
* [ ] Inspect graph/adjacency information
* [ ] Visualize traffic distributions
* [ ] Visualize traffic over time
* [ ] Identify abnormal values
* [ ] Document dataset statistics

## Questions to Answer

* How many nodes are there?
* What does each node represent?
* What is the sampling interval?
* What does the traffic value represent?
* How are nodes connected?
* How much data is available?
* Are there missing values?
* Are there anomalies?

## Deliverable

```text
notebooks/01_dataset_exploration.ipynb
```

---

# 9. Phase 2 — Data Preprocessing

## Objective

Convert raw traffic data into model-ready sequences.

## Tasks

* [ ] Clean data
* [ ] Handle missing values
* [ ] Normalize traffic values
* [ ] Construct graph representation
* [ ] Construct adjacency matrix
* [ ] Create sliding windows
* [ ] Create train/validation/test split
* [ ] Prevent temporal data leakage
* [ ] Create PyTorch Dataset
* [ ] Create DataLoader
* [ ] Verify tensor dimensions

## Target Input

```text
(batch, time, nodes, features)
```

Example:

```text
(32, 12, 207, 1)
```

## Target Output

```text
(batch, prediction_time, nodes, features)
```

Example:

```text
(32, 3, 207, 1)
```

## Definition of Done

This pipeline works:

```text
Raw Dataset
     ↓
Preprocessing
     ↓
PyTorch Dataset
     ↓
DataLoader
```

---

# 10. Phase 3 — Baseline Models

## Objective

Create simple models before using GNNs.

This phase is mandatory.

---

## Baseline 1 — Historical Average

Predict historical average traffic.

Tasks:

* [ ] Implement
* [ ] Evaluate
* [ ] Save results

---

## Baseline 2 — Last Value

Predict the most recent traffic value.

Tasks:

* [ ] Implement
* [ ] Evaluate
* [ ] Save results

---

## Baseline 3 — Linear Regression

Tasks:

* [ ] Implement
* [ ] Train
* [ ] Evaluate
* [ ] Save results

---

## Baseline 4 — LSTM

Tasks:

* [ ] Implement LSTM
* [ ] Train
* [ ] Tune
* [ ] Evaluate
* [ ] Save checkpoint

---

## Deliverable

Create:

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

# 11. Phase 4 — STGCN

## Objective

Implement an established Spatio-Temporal Graph Neural Network.

Initial model:

```text
STGCN
```

Architecture concept:

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
Future Traffic
```

## Tasks

* [ ] Read the STGCN paper
* [ ] Understand graph convolution
* [ ] Understand temporal convolution
* [ ] Implement STGCN
* [ ] Test tensor shapes
* [ ] Run forward pass
* [ ] Train model
* [ ] Monitor training loss
* [ ] Monitor validation loss
* [ ] Tune hyperparameters
* [ ] Save best checkpoint
* [ ] Evaluate on test data

## Deliverables

```text
ml/models/stgcn.py
```

and:

```text
experiments/stgcn/
```

---

# 12. Phase 5 — Evaluation Framework

## Objective

Create one consistent evaluation system for all models.

## Metrics

Implement:

* [ ] MAE
* [ ] RMSE
* [ ] MAPE

Optional:

* [ ] WAPE
* [ ] R²
* [ ] MAE by prediction horizon

## Model Comparison

Compare:

```text
Historical Average
        ↓
Last Value
        ↓
Linear Regression
        ↓
LSTM
        ↓
STGCN
```

## Visualizations

Create:

* [ ] Actual vs predicted traffic
* [ ] Training loss
* [ ] Validation loss
* [ ] Error distribution
* [ ] Error by prediction horizon
* [ ] Node-level performance

## Deliverable

Create:

```text
ml/evaluation/metrics.py
ml/evaluation/evaluate.py
ml/evaluation/visualization.py
```

---

# 13. Phase 6 — Literature Review

## Objective

Find a genuine research opportunity.

Do NOT choose the research contribution before completing this phase.

## Research Topics

Study:

* STGCN
* DCRNN
* Graph WaveNet
* ASTGCN
* MTGNN
* Dynamic graph methods
* Weather-aware forecasting
* Event-aware forecasting
* Accident-aware forecasting
* Explainable GNNs
* Lightweight GNNs

## Tasks

* [ ] Collect 15–25 relevant papers
* [ ] Read abstracts
* [ ] Read methodology
* [ ] Record datasets
* [ ] Record models
* [ ] Record metrics
* [ ] Record limitations
* [ ] Record future work
* [ ] Create literature comparison table
* [ ] Identify recurring limitations
* [ ] Select one research direction

## Deliverable

```text
docs/research/literature_review.md
```

---

# 14. Phase 7 — Select Research Contribution

Possible directions:

## Option A — Dynamic Graph

Allow the model to learn changing relationships between traffic nodes.

```text
Graph at t1
    ↓
Graph at t2
    ↓
Graph at t3
```

---

## Option B — Weather-Aware STGNN

Add weather information:

```text
Traffic
   +
Weather
   ↓
STGNN
```

---

## Option C — Event-Aware STGNN

Add events such as:

* Concerts
* Sports events
* Festivals
* Public gatherings

---

## Option D — Accident-Aware STGNN

Add accident information.

---

## Option E — Explainable STGNN

Provide explanations for predictions.

Example:

```text
Predicted Congestion: HIGH

Possible contributing factors:

1. Upstream congestion
2. Sudden traffic increase
3. Adjacent road congestion
4. External conditions
```

---

## Option F — Lightweight STGNN

Reduce:

* Model parameters
* Memory usage
* Inference time

while maintaining prediction accuracy.

---

# 15. Research Direction Selection

Choose the final research direction using:

| Criterion                 | Importance |
| ------------------------- | ---------- |
| Novelty                   | High       |
| Dataset availability      | High       |
| Reproducibility           | High       |
| Research value            | High       |
| Implementation difficulty | Medium     |
| Evaluation feasibility    | High       |
| Time required             | High       |

Do not choose an idea just because it sounds impressive.

Choose an idea that can be experimentally validated.

---

# 16. Phase 8 — Proposed Model

## Objective

Implement the selected research improvement.

Example:

```text
STGCN
  +
Weather Features
  ↓
Proposed STGNN
```

## Tasks

* [ ] Define research hypothesis
* [ ] Define architecture
* [ ] Define additional inputs
* [ ] Implement proposed model
* [ ] Train model
* [ ] Tune hyperparameters
* [ ] Evaluate
* [ ] Compare with STGCN
* [ ] Save checkpoints
* [ ] Record experiments

---

# 17. Phase 9 — Ablation Study

This is mandatory for the research paper.

Suppose the final model contains multiple components.

Test each component independently.

Example:

```text
Model A: STGCN

Model B: STGCN + Weather

Model C: STGCN + Dynamic Graph

Model D: STGCN + Weather + Dynamic Graph
```

## Tasks

* [ ] Define ablation experiments
* [ ] Run experiments
* [ ] Record metrics
* [ ] Compare results
* [ ] Analyze failures
* [ ] Identify useful components

## Deliverable

```text
docs/experiments/ablation_study.md
```

---

# 18. Phase 10 — Error Analysis

Do not report only average metrics.

Analyze where the model fails.

## Analyze

* [ ] Rush hours
* [ ] High congestion
* [ ] Low congestion
* [ ] Different sensors
* [ ] Different prediction horizons
* [ ] Sudden traffic changes
* [ ] Missing/noisy data
* [ ] Weather conditions if applicable
* [ ] Events if applicable
* [ ] Accidents if applicable

## Questions

* Where does the model perform poorly?
* Which nodes are difficult to predict?
* Does performance degrade at 30/60 minutes?
* Does the proposed improvement help during unusual conditions?
* Does the proposed model actually beat STGCN?

---

# 19. Phase 11 — ML Inference Service

Once the research model works, create a reusable inference module.

## Tasks

* [ ] Load trained model
* [ ] Load preprocessing configuration
* [ ] Load graph structure
* [ ] Accept traffic input
* [ ] Generate predictions
* [ ] Inverse-transform predictions
* [ ] Return structured results

Target interface:

```python
predictor.predict(input_data)
```

## Deliverable

```text
ml/inference/predictor.py
```

---

# 20. Phase 12 — FastAPI Backend

## Objective

Expose the ML system through REST APIs.

## Initial Endpoints

```text
GET  /health
GET  /roads
GET  /traffic/current
GET  /traffic/history
GET  /predictions
POST /predict
GET  /analytics
```

## Tasks

* [ ] Create FastAPI application
* [ ] Configure environment variables
* [ ] Create Pydantic schemas
* [ ] Create API routes
* [ ] Connect inference engine
* [ ] Add error handling
* [ ] Add API documentation
* [ ] Add API tests

---

# 21. Phase 13 — PostgreSQL

## Objective

Store traffic and prediction information.

## Table: roads

```text
id
name
source_node
destination_node
geometry
```

## Table: traffic_measurements

```text
id
sensor_id
timestamp
traffic_value
```

## Table: predictions

```text
id
sensor_id
timestamp
prediction
model_version
```

Optional tables:

```text
weather
events
accidents
```

Only create these if required by the selected research contribution.

---

# 22. Phase 14 — React Dashboard

## Objective

Build a traffic intelligence dashboard.

## Dashboard Sections

### Overview

Display:

* Current traffic
* Predicted congestion
* Average speed
* Active alerts

### Road Network

Display:

* Road graph
* Sensors
* Current traffic
* Predicted traffic
* Congestion levels

### Prediction

Allow user to select:

* Road
* Sensor
* Prediction horizon

Display:

```text
Current
   ↓
+15 min
   ↓
+30 min
   ↓
+60 min
```

### Analytics

Display:

* Historical traffic
* Prediction accuracy
* Congestion trends
* Prediction history
* Model performance

---

# 23. Phase 15 — Dashboard Design

The dashboard should demonstrate the ML system.

It should not look like a generic admin panel.

Concept:

```text
+------------------------------------------------+
| UrbanFlow AI                         LIVE       |
+------------------------------------------------+
|                                                |
|              ROAD NETWORK MAP                  |
|                                                |
|        ●--------●--------●                     |
|        |        |        |                     |
|        ●--------●--------●                     |
|                                                |
+------------------------------------------------+
| Congestion | Prediction | Average Speed | Alerts|
|    67%     |    HIGH    |    34 km/h    |   3   |
+------------------------------------------------+
|                                                |
| Traffic Forecast                              |
|                                                |
|       Actual                                   |
|       Predicted                                |
|                                                |
+------------------------------------------------+
```

---

# 24. Phase 16 — Docker

Create:

```text
Dockerfile
docker-compose.yml
```

Initial services:

```text
frontend
backend
postgres
```

Possible final architecture:

```text
frontend
backend
ml-service
postgres
```

Use a separate ML service only if the project actually requires it.

---

# 25. Phase 17 — Testing

## Machine Learning

* [ ] Dataset tests
* [ ] Preprocessing tests
* [ ] Tensor shape tests
* [ ] Model forward-pass tests
* [ ] Evaluation tests
* [ ] Inference tests

## Backend

* [ ] API tests
* [ ] Database tests
* [ ] Prediction endpoint tests
* [ ] Error handling tests

## Frontend

* [ ] Component tests
* [ ] API integration tests
* [ ] Responsive layout testing

---

# 26. Phase 18 — Deployment

## Tasks

* [ ] Build Docker images
* [ ] Configure production environment
* [ ] Deploy backend
* [ ] Deploy frontend
* [ ] Deploy PostgreSQL
* [ ] Configure environment variables
* [ ] Configure CORS
* [ ] Configure logging
* [ ] Test production API
* [ ] Test production dashboard

## Final Flow

```text
User
  ↓
React Dashboard
  ↓
FastAPI
  ↓
ML Inference
  ↓
STGNN
  ↓
Prediction
```

PostgreSQL stores historical and prediction data.

---

# 27. Phase 19 — Research Paper

Recommended structure:

```text
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
```

Important:

Do not write final research claims before running the experiments.

The results determine the claims.

---

# 28. Phase 20 — Final Presentation

Final presentation should cover:

1. Problem
2. Motivation
3. Existing limitations
4. Proposed system
5. Dataset
6. STGNN methodology
7. Research contribution
8. Experimental results
9. System architecture
10. Dashboard demonstration
11. Conclusion
12. Future work

---

# 29. Experiment Tracking

Every experiment must record:

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
Training configuration
MAE
RMSE
MAPE
Training time
Inference time
Notes
```

Example:

```text
Model: STGCN
Dataset: METR-LA
Input: 12 steps
Output: 3 steps
Learning Rate: 0.001
Batch Size: 32
Epochs: 100
MAE: ...
RMSE: ...
MAPE: ...
```

Keep experiment results organized.

Do not rely on memory.

---

# 30. Milestones

## Milestone 1 — Dataset Ready

Definition of done:

* [ ] Dataset downloaded
* [ ] Dataset understood
* [ ] Nodes identified
* [ ] Graph understood
* [ ] Traffic visualized
* [ ] Dataset documented

---

## Milestone 2 — Data Pipeline Ready

Definition of done:

* [ ] Preprocessing works
* [ ] Normalization works
* [ ] Graph matrix works
* [ ] Sliding windows work
* [ ] DataLoader works
* [ ] Train/validation/test split works

---

## Milestone 3 — Baselines Ready

Definition of done:

* [ ] Historical Average
* [ ] Last Value
* [ ] Linear Regression
* [ ] LSTM
* [ ] Metrics calculated
* [ ] Results saved

---

## Milestone 4 — STGCN Ready

Definition of done:

* [ ] STGCN implemented
* [ ] Model trains
* [ ] Validation works
* [ ] Test results generated
* [ ] Baseline comparison completed

---

## Milestone 5 — Research Direction Selected

Definition of done:

* [ ] Literature review completed
* [ ] Existing approaches compared
* [ ] Research gap identified
* [ ] Research hypothesis defined
* [ ] One research direction selected

---

## Milestone 6 — Proposed Model Ready

Definition of done:

* [ ] Proposed model implemented
* [ ] Training completed
* [ ] Baseline comparison completed
* [ ] Ablation study completed
* [ ] Error analysis completed

---

## Milestone 7 — ML Inference Ready

Definition of done:

* [ ] Model loading works
* [ ] Preprocessing works
* [ ] Prediction works
* [ ] Inverse transformation works
* [ ] Inference API interface defined

---

## Milestone 8 — Backend Ready

Definition of done:

* [ ] FastAPI running
* [ ] ML inference integrated
* [ ] PostgreSQL connected
* [ ] Prediction API working
* [ ] API tests passing

---

## Milestone 9 — Dashboard Ready

Definition of done:

* [ ] Road network visualization
* [ ] Current traffic
* [ ] Predictions
* [ ] Historical analytics
* [ ] Prediction history
* [ ] Model metrics

---

## Milestone 10 — Production Ready

Definition of done:

* [ ] Docker setup complete
* [ ] Application deployed
* [ ] README complete
* [ ] Demo working
* [ ] Research results finalized
* [ ] Presentation finalized

---

# 31. Suggested 12-Week Timeline

## Week 1

Dataset + Environment

* [ ] Setup repository
* [ ] Setup Python environment
* [ ] Select dataset
* [ ] Download dataset
* [ ] Explore dataset

## Week 2

Data Pipeline

* [ ] Preprocessing
* [ ] Normalization
* [ ] Graph construction
* [ ] Sliding windows
* [ ] DataLoader

## Week 3

Baselines

* [ ] Historical Average
* [ ] Last Value
* [ ] Linear Regression
* [ ] LSTM
* [ ] Evaluation

## Week 4

STGCN

* [ ] Study architecture
* [ ] Implement
* [ ] Train
* [ ] Evaluate
* [ ] Tune

## Week 5

Research Review

* [ ] Read papers
* [ ] Compare models
* [ ] Identify limitations
* [ ] Identify research gap

## Week 6

Research Design

* [ ] Select research direction
* [ ] Define hypothesis
* [ ] Design proposed model

## Week 7

Proposed Model

* [ ] Implement
* [ ] Train
* [ ] Tune

## Week 8

Experiments

* [ ] Baseline comparison
* [ ] Ablation study
* [ ] Error analysis

## Week 9

ML Finalization

* [ ] Final model
* [ ] Final metrics
* [ ] Inference pipeline
* [ ] Documentation

## Week 10

Backend

* [ ] FastAPI
* [ ] PostgreSQL
* [ ] ML integration
* [ ] API tests

## Week 11

Frontend

* [ ] React
* [ ] Traffic map
* [ ] Prediction charts
* [ ] Analytics

## Week 12

Deployment + Finalization

* [ ] Docker
* [ ] Deployment
* [ ] Documentation
* [ ] Research paper
* [ ] Presentation
* [ ] Demo

---

# 32. Project Rules

## Rule 1

Do not build the UI before the ML pipeline works.

## Rule 2

Do not claim novelty before the literature review.

## Rule 3

Do not compare only against weak baselines.

## Rule 4

Do not use only one evaluation metric.

## Rule 5

Do not randomly modify the model until the result improves.

Track every experiment.

## Rule 6

Every major result must be reproducible.

## Rule 7

Keep research code separate from production code.

## Rule 8

Do not add weather, events, or accidents unless they have a clear research purpose.

## Rule 9

The research contribution must be measurable.

## Rule 10

If the proposed method does not improve performance, investigate why.

Do not pretend that a negative result is a positive result.

---

# 33. Current Project Status

Update this section as development progresses.

```text
Phase 0  — Environment        [ ] Not Started
Phase 1  — Dataset            [ ] Not Started
Phase 2  — Preprocessing      [ ] Not Started
Phase 3  — Baselines          [ ] Not Started
Phase 4  — STGCN              [ ] Not Started
Phase 5  — Evaluation         [ ] Not Started
Phase 6  — Literature Review  [ ] Not Started
Phase 7  — Research Model     [ ] Not Started
Phase 8  — Experiments        [ ] Not Started
Phase 9  — ML Inference       [ ] Not Started
Phase 10 — FastAPI            [ ] Not Started
Phase 11 — PostgreSQL         [ ] Not Started
Phase 12 — Dashboard          [ ] Not Started
Phase 13 — Docker             [ ] Not Started
Phase 14 — Deployment         [ ] Not Started
Phase 15 — Research Paper     [ ] Not Started
Phase 16 — Presentation       [ ] Not Started
```

---

# 34. Immediate Next Steps

Do ONLY these tasks first.

* [ ] Create GitHub repository
* [ ] Create Python virtual environment
* [ ] Create project structure
* [ ] Select initial dataset
* [ ] Download dataset
* [ ] Create dataset exploration notebook
* [ ] Understand nodes
* [ ] Understand graph structure
* [ ] Visualize traffic data
* [ ] Document dataset characteristics

Do NOT start:

* [ ] FastAPI
* [ ] React
* [ ] PostgreSQL
* [ ] Docker
* [ ] Weather integration
* [ ] Event integration
* [ ] Accident integration
* [ ] Research improvement

until the initial ML pipeline is working.

---

# 35. Final Target Architecture

```text
                         URBANFLOW AI
                              |
             +----------------+----------------+
             |                                 |
          RESEARCH                         SOFTWARE
             |                                 |
       STGNN Model                       FastAPI Backend
             |                                 |
    Research Improvement                  PostgreSQL
             |                                 |
       Experiments                       React Dashboard
             |                                 |
       Ablation Study                       Docker
             |                                 |
      Research Paper                       Deployment
```

Final technical flow:

```text
                    +----------------------+
                    |   Traffic Dataset    |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Data Preprocessing   |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Graph Construction   |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Baseline Models      |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | STGCN                |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Research Improvement |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Evaluation           |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | ML Inference Engine  |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | FastAPI Backend      |
                    +----------+-----------+
                               |
                +--------------+--------------+
                |                             |
                v                             v
       +----------------+           +----------------+
       | PostgreSQL     |           | React          |
       | Database       |           | Dashboard      |
       +----------------+           +----------------+
                                             |
                                             v
                                    +----------------+
                                    | Docker / Cloud |
                                    +----------------+
```

# Final Objective

UrbanFlow AI should not simply be:

"An AI traffic dashboard."

It should be:

"A reproducible traffic forecasting research system with a validated STGNN model, a measurable research contribution, and a production-style software platform built around the model."

The ML research comes first.

The dashboard comes second.

```
```
