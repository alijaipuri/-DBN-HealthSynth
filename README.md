<div align="center">

# 🧬 DBN HealthSynth

<img src="https://img.shields.io/badge/version-2.0.0-00d4ff?style=for-the-badge&logo=git&logoColor=white"/>
<img src="https://img.shields.io/badge/license-MIT-7c3aed?style=for-the-badge"/>
<img src="https://img.shields.io/badge/python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
<img src="https://img.shields.io/badge/pytorch-2.3.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>

<br/>

> **Synthetic Patient Record Generation using Deep Belief Networks (DBN)**
> with Privacy-Preserving Evaluation, Downstream ML Task Assessment,
> and LLaMA-3.3-70B powered Clinical AI Insights.

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&pause=1000&color=00D4FF&center=true&vCenter=true&width=700&lines=Deep+Belief+Network+%E2%80%94+Synthetic+Healthcare+Data;Privacy-Preserving+AI+%E2%80%94+HIPAA+Aware;LLaMA-3.3-70B+Clinical+AI+Insights;Train+on+Synthetic%2C+Test+on+Real+(TSTR);Dockerised+%E2%80%94+One+Command+Deploy" alt="Typing SVG" />

<br/><br/>

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Step-by-Step Setup](#-step-by-step-setup)
- [Usage Guide](#-usage-guide)
- [DBN Architecture Deep Dive](#-dbn-architecture-deep-dive)
- [Privacy Evaluation Metrics](#-privacy-evaluation-metrics)
- [ML Evaluation — TSTR vs TRTR](#-ml-evaluation--tstr-vs-trtr)
- [Datasets Used](#-datasets-used)
- [API Reference](#-api-reference)
- [Docker Commands](#-docker-commands)
- [Academic Context](#-academic-context)
- [License](#-license)

---

## 🔭 Overview

**DBN HealthSynth** is a full-stack, research-grade AI application that generates **realistic synthetic patient records** using a **Deep Belief Network (DBN)** — a generative probabilistic model composed of stacked Restricted Boltzmann Machines (RBMs). The synthetic data is then rigorously evaluated across two axes:

| Axis | What it measures |
|------|-----------------|
| 🔒 **Privacy** | How well the synthetic data protects real patient identities |
| 🤖 **ML Utility** | How well models trained on synthetic data perform on real data (TSTR) |

The project integrates **Groq's LLaMA-3.3-70B** to provide expert-level clinical and privacy analysis, and is fully containerised with **Docker** for one-command deployment.

> **Course:** Advanced Generative AI (AGA) — Practical CIE Project
> **Problem Statement 1:** *Construct a DBN for generating synthetic patient records in healthcare; evaluate privacy-preserving sample quality using downstream ML task performance.*

---

## ✨ Features

### 🧠 Core AI/ML
- **Deep Belief Network (DBN)** with greedy layer-wise pre-training via Contrastive Divergence
- **Fine-tuning** with backpropagation, BatchNorm, Dropout, and Cosine LR scheduling
- **Configurable architecture** — adjust layers, learning rate, epochs, batch size from the UI
- **Temperature-controlled generation** — tune diversity of synthetic samples
- **Noise injection** for realistic variability

### 🔒 Privacy Evaluation
- **Membership Inference Risk** — Distance to Closest Record (DCR) analysis
- **Attribute Disclosure Risk** — K-nearest neighbour attribute leakage estimation
- **Statistical Fidelity** — Kolmogorov–Smirnov test + Wasserstein Distance per feature
- **Correlation Preservation** — Real vs synthetic feature correlation matrix comparison
- **Privacy Grade** (A / B / C / D) with overall composite score

### 🤖 ML Utility Evaluation
- **TSTR** (Train on Synthetic, Test on Real) across 4 classifiers
- **TRTR** (Train on Real, Test on Real) as baseline
- Classifiers: Random Forest, Gradient Boosting, Logistic Regression, MLP Neural Net
- Metrics: Accuracy, F1-Score, AUC-ROC, Precision, Recall
- **Utility Gap** and **Utility Preservation %** scores

### ✨ LLM-Powered Features (LLaMA-3.3-70B via Groq)
- **Privacy Expert Analysis** — clinical-grade assessment of privacy metrics
- **Architecture Explainer** — DBN explained in plain language for medical professionals
- **Patient Narrative Generator** — realistic clinical chart notes for synthetic patients

### 📊 Interactive Visualisations (Plotly)
- Real-time **training loss curves** (per RBM layer + fine-tune)
- **PCA scatter plot** — real vs synthetic distribution overlap
- **Feature distribution histograms** — per-column comparison
- **Correlation heatmaps** — side-by-side real vs synthetic
- **Privacy radar chart** — multi-axis privacy profile
- **TSTR vs TRTR bar chart** — model-by-model utility comparison

### 🖥️ UI/UX
- Futuristic dark-mode interface with animated gradients
- **Real-time training progress** via WebSockets (Socket.IO)
- Step-by-step workflow with status pills
- One-click **CSV export** of synthetic patient records
- Synthetic data table with human-readable values (Male/Female, not 0/1)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DBN HealthSynth                          │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐  │
│  │   UCI Repos  │───▶│        Data Generator                │  │
│  │ Heart Disease│    │  - Combined dataset (18 features)    │  │
│  │   Diabetes   │    │  - StandardScaler preprocessing      │  │
│  └──────────────┘    │  - Clinical postprocessing           │  │
│                      └──────────────┬───────────────────────┘  │
│                                     │                           │
│                      ┌──────────────▼───────────────────────┐  │
│                      │         Deep Belief Network           │  │
│                      │                                       │  │
│                      │  Input(18) → RBM₁(64) → RBM₂(32)    │  │
│                      │           → RBM₃(16)                 │  │
│                      │                                       │  │
│                      │  Phase 1: Greedy Pre-training (CD-1)  │  │
│                      │  Phase 2: Fine-tune (Backprop + Adam) │  │
│                      │  Generation: Decoder with temperature │  │
│                      └──────────────┬───────────────────────┘  │
│                                     │                           │
│               ┌─────────────────────┼────────────────────┐     │
│               │                     │                    │     │
│  ┌────────────▼──────┐  ┌───────────▼────────┐  ┌───────▼───┐ │
│  │  Privacy Evaluator│  │   ML Evaluator     │  │  LLM AI   │ │
│  │  - DCR / MIR      │  │  - TSTR / TRTR     │  │  Groq     │ │
│  │  - KS Test        │  │  - RF, GBT, LR,MLP │  │  LLaMA    │ │
│  │  - Wasserstein    │  │  - Utility Gap      │  │  3.3-70B  │ │
│  │  - Correlation    │  │  - Feature Imp.     │  │           │ │
│  └───────────────────┘  └────────────────────┘  └───────────┘ │
│                                     │                           │
│                      ┌──────────────▼───────────────────────┐  │
│                      │     Flask + Socket.IO Web App         │  │
│                      │     Plotly Charts  |  Dark UI         │  │
│                      └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### DBN Layer Structure

```
Real Patient Data (18 features)
         │
         ▼
┌─────────────────┐
│   Input Layer   │  18 neurons  (age, sex, cholesterol, glucose, BMI…)
└────────┬────────┘
         │  RBM₁ — Contrastive Divergence k=1
         ▼
┌─────────────────┐
│  Hidden Layer 1 │  64 neurons
└────────┬────────┘
         │  RBM₂ — Contrastive Divergence k=1
         ▼
┌─────────────────┐
│  Hidden Layer 2 │  32 neurons
└────────┬────────┘
         │  RBM₃ — Contrastive Divergence k=1
         ▼
┌─────────────────┐
│  Latent Space   │  16 neurons  ← generation starts here
└────────┬────────┘
         │  Decoder (reversed) + temperature scaling
         ▼
Synthetic Patient Record (18 features)
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|-------|-----------|
| **Deep Learning** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) PyTorch 2.3.1 |
| **ML Evaluation** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) scikit-learn 1.5 |
| **LLM / AI** | ![Meta](https://img.shields.io/badge/LLaMA_3.3_70B-0467DF?style=flat-square&logo=meta&logoColor=white) via Groq API |
| **Backend** | ![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white) Flask 3.0 + Socket.IO |
| **Data** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) |
| **Visualisation** | ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white) Plotly 5.22 |
| **Dataset Source** | ![UCI](https://img.shields.io/badge/UCI_ML_Repo-003366?style=flat-square&logo=databricks&logoColor=white) ucimlrepo |
| **Containerisation** | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white) Docker + Compose |
| **Frontend** | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black) |
| **Runtime** | ![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat-square&logo=python&logoColor=white) |

</div>

---

## 📁 Project Structure

```
dbn-healthcare/
│
├── 📄 Dockerfile                  # Docker image definition
├── 📄 docker-compose.yml          # Docker Compose config
├── 📄 .env                        # API keys (not committed)
├── 📄 requirements.txt            # Python dependencies
├── 📄 app.py                      # Flask app + all API routes
│
├── 📂 src/
│   ├── 🧠 dbn_model.py            # RBM + DBN implementation (PyTorch)
│   ├── 🏥 data_generator.py       # UCI data loading + clinical postprocessing
│   ├── 🔒 privacy_evaluator.py    # DCR, KS, Wasserstein, correlation metrics
│   ├── 🤖 ml_evaluator.py         # TSTR / TRTR evaluation pipeline
│   ├── ✨ llm_insights.py         # Groq LLaMA-3.3-70B integration
│   └── 📊 visualizer.py           # Plotly chart generators
│
├── 📂 data/
│   └── 📂 datasets/               # Auto-downloaded UCI datasets
│
├── 📂 static/
│   ├── 📂 css/
│   │   └── style.css              # Futuristic dark-mode UI styles
│   └── 📂 js/
│       └── main.js                # Frontend logic + Socket.IO client
│
└── 📂 templates/
    └── index.html                 # Main single-page app template
```

---

## ⚡ Quick Start

> **Prerequisites:** Docker Desktop installed and running on your Mac.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/dbn-healthcare.git
cd dbn-healthcare

# 2. Add your Groq API key
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# 3. Build and run
docker build -t dbn-healthsynth:latest .
docker run -d --name dbn-healthsynth -p 8080:8080 --env-file .env dbn-healthsynth:latest

# 4. Open in browser
open http://localhost:8080
```

> 🔑 Get your free Groq API key at [console.groq.com](https://console.groq.com)

---

## 🔧 Step-by-Step Setup

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Docker Desktop | Latest | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) |
| Git | Any | Pre-installed on Mac |
| Groq API Key | Free | [console.groq.com](https://console.groq.com) |

### 1 — Clone & Enter Directory

```bash
git clone https://github.com/YOUR_USERNAME/dbn-healthcare.git
cd dbn-healthcare
```

### 2 — Configure Environment

```bash
nano .env
```

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3 — Build Docker Image

```bash
docker build -t dbn-healthsynth:latest .
```

> ⏳ First build takes ~8–12 minutes (downloads PyTorch, scikit-learn, etc.)

### 4 — Run Container

```bash
docker run -d \
  --name dbn-healthsynth \
  -p 8080:8080 \
  --env-file .env \
  dbn-healthsynth:latest
```

### 5 — Open Application

```bash
open http://localhost:8080
```

Or via **Docker Desktop** → Click the `dbn-healthsynth` container → Click the **8080** port link.

---

## 📖 Usage Guide

The application follows a **5-step sequential workflow**:

```
Step 1          Step 2          Step 3          Step 4          Step 5
Load Data  ──▶  Train DBN  ──▶  Generate  ──▶  Evaluate  ──▶  AI Insights
```

### Step 1 — Load Healthcare Data

Click **"Load Healthcare Data"** in the sidebar.

The app automatically fetches two UCI datasets and combines them into an 18-feature patient dataset:
- UCI Heart Disease Dataset (id=45)
- Pima Indians Diabetes Dataset (id=34)

Features include: `age`, `sex`, `chest_pain`, `resting_bp`, `cholesterol`, `fasting_bs`, `rest_ecg`, `max_hr`, `exercise_angina`, `st_depression`, `glucose`, `bmi`, `insulin`, `skin_thickness`, `pregnancies`, `dpf`, `heart_disease`, `diabetes`, `risk_score`

### Step 2 — Configure & Train the DBN

Adjust the sidebar controls:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Pre-train Epochs | 50 | RBM Contrastive Divergence iterations |
| Fine-tune Epochs | 30 | Backpropagation refinement iterations |
| Learning Rate | 0.01 | Initial learning rate for CD |
| Batch Size | 64 | Mini-batch size for training |

Click **"🚀 Train DBN"** — watch real-time layer-by-layer training progress with live loss curves.

### Step 3 — Generate Synthetic Records

| Parameter | Default | Description |
|-----------|---------|-------------|
| Samples | 500 | Number of synthetic patients to generate |
| Temperature | 1.0 | Higher = more diverse, lower = more conservative |
| Noise Level | 0.1 | Gaussian noise added to raw samples |

Click **"🧬 Generate Synthetic Records"**. The app will:
1. Sample from the DBN latent space
2. Decode through the learned distribution
3. Apply clinical postprocessing (correct types, valid ranges, Male/Female labels)
4. Display in the data table and auto-render PCA + correlation charts

### Step 4 — Evaluate

Click **"🔒 Privacy Evaluation"** → runs membership inference, attribute disclosure, KS test, and correlation metrics.

Click **"🤖 ML Task Evaluation"** → runs TSTR vs TRTR across 4 classifiers.

### Step 5 — AI Expert Analysis

Click **"✨ AI Expert Analysis"** to consult LLaMA-3.3-70B for:
- Clinical privacy assessment
- HIPAA/GDPR compliance analysis
- Architecture explanation in plain language

Use the **Patient Narrative** tab to generate a realistic clinical chart note for any synthetic patient by index.

### Export

Click **"⬇ Export Synthetic CSV"** to download all generated records as a `.csv` file.

---

## 🧠 DBN Architecture Deep Dive

### Restricted Boltzmann Machine (RBM)

Each RBM is a two-layer undirected probabilistic graphical model with:
- **Visible layer** $v \in \{0,1\}^D$
- **Hidden layer** $h \in \{0,1\}^H$
- **Weight matrix** $W \in \mathbb{R}^{H \times D}$

**Energy function:**

$$E(v, h) = -b^T v - c^T h - h^T W v$$

**Conditional distributions:**

$$P(h_j = 1 \mid v) = \sigma\left(\sum_i W_{ji} v_i + c_j\right)$$

$$P(v_i = 1 \mid h) = \sigma\left(\sum_j W_{ji} h_j + b_i\right)$$

### Contrastive Divergence (CD-k)

Training uses CD-1 (k=1 Gibbs sampling steps):

$$\Delta W = \eta \left( \langle v h^T \rangle_{\text{data}} - \langle v h^T \rangle_{\text{recon}} \right)$$

### Greedy Layer-wise Pre-training

Each RBM is trained independently, bottom-up:
1. Train RBM₁ on input data → get hidden representation $h^{(1)}$
2. Train RBM₂ on $h^{(1)}$ → get $h^{(2)}$
3. Train RBM₃ on $h^{(2)}$ → get latent code $z$

### Fine-tuning

After pre-training, the full network (encoder + decoder) is fine-tuned end-to-end with:
- **Optimiser:** Adam (lr=0.001, weight_decay=1e-5)
- **Scheduler:** Cosine Annealing
- **Loss:** MSE reconstruction loss
- **Regularisation:** BatchNorm + Dropout(0.2) + Gradient Clipping

---

## 🔒 Privacy Evaluation Metrics

### 1. Membership Inference Risk (Distance to Closest Record)

Measures how close synthetic records are to real records. A high DCR ratio means synthetic records are far from real ones — better privacy.

$$\text{DCR ratio} = \frac{\text{mean distance(synthetic → nearest real)}}{\text{mean distance(real → nearest real)}}$$

| Risk Level | DCR Ratio |
|-----------|-----------|
| 🟢 Low | > 0.70 |
| 🟡 Medium | 0.40 – 0.70 |
| 🔴 High | < 0.40 |

### 2. Attribute Disclosure Risk

K-NN based analysis measuring how precisely real patient attributes can be inferred from synthetic records.

### 3. Statistical Fidelity

Per-feature Kolmogorov–Smirnov test comparing real vs synthetic marginal distributions. Score = $(1 - \bar{D}_{KS}) \times 100$

### 4. Correlation Preservation

Frobenius-norm difference between real and synthetic correlation matrices. Higher score = structural patterns preserved.

### Privacy Grade

| Grade | Score | Interpretation |
|-------|-------|----------------|
| 🏆 A | ≥ 80 | Excellent — production ready |
| ✅ B | 65–79 | Good — minor refinement needed |
| ⚠️ C | 50–64 | Moderate — review before deployment |
| ❌ D | < 50 | Poor — significant privacy risk |

---

## 🤖 ML Evaluation — TSTR vs TRTR

**Train on Synthetic, Test on Real (TSTR)** is the gold-standard utility metric for synthetic data.

```
TSTR Pipeline:
  Train ──▶ [Synthetic Data]  ──▶ Model ──▶ Test on [Real Data] ──▶ Metrics

TRTR Baseline:
  Train ──▶ [Real Data (CV)]  ──▶ Model ──▶ Cross-Val Score     ──▶ Metrics
```

**Utility Gap** = TRTR Accuracy − TSTR Accuracy

A small utility gap means the synthetic data is a good substitute for real data for ML training.

### Classifiers Used

| Model | Why included |
|-------|-------------|
| Random Forest | Robust, handles mixed types well |
| Gradient Boosting | High performance on tabular data |
| Logistic Regression | Linear baseline |
| MLP Neural Network | Non-linear deep baseline |

---

## 🏥 Datasets Used

### UCI Heart Disease Dataset
- **Source:** UCI ML Repository (id=45)
- **Origin:** Cleveland Clinic Foundation
- **Samples:** 303 patients
- **Features used:** age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression

### Pima Indians Diabetes Dataset
- **Source:** UCI ML Repository (id=34)
- **Origin:** National Institute of Diabetes
- **Samples:** 768 patients
- **Features used:** pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function

### Combined Dataset
Both datasets are merged to create an **18-feature synthetic patient profile** with a computed `risk_score` and `heart_disease` as the primary target label. All datasets are **auto-downloaded** at runtime via `ucimlrepo` — no manual download needed.

---

## 🔌 API Reference

All endpoints return JSON. Base URL: `http://localhost:8080`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/load_data` | Load and preprocess UCI datasets |
| `POST` | `/api/train` | Start DBN training (async via WebSocket) |
| `POST` | `/api/generate` | Generate synthetic patient records |
| `POST` | `/api/evaluate_privacy` | Run all privacy metrics |
| `POST` | `/api/evaluate_ml` | Run TSTR vs TRTR evaluation |
| `POST` | `/api/llm_analysis` | Get LLaMA-3.3-70B expert analysis |
| `POST` | `/api/patient_narrative` | Generate clinical note for one patient |
| `GET` | `/api/export` | Download synthetic records as JSON |
| `GET` | `/api/status` | Check current pipeline state |
| `GET` | `/api/charts/training` | Training loss Plotly figure |
| `GET` | `/api/charts/pca` | PCA scatter Plotly figure |
| `GET` | `/api/charts/correlation` | Correlation heatmap Plotly figure |
| `GET` | `/api/charts/distribution?feature=age` | Feature distribution Plotly figure |
| `GET` | `/api/charts/privacy_radar` | Privacy radar Plotly figure |
| `GET` | `/api/charts/ml_comparison` | TSTR vs TRTR bar chart Plotly figure |

### WebSocket Events (Socket.IO)

| Event | Direction | Payload |
|-------|-----------|---------|
| `training_progress` | Server → Client | `{layer, epoch, total_epochs, loss, phase}` |
| `training_status` | Server → Client | `{message, phase}` |
| `training_done` | Server → Client | `{success: true}` |
| `training_error` | Server → Client | `{message}` |

### Example: Train Request

```bash
curl -X POST http://localhost:8080/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "pretrain_epochs": 50,
    "finetune_epochs": 30,
    "lr": 0.01,
    "batch_size": 64,
    "layers": [18, 64, 32, 16]
  }'
```

### Example: Generate Request

```bash
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "n_samples": 500,
    "temperature": 1.0,
    "noise": 0.1
  }'
```

---

## 🐳 Docker Commands

```bash
# Build image
docker build -t dbn-healthsynth:latest .

# Run container
docker run -d --name dbn-healthsynth -p 8080:8080 --env-file .env dbn-healthsynth:latest

# View live logs
docker logs -f dbn-healthsynth

# Stop container
docker stop dbn-healthsynth

# Remove container
docker rm dbn-healthsynth

# Restart container
docker restart dbn-healthsynth

# Rebuild after code changes (full cycle)
docker stop dbn-healthsynth && docker rm dbn-healthsynth && \
docker build -t dbn-healthsynth:latest . && \
docker run -d --name dbn-healthsynth -p 8080:8080 --env-file .env dbn-healthsynth:latest

# Open shell inside container (for debugging)
docker exec -it dbn-healthsynth bash

# Check container resource usage
docker stats dbn-healthsynth

# Using Docker Compose instead
docker compose up --build -d
docker compose down
```

---

## 📚 Academic Context

### Problem Statement
> *"Construct a DBN for generating synthetic patient records in healthcare; evaluate privacy-preserving sample quality using downstream ML task performance."*
> — AGA Lab CIE, Problem Statement 1

### Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Deep Belief Networks | Stacked RBMs with CD-1 pre-training |
| Generative Modelling | Latent space sampling + decoder |
| Privacy-Preserving AI | DCR, KS, Wasserstein, correlation metrics |
| Downstream Evaluation | TSTR / TRTR with 4 ML models |
| Healthcare AI Ethics | HIPAA-aware privacy scoring |
| Transfer Learning | Pre-train then fine-tune paradigm |

### References

1. Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). *A fast learning algorithm for deep belief nets.* Neural Computation, 18(7), 1527–1554.
2. Hinton, G. E. (2002). *Training products of experts by minimizing contrastive divergence.* Neural Computation, 14(8), 1771–1800.
3. Jordon, J., Yoon, J., & van der Schaar, M. (2018). *PATE-GAN: Generating synthetic data with differential privacy guarantees.* ICLR.
4. UCI ML Repository — Heart Disease Dataset. Janosi, A., et al. (1988).
5. UCI ML Repository — Pima Indians Diabetes Dataset. Smith, J.W., et al. (1988).

---

## 🗺️ Roadmap

- [x] DBN with greedy pre-training + fine-tuning
- [x] TSTR / TRTR evaluation pipeline
- [x] Privacy metrics (DCR, KS, Wasserstein)
- [x] LLaMA-3.3-70B integration
- [x] Real-time training via WebSockets
- [x] Docker containerisation
- [ ] Differential Privacy (DP-SGD) support
- [ ] t-SNE visualisation alongside PCA
- [ ] CTGAN comparison benchmark
- [ ] SHAP feature importance charts
- [ ] PDF report auto-generation
- [ ] Multi-user session support

---

## 📄 License

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
```

---

<div align="center">

**Built for AGA Lab CIE — Advanced Generative AI**

*Deep Belief Networks · Privacy-Preserving AI · LLaMA-3.3-70B · Docker*

<br/>

⭐ If this project helped you, consider starring the repo!

</div>
