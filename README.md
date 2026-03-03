#  Grain Challenge — M1 AI 2025-26

> **Multi-class classification of 8 grain varieties from spectral RGB images.**  
> Hosted on [Codabench](https://www.codabench.org/) · Organized as part of the M1 AI curriculum.

[![Open Starter Kit in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moujar/Grain-Challenge-M1-AI/blob/main/starter_kit/README.ipynb)

---

## 📌 Challenge at a Glance

| Aspect | Detail |
|---|---|
| **Task** | Classify grain variety from a single seed image |
| **Classes** | 8 varieties (labels 1–8) |
| **Images** | 252 × 252 × 3 (`int16`), spectral bands [22, 53, 89] |
| **Metric** | Classification **Accuracy** |
| **Constraint** | ≤ 20 min on Codabench GPU |
| **Data source** | INRAE (French National Research Institute for Agriculture, Food & Environment) |

---

## 📁 Repository Structure

```
Grain-Challenge-M1-AI/
│
├── README.md                       # ← You are here
├── requirements.txt                # Python dependencies
│
├── starter_kit/                    # Everything participants need
│   ├── README.ipynb                # 📓 Full walkthrough notebook (EDA → Train → Submit)
│   ├── README.md                   # Starter kit quick-reference
│   ├── input_data/                 # Place dataset .npz files here
│   └── submission/
│       └── model.py                # Baseline model (RandomForest + feature extraction)
│
└── Codabench Bundle/               # Competition platform configuration
    ├── competition.yaml            # Challenge definition & phases
    ├── logo.png
    ├── pages/                      # Challenge pages (overview, data, evaluation, …)
    ├── ingestion_program/          # Server-side ingestion pipeline
    ├── scoring_program/            # Server-side scoring (accuracy)
    ├── sample_code_submission/     # Reference model.py
    ├── input_data/
    ├── reference_data/
    └── utilities/
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/moujar/Grain-Challenge-M1-AI.git
cd Grain-Challenge-M1-AI
pip install -r requirements.txt
```

### 2. Get the data

Download the dataset `.npz` files from the **Codabench** competition page (Files section) and place them in `starter_kit/input_data/`.

### 3. Run the notebook

```bash
jupyter notebook starter_kit/README.ipynb
```

Or click the **Open in Colab** badge above to run in Google Colab — no local setup needed.

---

## 📓 What's in the Starter Notebook?

The [`starter_kit/README.ipynb`](starter_kit/README.ipynb) notebook is a complete end-to-end pipeline:

| Section | What it covers |
|---|---|
| **0 — Imports & Settings** | Environment setup, data paths |
| **1 — Data Loading** | `Data` class — loads `.npz` files, filters by year (2019/2020) |
| **2 — Visualization** | Class distribution, sample gallery, channel histograms, sparsity analysis, band correlations |
| **3 — Training** | ResNet-18 baseline with data augmentation, mixed-precision training, cosine LR schedule |
| **4 — Scoring** | Test-Time Augmentation (TTA), accuracy evaluation, submission packaging |

---

## 📤 Submission Format

Package your solution as a **ZIP file**:

```
submission.zip
└── model.py
```

`model.py` must contain a **`Model` class** with three methods:

```python
class Model:
    def __init__(self):      ...  # Initialize your model
    def fit(self, data):     ...  # Train on {'X': images, 'y': labels}
    def predict(self, data): ...  # Return predicted labels from {'X': images}
```

See [`starter_kit/submission/model.py`](starter_kit/submission/model.py) for a working baseline.

---

## 🏗️ Baselines Provided

| Model | Description |
|---|---|
| **RandomForest** | Feature extraction (histograms + statistics + PCA) → sklearn classifier *(in `submission/model.py`)* |
| **ResNet-18** | End-to-end CNN trained from scratch with augmentation & TTA *(in `README.ipynb`)* |

Participants are encouraged to beat these baselines using deeper architectures (EfficientNet, ViT, …), better augmentation, ensembling, or any other technique.

---

## 👥 Organizers

| Member |
|---|
| Oudoum Ali Houmed |
| Abderrahmane Moujar |
| Daryl Okou |
| Olutola Paul |
| Ran Lu |
| Cristian-Ioan Bratu |

---

## 📜 License

This project is part of the M1 AI Challenge 2025-26.  
Data provided by **INRAE**. See the competition terms on Codabench for usage restrictions.
