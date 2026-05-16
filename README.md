# 🔍 Data Leakage Analysis in Earthquake Magnitude Prediction

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A critical reproduction study exposing data leakage in a published earthquake prediction paper (Yavas et al., *Scientific Reports*, 2024) that reported **97.97% accuracy** using machine learning. By correcting a single methodological flaw — random data splitting on temporal data — accuracy collapses to **21.3%**, falling *below* majority-class guessing (27.7%).

---

## 💥 The Finding

The original study used `random_state=15` to split earthquake data, allowing future seismic events to leak into the training set. This violates the fundamental requirement of temporal data: **the model must never see the future**.

| Model | Reported Accuracy | Corrected (Temporal Split) |
|-------|------------------|---------------------------|
| Random Forest | 97.97% | **21.3%** |
| XGBoost | 96.8% | 20.8% |
| LightGBM | 93.4% | 21.4% |
| MLP | 92.6% | 20.0% |
| Decision Tree | 89.7% | 20.8% |

All 16 models were fully reproduced with identical hyperparameters. Every single one degrades below chance-level when evaluated correctly.

---

## 🔬 Methodology

**The flaw:** The original paper applied random train/test splitting (`random_state=15`) to inherently temporal earthquake data. This allows the model to train on events from 2020 while predicting events from 2018 — a form of *overlap leakage*.

**The fix:** Strict chronological splitting with a hard cutoff date. Training and test sets are preprocessed independently. All rolling features (b-values, magnitude statistics, inter-event times) are recomputed without future knowledge.

**Dataset:** 23,284 earthquake events from the Southern California Earthquake Data Center (SCEDC), Los Angeles region, January 2012 – September 2024. Target variable: maximum magnitude within the next 30 days (6-class Jenks Natural Breaks classification).

---

## 🚀 Getting Started

```bash
git clone https://github.com/utkuefeakdogan/earthquake_pred.git
cd earthquake_pred
pip install pandas scikit-learn xgboost lightgbm jupyter

jupyter notebook table-script.ipynb
```

The dataset (`LosAngeles_Earthquake_Dataset.csv`) is included in the repository.

---

## 📁 Repository Structure

```
earthquake_pred/
├── table-script.ipynb                 # Full reproduction & corrected analysis
├── LosAngeles_Earthquake_Dataset.csv  # SCEDC earthquake data (LA region)
├── manuscript.pdf                     # Response paper
└── LICENSE
```

---

## 📄 Citation
Akdoğan, U.E., Gerek, A., Diner, Ç., Ata, E., Kalem, B., Özel, N.M. (2024).
Analysis of Data Leakage in Earthquake Magnitude Prediction: A Response Paper.
Boğaziçi University.

---

## 📝 License

Licensed under the MIT License. See [LICENSE](./LICENSE) for details.
