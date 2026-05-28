# Informer for Time Series Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Implementation of the **Informer** architecture (AAAI 2021) for multi-step time series forecasting in two real-world domains:
- 🌡️ **Delhi Climate** – 24‑hour temperature prediction
- ✈️ **Aircraft Engine Prognostics** – Remaining Useful Life (RUL) estimation using NASA C‑MAPSS data

The model outperforms LSTMs and standard Transformers while being computationally efficient through **ProbSparse Self‑Attention**.

---

## 📊 Results Summary

### Delhi Temperature Forecast (24-step ahead)
| Metric | Value |
|--------|-------|
| R² Score | 0.8567 |
| RMSE | 2.38 °C |
| MAE | 2.10 °C |
| Best Val Loss | 0.0235 |

### NASA C‑MAPSS (FD001) – Sensor Prediction & Anomaly Detection
| Task | Metric | Value |
|------|--------|-------|
| Sensor forecasting | R² | 0.9988 |
| Sensor forecasting | RMSE | 0.0161 °C |
| Anomaly detection | AUC‑ROC | 0.916 |
| Early warnings (5–15 cycles before failure) | 44.6% of cases |

---

## 🏗️ Model Architecture

- **Input dimensions** : 9 features (meteorological) / 14 sensors (C‑MAPSS)
- **d_model** : 128  
- **n_heads** : 4  
- **e_layers** : 2  
- **d_layers** : 1  
- **Dropout** : 0.1  

Special component: **TemporalWeightedLoss** – linearly increases importance for later prediction steps (weight 1.0 → 1.6 over 24 steps) to reduce error accumulation.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/informer-time-series-forecasting.git
cd informer-time-series-forecasting
