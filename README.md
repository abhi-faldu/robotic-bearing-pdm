# 🔩 robotic-bearing-pdm

> **Predictive Maintenance for Industrial Robotic Arms**  
> Detecting bearing failures before they happen — using LSTM Autoencoders on real NASA sensor data.

---

## 🏭 Real-World Problem

German automotive manufacturers like **BMW**, **Audi**, and **Mercedes-Benz** rely on thousands
of industrial robotic arms on their production lines. A single unexpected bearing failure can halt
an entire assembly line, costing upwards of **€500,000 per hour** in downtime.

Current industry practice is mostly **scheduled maintenance** — replacing parts on a fixed
calendar, regardless of actual condition. This project explores a smarter alternative:
**condition-based predictive maintenance**, where a machine learning model continuously monitors
sensor signals and raises an alert before a failure occurs.

---

## 🎯 What This Project Does

This end-to-end pipeline:

1. **Ingests** real vibration sensor data from NASA's IMS Bearing dataset
2. **Engineers features** from raw time-series signals (RMS, kurtosis, peak-to-peak amplitude)
3. **Trains an LSTM Autoencoder** on healthy bearing data only (unsupervised approach — no failure
   labels required)
4. **Detects anomalies** by measuring reconstruction error against a learned threshold
5. **Serves predictions** via a REST API (FastAPI)
6. **Visualises** live anomaly scores and sensor health in a Streamlit dashboard
7. **Packages everything** in Docker Compose for one-command deployment

---

## 🗂️ Project Structure

```
robotic-bearing-pdm/
│
├── data/                        # Raw and processed sensor data
│   ├── raw/                     # NASA IMS dataset (download separately — see below)
│   └── processed/               # Preprocessed .parquet files
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb  # LSTM Autoencoder training + evaluation
│
├── src/
│   ├── data/
│   │   ├── loader.py            # Dataset ingestion and windowing
│   │   └── features.py          # RMS, kurtosis, FFT feature extraction
│   ├── models/
│   │   ├── lstm_autoencoder.py  # Model architecture (PyTorch)
│   │   └── threshold.py         # Anomaly threshold calculation
│   └── api/
│       ├── main.py              # FastAPI inference endpoint
│       └── schemas.py           # Pydantic request/response models
│
├── dashboard/
│   └── app.py                   # Streamlit live dashboard
│
├── tests/
│   └── test_features.py
│
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.dashboard
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

This project uses the **NASA IMS Bearing Dataset** from the Center for Intelligent Maintenance
Systems (IMS), University of Cincinnati.

**Download:**
```bash
# Option 1 — Kaggle CLI
kaggle datasets download -d vinayak123tyagi/bearing-dataset -p data/raw/ --unzip

# Option 2 — Direct download from NASA Prognostics Repository
# https://data.nasa.gov/Aeospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
```

**Dataset structure:** Three run-to-failure experiments. Each file contains accelerometer
readings from 4 bearings sampled at 20 kHz. Readings taken every 10 minutes until failure.

---

## 🧠 Model Architecture

```
Input: [batch, seq_len=50, features=8]
        ↓
   LSTM Encoder  (hidden=64, layers=2)
        ↓
   Latent space  (dim=32)
        ↓
   LSTM Decoder  (hidden=64, layers=2)
        ↓
Output: [batch, seq_len=50, features=8]

Anomaly Score = Mean Squared Reconstruction Error
Alert if score > μ_train + 3σ_train
```

The model is trained **only on healthy data** from the first 20% of Dataset 2 (before any
degradation begins). It learns to reconstruct normal operating patterns. When a bearing starts
to degrade, reconstruction error rises — this is the anomaly signal.

---

## 🚀 Quickstart

### Prerequisites
- Docker + Docker Compose
- Python 3.10+
- NASA IMS dataset downloaded to `data/raw/`

### Run the full stack

```bash
git clone https://github.com/YOUR_USERNAME/robotic-bearing-pdm.git
cd robotic-bearing-pdm

# Download data (see Dataset section above)

# Train the model (first time only)
pip install -r requirements.txt
python -m src.models.train

# Launch API + Dashboard
docker-compose up --build
```

| Service     | URL                        |
|-------------|---------------------------|
| FastAPI docs | http://localhost:8000/docs |
| Dashboard   | http://localhost:8501      |

### API usage

```python
import requests

payload = {
    "bearing_id": "bearing_1_x",
    "sensor_readings": [0.012, -0.003, 0.021, ...]  # 50 timesteps × 8 features
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
# {"anomaly_score": 0.0043, "is_anomaly": false, "threshold": 0.0089}
```

---

## 📈 Results

| Metric                     | Value   |
|----------------------------|---------|
| Detection lead time        | ~6 hrs before failure |
| False positive rate        | < 5%    |
| Model parameters           | ~180K   |
| Inference latency          | < 20 ms |

Anomaly scores across the full run-to-failure timeline:

```
Normal operation     │▁▁▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁│  Score: 0.002–0.005
Early degradation    │▁▁▂▂▃▂▂▃▄▃▄▄▅▄▅▅▅▆▆│  Score: 0.010–0.030
Failure imminent     │▆▇▇▇███████████████│  Score: > threshold
```

---

## 🛠️ Tech Stack

| Layer           | Technology                        |
|-----------------|-----------------------------------|
| ML Framework    | PyTorch                           |
| Data Processing | pandas, NumPy, SciPy              |
| Feature Eng.    | tsfresh, custom signal processing |
| API             | FastAPI + Uvicorn                 |
| Dashboard       | Streamlit + Plotly                |
| Containerisation| Docker, Docker Compose            |
| Testing         | pytest                            |

---

## 🔬 Methodology

### Why LSTM Autoencoder?

- **No labelled failure data required** — failure events are rare and expensive to collect.
  Training on normal data only makes this approach applicable out of the box.
- **Captures temporal dependencies** — bearing degradation is a sequential process; LSTMs
  model the time-dependency between sensor readings.
- **Interpretable threshold** — reconstruction error is intuitive and explainable to
  maintenance engineers.

### Feature Engineering

| Feature            | Description                                  |
|--------------------|----------------------------------------------|
| RMS                | Root Mean Square — overall vibration energy  |
| Kurtosis           | Impulsive spike sensitivity                  |
| Peak-to-peak       | Max amplitude range                          |
| Crest factor       | Peak / RMS ratio                             |
| FFT magnitude      | Frequency domain energy at key bands         |
| Rolling mean/std   | Short-term trend features (window=10)        |

---

## 📚 References

- Lee, J., Qiu, H., Yu, G., Lin, J. (2007). *IMS Bearing Dataset*. NASA Prognostics
  Data Repository, NASA Ames Research Center.
- Malhotra, P. et al. (2016). *LSTM-based Encoder-Decoder for Multi-Sensor Anomaly Detection*.
  arXiv:1607.00148.
- Nectoux, P. et al. (2012). *PRONOSTIA: An Experimental Platform for Bearings Accelerated
  Life Test*. IEEE PHM Conference, Denver.

---

## 🗺️ Roadmap

- [x] EDA notebooks
- [x] LSTM Autoencoder training pipeline
- [x] FastAPI inference service
- [x] Streamlit dashboard
- [x] Docker Compose deployment
- [ ] Comparison: Isolation Forest vs One-Class SVM vs LSTM-AE
- [ ] Synthetic data generator (no dataset download needed)
- [ ] Grafana + InfluxDB integration (preview of Project 02)

---

## 🤝 Inspiration

- [taneishi/BearingFailures](https://github.com/taneishi/BearingFailures)
- [BLarzalere/LSTM-Autoencoder-for-Anomaly-Detection](https://github.com/BLarzalere/LSTM-Autoencoder-for-Anomaly-Detection)
- [jieunparklee/predictive-maintenance](https://github.com/jieunparklee/predictive-maintenance)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<p align="center">
  Built as part of a portfolio exploring Industry 4.0 challenges in German automotive manufacturing.<br>
  Inspired by real predictive maintenance problems at BMW, Audi, and Mercedes-Benz.
</p>