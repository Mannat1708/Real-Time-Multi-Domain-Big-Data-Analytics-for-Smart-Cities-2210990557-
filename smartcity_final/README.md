# Real-Time Multi-Domain Big Data Analytics for Smart Cities

**Author:** Mannat  
**University:** Chitkara University Institute of Engineering and Technology  
**Paper:** Real-Time Multi-Domain Big Data Analytics for Smart Cities: Integration, Privacy, and Performance Evaluation

---

## Project Overview

This project implements a real-time multi-domain big data analytics framework
for smart cities. It integrates traffic, energy, and environmental data streams
and applies machine learning models to provide predictive insights and anomaly
detection for city administrators.

---

## Models Used

| Domain | Model | Task | Metric |
|---|---|---|---|
| Traffic | Random Forest | Congestion prediction | Accuracy, F1 |
| Energy | Linear Regression | kWh demand forecast | RMSE, R² |
| Environment | Isolation Forest | Anomaly detection | F1, Precision |

---

## Folder Structure

```
smartcity/
│
├── data/                          ← Place your CSV files here
│   ├── Metro_Interstate_Traffic_Volume.csv
│   ├── smart_home_energy.csv
│   └── AirQualityUCI.csv
│
├── outputs/                       ← Generated graphs saved here
│   ├── graph1_accuracy_comparison.png
│   ├── graph2_latency_comparison.png
│   ├── graph3_throughput.png
│   ├── graph4_scalability.png
│   └── graph5_f1_breakdown.png
│
├── smartcity_analysis.ipynb       ← Main Jupyter Notebook
├── smartcity_analysis.py          ← Plain Python version
├── requirements.txt               ← Required libraries
└── README.md                      ← This file
```

---

## Datasets

Download and place in the `data/` folder:

| Dataset | Source | Link |
|---|---|---|
| Metro Interstate Traffic Volume | Kaggle | https://www.kaggle.com/datasets/anshtanwar/metro-interstate-traffic-volume |
| Smart Home Energy Consumption | Kaggle | https://www.kaggle.com/datasets/mexwell/smart-home-energy-consumption |
| Air Quality UCI | UCI Repository | https://archive.ics.uci.edu/dataset/387/air+quality |

---

## Installation

Open Anaconda Prompt and run:

```bash
conda install pandas numpy scikit-learn matplotlib seaborn
```

Or using pip:

```bash
pip install -r requirements.txt
```

---

## How to Run

1. Download the three datasets and place CSV files in the `data/` folder
2. Open Anaconda Navigator → Launch Jupyter Notebook
3. Navigate to the project folder and open `smartcity_analysis.ipynb`
4. Run each cell using **Shift + Enter**
5. Graphs are saved automatically in the `outputs/` folder

---

## Graphs Generated

| Graph | Description |
|---|---|
| Graph 1 | ML Model Accuracy Comparison — Proposed vs Baseline |
| Graph 2 | Processing Latency — Batch vs Real-Time Streaming |
| Graph 3 | System Throughput vs Number of Spark Worker Nodes |
| Graph 4 | Scalability — Proposed vs Existing System |
| Graph 5 | Precision, Recall and F1 Score per Domain |

---

## Technologies Used

- Python 3.10
- Scikit-learn 1.3
- Apache Kafka (simulated via data streams)
- Apache Spark Structured Streaming (simulated)
- HDFS + MongoDB (architecture)
- Matplotlib for visualisation
