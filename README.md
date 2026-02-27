# ⚡ ML-Based Bhopal Electricity Consumption Analysis  
### *An End-to-End Data Science & Machine Learning Pipeline*

The **ML-Based Bhopal Electricity Consumption Analysis** project is a comprehensive data science pipeline designed to analyze and predict electricity usage patterns in the **Bhopal region (Madhya Pradesh)**.

Built using **Python in Google Colab**, this project integrates **weather data and energy consumption datasets** to uncover meaningful insights and build predictive machine learning models.

It demonstrates a complete workflow — from **data preprocessing and exploratory analysis to model training and evaluation**, making it a strong portfolio project for aspiring data scientists.

---

<p align="center">
  <strong>⚡ Energy AI Analytics</strong><br/>
  <em>Data → Insights → Predictions</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/ML-XGBoost-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/DataScience-EDA-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Visualization-Matplotlib%20%7C%20Seaborn-lightgrey?style=flat-square"/>
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-yellow?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square"/>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Highlights](#-key-highlights)
- [Workflow](#-workflow)
- [Technology Stack](#-technology-stack)
- [Data Sources](#-data-sources)
- [Data Processing](#-data-processing)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering](#-feature-engineering)
- [ML Pipeline](#-ml-pipeline)
- [Model](#-model)
- [Evaluation](#-evaluation-metrics)
- [How to Run](#-how-to-run)
- [Use Cases](#-use-cases)
- [Future Enhancements](#-future-enhancements)

---

## 🌟 Overview

This project focuses on building a **scalable machine learning pipeline** for analyzing electricity consumption patterns by combining:

- ⚡ Energy usage data  
- 🌤 Weather conditions  
- 📊 Statistical and ML techniques  

By merging these datasets, the system identifies patterns and predicts energy demand, helping in **better energy planning and optimization**.

---

## 🧠 Key Highlights

✔ Multi-source dataset integration (weather + energy)  
✔ Advanced data visualization techniques  
✔ Robust preprocessing pipeline using `ColumnTransformer`  
✔ Feature scaling using `StandardScaler`  
✔ Modular and reusable ML workflow  
✔ Real-world data science problem  

---

## 🏗 Workflow

### 🔄 Step-by-Step Pipeline

```
1. Import libraries
2. Load datasets (CSV & Excel)
3. Data cleaning & preprocessing
4. Merge datasets on date
5. Handle missing values
6. Perform EDA
7. Feature engineering
8. Train-test split
9. Preprocessing pipeline
10. Model training (XGBoost)
11. Evaluation & prediction
```

---

## 🛠 Technology Stack

| Layer | Technology | Purpose |
|------|-----------|--------|
| **Language** | Python | Core development |
| **ML Model** | XGBoost | Prediction model |
| **Data Handling** | Pandas, NumPy | Data manipulation |
| **Visualization** | Matplotlib, Seaborn | Data analysis |
| **ML Tools** | Scikit-learn | Preprocessing & evaluation |
| **Platform** | Google Colab | Development environment |

---

## 📂 Data Sources

- **Weather Dataset** → `Madhya_Pradesh.csv`  
- **Energy Dataset** → `energy.xlsx`  

Both datasets are merged on a **common date column** to create a unified dataset for analysis.

---

## 🔗 Data Processing

### Dataset Merging

```python
factor_data['date'] = pd.to_datetime(factor_data['date'])
energy_data['Date'] = pd.to_datetime(energy_data['Date'])

energy_data = energy_data.rename(columns={'Date': 'date'})
merged_data = pd.merge(factor_data, energy_data, on='date', how='inner')
```

### Handling Missing Values

```python
for column in merged_data.columns:
    if merged_data[column].dtype != 'datetime64[ns]':
        if merged_data[column].isnull().any():
            merged_data[column].fillna(merged_data[column].mean(), inplace=True)
```

---

## 📊 Exploratory Data Analysis

- 📈 Time series plots (energy consumption trends)  
- 🔥 Correlation heatmap (feature relationships)  
- 📉 Distribution analysis (temperature, radiation, etc.)  
- 🔍 Pair plots (multi-variable relationships)  

---

## 🧠 Feature Engineering

Key features used:

- Temperature (`2m_temperature_mean`)  
- Solar Radiation  
- Cloud Cover  
- UTCI  
- Energy metrics (`daily_energy_met_MU`, `peak_met_MW`)  

---

## ⚙️ ML Pipeline

### Preprocessing Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

numerical_features = X.select_dtypes(include=np.number).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='passthrough'
)
```

### Apply Pipeline

```python
X_train_scaled_pipeline = preprocessor.fit_transform(X_train)
X_test_scaled_pipeline = preprocessor.transform(X_test)
```

---

### 🚀 Why Use a Pipeline?

- 🔁 Reusable transformations  
- 🧠 Consistent preprocessing  
- ⚡ Efficient workflow  
- 🧩 Scalable architecture  
- 🚀 Production-ready  

---

## 🤖 Model

### XGBoost

```python
import xgboost as xgb
```

**Advantages:**
- High accuracy  
- Handles complex relationships  
- Optimized for structured datasets  

---

## 📈 Evaluation Metrics

- Mean Squared Error (MSE)  
- Mean Absolute Error (MAE)  
- R² Score  

---

## ⚡ How to Run

### ▶️ Google Colab
1. Upload notebook  
2. Mount Google Drive  
3. Run all cells  

---

### 💻 Local Setup
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## 🎯 Use Cases

- ⚡ Electricity demand forecasting  
- 🌤 Weather impact analysis  
- 📈 Energy optimization  
- 🧠 Data science learning  

---

## 🌟 Highlights

✔ Real-world dataset integration  
✔ Strong EDA foundation  
✔ Advanced ML preprocessing pipeline  
✔ Clean and modular implementation  
✔ End-to-end ML workflow  

---

## 🔮 Future Enhancements

- 📊 Time-series models (LSTM, ARIMA)  
- 🌐 Interactive dashboard (Streamlit/Flask)  
- ⚡ Real-time prediction system  
- 📈 Hyperparameter tuning  
- 🧠 Deep learning integration  

---

## 👨‍💻 Author

**Vaibhav Sharma**  
*Data Science & ML Enthusiast*

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 💡 Final Note

> Combining **weather data + machine learning** unlocks powerful insights into energy consumption.

This project showcases how data science can be applied to solve **real-world energy challenges 🚀**

---

<p align="center">
  Built with ❤️ using Python & Machine Learning<br/>
  <strong>Energy AI Analytics</strong> — Smarter Predictions for Smarter Cities
</p>
