# ⚡ ML-Based Bhopal Electricity Consumption Analysis

### End-to-End Data Analysis & Machine Learning Project (Google Colab)

---

## 📌 Overview

This project presents a complete **data science and machine learning pipeline** to analyze and predict **electricity consumption patterns in Madhya Pradesh (Bhopal region)**.

Developed in **Google Colab using Python**, this project demonstrates:

* 📊 Data preprocessing & merging from multiple sources
* 📈 Exploratory Data Analysis (EDA)
* 🔍 Feature correlation & distribution analysis
* 🤖 Machine Learning preprocessing pipeline
* ⚡ Scalable and structured workflow

---

## 🧠 Key Highlights

✔ Real-world dataset integration (weather + energy)
✔ Advanced visualization techniques
✔ Clean preprocessing pipeline using `ColumnTransformer`
✔ Feature scaling using `StandardScaler`
✔ Modular and reusable ML workflow

---

## 🏗️ Project Workflow

### 🔄 Step-by-Step Pipeline

```id="flow123"
1. Import Libraries
2. Mount Google Drive
3. Load Datasets (CSV + Excel)
4. Data Cleaning & Preprocessing
5. Merge Datasets on Date
6. Handle Missing Values
7. Exploratory Data Analysis (EDA)
8. Feature Engineering
9. Train-Test Split
10. Preprocessing Pipeline (Scaling)
11. Model Training (XGBoost)
12. Evaluation & Predictions
```

---

## 📦 Libraries Used

```python
pandas, numpy
xgboost
matplotlib, seaborn
sklearn (metrics, preprocessing, model_selection)
```

---

## 📂 Data Sources

* **Factor Data (Weather Data)** → `Madhya_Pradesh.csv`
* **Energy Data** → `energy.xlsx`

These datasets are merged on a common **date column** to create a unified dataset.

---

## 🔗 Data Merging Process

```python
factor_data['date'] = pd.to_datetime(factor_data['date'])
energy_data['Date'] = pd.to_datetime(energy_data['Date'])

energy_data = energy_data.rename(columns={'Date': 'date'})
merged_data = pd.merge(factor_data, energy_data, on='date', how='inner')
```

### ✅ Purpose:

* Align weather + energy data
* Enable feature-based prediction

---

## 🧹 Handling Missing Values

```python
for column in merged_data.columns:
    if merged_data[column].dtype != 'datetime64[ns]':
        if merged_data[column].isnull().any():
            merged_data[column].fillna(merged_data[column].mean(), inplace=True)
```

### ✅ Strategy:

* Replace missing values with **mean**
* Avoid modifying date column

---

## 📊 Exploratory Data Analysis (EDA)

### 📈 Time Series Visualization

* Daily Energy Consumption
* Peak Energy Demand

### 🔥 Correlation Heatmap

* Shows relationships between all numerical features
* Helps identify important predictors

### 📉 Distribution Analysis

* Temperature
* Solar Radiation
* Cloud Cover
* UTCI

### 🔍 Pair Plot

* Multi-variable relationship visualization

---

## 🧠 Feature Engineering

### Selected Important Features:

* Temperature (`2m_temperature_mean`)
* Solar Radiation
* Cloud Cover
* UTCI
* Energy Metrics (`daily_energy_met_MU`, `peak_met_MW`)

---

## ⚙️ Machine Learning Preprocessing Pipeline

### 🎯 Objective

To standardize numerical features using a scalable and reusable pipeline.

---

### 🔧 Step 1: Identify Numerical Features

```python
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
```

---

### 🔧 Step 2: Create ColumnTransformer Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='passthrough'
)
```

---

### 🔧 Step 3: Apply Pipeline

```python
X_train_scaled_pipeline = preprocessor.fit_transform(X_train)
X_test_scaled_pipeline = preprocessor.transform(X_test)
```

---

### 🔧 Step 4: Verify Output

```python
print(X_train_scaled_pipeline.shape)
print(X_test_scaled_pipeline.shape)

print(X_train_scaled_pipeline[:5])
```

---

## 📊 Why Use a Preprocessing Pipeline?

### ✅ Key Benefits

#### 1. 🔁 Reusability

* Same preprocessing applied to training & testing data
* Avoids duplication of code

#### 2. 🧠 Consistency

* Prevents data leakage
* Ensures identical transformations

#### 3. ⚡ Efficiency

* Combines multiple preprocessing steps into one object

#### 4. 🧩 Scalability

* Easy to extend (add encoding, feature selection, etc.)

#### 5. 🚀 Production Ready

* Pipeline can be directly used in deployment

---

## 🤖 Model (XGBoost)

The project uses:

```python
import xgboost as xgb
```

### Why XGBoost?

* High performance
* Handles complex relationships
* Works well with structured data

---

## 📈 Evaluation Metrics

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* R² Score

---

## 📊 Visual Outputs

* Time series plots
* Correlation heatmap
* Feature distributions
* Pair plots

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

## 📊 Use Cases

* ⚡ Electricity demand forecasting
* 🌤 Weather impact analysis
* 📈 Energy optimization
* 🧠 Data science learning

---

## 🌟 Highlights

✔ Multi-source dataset merging
✔ Strong EDA foundation
✔ Advanced preprocessing pipeline
✔ Real-world ML problem solving
✔ Clean and modular code

---

## 🧩 Future Improvements

* 📊 Add time-series models (LSTM, ARIMA)
* 🌐 Build web dashboard
* ⚡ Real-time prediction system
* 📈 Hyperparameter tuning
* 🧠 Deep learning models

---

## 👨‍💻 Author

**Your Name**

* Data Science & ML Enthusiast
* Passionate about solving real-world problems using AI

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 💡 Final Note

This project demonstrates how combining **weather data + machine learning** can provide powerful insights into **electricity consumption patterns**.

A strong portfolio project for aspiring **Data Scientists & ML Engineers 🚀**

---
