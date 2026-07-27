# 🚦 Smart City Traffic Patterns Forecasting

An end-to-end Machine Learning project that predicts traffic volume at different city junctions using historical traffic data. The project involves data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment through a Streamlit web application.

---

## 📌 Project Overview

Traffic congestion is a major challenge in smart cities. Accurate traffic prediction helps improve traffic management, optimize signal timing, reduce congestion, and support better urban planning.

This project predicts the number of vehicles passing through a junction based on historical traffic patterns using Machine Learning.

---

## 🎯 Objectives

- Analyze historical traffic data.
- Perform Exploratory Data Analysis (EDA).
- Engineer meaningful features from DateTime.
- Train and compare multiple regression models.
- Select the best-performing model.
- Deploy the model using Streamlit.

---

## 📂 Dataset

The dataset contains historical traffic information with the following features:

| Feature | Description |
|---------|-------------|
| ID | Unique record identifier |
| DateTime | Date and time of observation |
| Junction | Junction number |
| Vehicles | Number of vehicles (Target Variable) |

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

## 📊 Project Workflow

```
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Feature Engineering
    │
    ▼
Train-Test Split
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ▼
Model Saving
    │
    ▼
Streamlit Deployment
```

---

## 🔍 Exploratory Data Analysis

Performed:

- Dataset inspection
- Missing value analysis
- Duplicate value checking
- Traffic distribution analysis
- Junction-wise traffic analysis
- Time-based traffic analysis
- Correlation analysis

---

## ⚙ Feature Engineering

Extracted the following features from the **DateTime** column:

- Year
- Month
- Day
- Hour
- Day of Week

Performed One-Hot Encoding for:

- DayOfWeek

Removed unnecessary columns:

- ID
- Original DateTime

---

## 🤖 Machine Learning Models

The following regression models were evaluated:

- Linear Regression
- Random Forest Regressor ✅ (Selected)
- XGBoost 

---

## 📈 Model Performance

### Linear Regression

| Metric | Score |
|---------|--------|
| MAE | 9.58 |
| RMSE | 12.69 |
| R² Score | 0.605 |


### XGBoost Regressor

| Metric | Score |
|---------|--------|
| MAE | 2.43 |
| RMSE | 3.74 |
| R² Score | 0.966 |

### Random Forest Regressor (Final Model)

| Metric | Score |
|---------|--------|
| MAE | 2.43 |
| RMSE | 3.74 |
| R² Score | 0.966 |

Random Forest significantly outperformed Linear Regression and was selected as the final model.

---

## 🚀 Streamlit Application

The project includes a Streamlit application that allows users to:

- Select Junction
- Choose Date
- Select Hour
- Predict Traffic Volume instantly

---

## 📁 Project Structure

```
Smart-City-Traffic-Forecasting/
│
├── Data/
│
├── Models/
│   └── traffic_prediction_model.pkl
│
├── Notebook/
│   └── Notebook.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── traffic_predictions.csv
```

---


Example:

```
images/
    home.png
    prediction.png
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Smart-City-Traffic-Forecasting.git
```

Move to the project directory

```bash
cd Smart-City-Traffic-Forecasting
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 💡 Future Improvements

- Add weather data for improved prediction.
- Include holidays and special events.
- Deploy the application on Streamlit Community Cloud.
- Experiment with XGBoost and LightGBM.
- Build a real-time traffic prediction dashboard.

---

## 👨‍💻 Author

**Kuldeep Kirit Prajapati**

B.Tech Computer Science & Engineering

Interested in Machine Learning, Data Analytics, Deep Learning, and Generative AI.

GitHub: https://github.com/yourusername

LinkedIn: https://linkedin.com/in/yourprofile

---

## ⭐ If you found this project useful, consider giving it a star!