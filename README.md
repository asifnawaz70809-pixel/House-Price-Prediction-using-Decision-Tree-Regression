# House-Price-Prediction-using-Decision-Tree-Regression# 🏠 House Price Prediction using Decision Tree Regression

## 📌 Project Overview

This project predicts house prices using the **Decision Tree Regression** algorithm. The objective is to build a Machine Learning model that can estimate the selling price of a house based on its features such as location, size, quality, garage, basement, and other property characteristics.

This project follows a complete Machine Learning workflow, including data understanding, exploratory data analysis, preprocessing, model training, evaluation, and hyperparameter tuning.

---

## 🎯 Problem Statement

House prices are influenced by many factors, making manual estimation difficult. The goal of this project is to train a regression model that learns the relationship between house features and their selling prices to provide accurate price predictions.

---

## 📂 Dataset

* **Dataset:** House Prices – Advanced Regression Techniques
* **Source:** Kaggle
* **Total Records:** 1460
* **Total Features:** 81
* **Target Variable:** SalePrice

---

## 🤖 Machine Learning Details

* **Learning Type:** Supervised Learning
* **Problem Type:** Regression
* **Algorithm:** Decision Tree Regressor

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## 📊 Project Workflow

1. Problem Understanding
2. Data Loading
3. Data Understanding
4. Exploratory Data Analysis (EDA)
5. Feature Selection
6. Data Preprocessing
7. Missing Value Handling
8. One-Hot Encoding
9. Train-Test Split
10. Decision Tree Regression Model
11. Model Training
12. Prediction
13. Model Evaluation
14. Hyperparameter Tuning
15. Final Model Analysis

---

## 📈 Model Evaluation

### Initial Model

* MAE: 27,604.67
* MSE: 1,764,755,665.04
* RMSE: 42,009.00
* R² Score: 0.7699

### Tuned Model

Hyperparameters:

* max_depth = 10
* min_samples_split = 20
* min_samples_leaf = 10

Results:

* MAE: 24,325.46
* MSE: 1,555,769,873.37
* RMSE: 39,443.25
* R² Score: 0.7972

The tuned model achieved better performance by reducing prediction error and improving the R² Score.

---

## 📚 Concepts Applied

* Data Loading
* Data Cleaning
* Missing Value Handling
* One-Hot Encoding
* Feature Selection
* Exploratory Data Analysis (EDA)
* Decision Tree Regression
* Model Training
* Prediction
* Regression Metrics (MAE, MSE, RMSE, R² Score)
* Hyperparameter Tuning

---

## 📁 Project Structure

```text
House-Price-Prediction-Decision-Tree/
│
├── dataset/
├── images/
├── House_Price_Prediction.ipynb
├── House_Price_Prediction.py
├── requirements.txt
└── README.md
```

---

## 🚀 Future Improvements

* Apply Random Forest Regression
* Compare multiple regression algorithms
* Perform feature engineering
* Optimize hyperparameters using GridSearchCV
* Deploy the trained model as a web application

---

## 👨‍💻 Author

**Asif Nawaz**

Computer Science Student | Aspiring AI & Machine Learning Engineer

Currently building real-world Machine Learning projects to strengthen practical skills in Data Science and Artificial Intelligence.
