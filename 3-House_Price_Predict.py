import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as train_test_split
import sklearn.tree as DecisionTreeRegressor

print("=== Phase 2: Load & Understand the Dataset ===")
df = pd.read_excel('House Price.xlsx')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print("\n Columns in the dataset:", df.columns)

print("=== Phase 2 - Exploratory Data Analysis (EDA) ===")
# Part 1 – Target Variable Analysis
print(" Sales Price Statistics:")
print(df['SalePrice'].describe())

# Part 2 – Categorical Features vs SalePrice
print(" Categorical Features vs SalePrice:")
print(df.groupby('Neighborhood')['SalePrice'].mean())
print(df.groupby('HouseStyle')['SalePrice'].mean())
print(df.groupby('BldgType')['SalePrice'].mean())
print(df.groupby('Foundation')['SalePrice'].mean())
print(df.groupby('GarageType')['SalePrice'].mean())
print(df.groupby('SaleCondition')['SalePrice'].mean())

# Part 3 – Numerical Features vs SalePrice
print(" Numerical Features vs SalePrice:")
print(df.groupby('OverallQual')['SalePrice'].mean())
print(df.groupby('OverallCond')['SalePrice'].mean())
print(df.groupby('GarageCars')['SalePrice'].mean())
print(df.groupby('FullBath')['SalePrice'].mean())
print(df.groupby('BedroomAbvGr')['SalePrice'].mean())

# Use correlation for continuous features
print(" Correlation for continuous features:")
print(df[["SalePrice",
          "GrLivArea",
          "LotArea",
          "GarageArea",
          "TotalBsmtSF",
          "YearBuilt"]].corr()
          )

print("=== Phase 3 - Feature Selection ===")
# Drop Unnecessary Columns
df.drop("Id", axis=1, inplace=True)
# Define Input (X) and Target (y)
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]
# Check Shape of X and y
print("Shape of X = ", X.shape)
print("Shape of y = ", y.shape)
# Check Feature Names
print("Feature Names = ", X.columns)

print("=== Phase 4 - Data Preprocessing ===")
# Step 1: Handle Missing Values: We divide missing value into 3 categories.

# Category 1: Missing values in Numerical features fill with median
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].median())
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["GarageYrBlt"].median())

# Category 2: Missing values in Categorical features fill with mode
df["MasVnrType"] = df["MasVnrType"].fillna(df["MasVnrType"].mode()[0])
df["BsmtQual"] = df["BsmtQual"].fillna(df["BsmtQual"].mode()[0])
df["BsmtCond"] = df["BsmtCond"].fillna(df["BsmtCond"].mode()[0])
df["BsmtExposure"] = df["BsmtExposure"].fillna(df["BsmtExposure"].mode()[0])
df["BsmtFinType1"] = df["BsmtFinType1"].fillna(df["BsmtFinType1"].mode()[0])
df["BsmtFinType2"] = df["BsmtFinType2"].fillna(df["BsmtFinType2"].mode()[0])
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
df["GarageType"] = df["GarageType"].fillna(df["GarageType"].mode()[0])
df["GarageFinish"] = df["GarageFinish"].fillna(df["GarageFinish"].mode()[0])
df["GarageQual"] = df["GarageQual"].fillna(df["GarageQual"].mode()[0])
df["GarageCond"] = df["GarageCond"].fillna(df["GarageCond"].mode()[0])
df["FireplaceQu"] = df["FireplaceQu"].fillna(df["FireplaceQu"].mode()[0])

# Category C —  Drop Columns with Too Many Missing Values
df.drop(["Alley",
         "PoolQC",
         "Fence",
         "MiscFeature"], axis=1 , inplace=True)

print("=== Step - 5 : Encoding ===")
# Label Encoding 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Street"] = le.fit_transform(df["Street"])
df["CentralAir"] = le.fit_transform(df["CentralAir"])

# One-Hot Encoding 
categorical_columns = df.select_dtypes(include='object').columns
df = pd.get_dummies(
     df,
     columns=categorical_columns,
     drop_first=True,
     dtype=int
)

print("== Again Encoding ==")
# Input Features
X = df.drop("SalePrice", axis=1)

# Target Variable
y = df["SalePrice"]

print("Shape of X =", X.shape)
print("Shape of y =", y.shape)

print("=== Phase 5 : Train Test Split ===")
from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(X,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=42
                                                      )
print("\n Training Set Shape ")
print("X_train = ",X_train.shape)
print("y_train = ",y_train.shape)

print("\n Testing Set Shape ")
print("X_test  = ", X_test.shape)
print("Y_test  = ", y_test.shape)

print("=== Phase 6 - Decision Tree Regression ===")
from sklearn.tree import DecisionTreeRegressor
# Create Model
model = DecisionTreeRegressor(random_state=42)
# Train the Model
model.fit(X_train , y_train)
print("Model trained successfully using DecisionTreeRegressor.")

print("=== Phase 7 - Prediction ")
# predict House Price 
y_pred = model.predict(X_test)
print("First 10 Predicted Prices:")
print(y_pred[:10])
print("\nActual Prices:")
print(y_test.values[:10])

print("===  Phase 8 - Model Evaluation ===")
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

#  Calculate Metrics 
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = root_mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

# Result 
print("MAE  =", mae)

print("MSE  =", mse)

print("RMSE =", rmse)

print("R2 Score =", r2)

print( "=======  Phase 9 - Hyperparameter Tuning  ========")

# Create Tuned Model
model = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
print("== Again Train The Model ===")
model.fit(X_train, y_train)
#  Prediction
y_pred = model.predict(X_test)

print("==== Evaluate Again ===== ")
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)

print("=== Hyperparameter Tuning Results ===")

print("MAE =", mean_absolute_error(y_test, y_pred))

print("MSE =", mean_squared_error(y_test, y_pred))

print("RMSE =", root_mean_squared_error(y_test, y_pred))

print("R2 Score =", r2_score(y_test, y_pred))

import matplotlib.pyplot as plt
import os

# Create images folder if it doesn't exist
os.makedirs("images", exist_ok=True)

# Metrics
metrics = ["MAE", "RMSE", "R² Score"]

before = [27604.67, 42009.00, 0.7699]
after = [24325.46, 39443.25, 0.7972]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(8,5))

plt.bar([i-width/2 for i in x], before, width=width, label="Before Tuning")
plt.bar([i+width/2 for i in x], after, width=width, label="After Tuning")

plt.xticks(x, metrics)

plt.title("Model Performance Before vs After Hyperparameter Tuning")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Metric Values")
plt.legend()

plt.tight_layout()

plt.savefig("images/before_after_tuning.png", dpi=300)

plt.show()