# ==========================================
# CROP YIELD PREDICTION USING MACHINE LEARNING
# ==========================================

# Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("dataset/yield_df.csv")

# Display First 5 Rows
print("First 5 Rows of Dataset")
print(df.head())

# Shape
print("\nShape of Dataset:")
print(df.shape)

# Columns
print("\nColumns:")
print(df.columns)

# Information
print("\nDataset Information:")
df.info()

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# Data Cleaning
# ==========================================

# Remove Unnecessary Column
df.drop("Unnamed: 0", axis=1, inplace=True)

print("\nColumns After Removing Unnecessary Column:")
print(df.columns)

# Check Duplicate Values
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ==========================================
# Distribution of Crop Yield
# ==========================================

plt.figure(figsize=(8,5))
sns.histplot(df["hg/ha_yield"], bins=30, kde=True)

plt.title("Distribution of Crop Yield")
plt.xlabel("Yield (hg/ha)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("images/crop_yield_distribution.png")
plt.show()

# ==========================================
# Correlation Heatmap
# ==========================================

numeric_df = df.select_dtypes(include=["int64","float64"])

plt.figure(figsize=(8,6))

sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu")

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.show()

# ==========================================
# Top 10 Crop Distribution
# ==========================================

plt.figure(figsize=(10,5))

df["Item"].value_counts().head(10).plot(kind="bar")

plt.title("Top 10 Crops in Dataset")
plt.xlabel("Crop")
plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("images/top10_crops.png")
plt.show()

plt.figure(figsize=(10,5))

df["Area"].value_counts().head(10).plot(kind="bar")

plt.title("Top 10 Areas")
plt.xlabel("Area")
plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("images/top10_areas.png")
plt.show()

# ==========================================
# Import ML Libraries
# ==========================================

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ==========================================
# Label Encoding
# ==========================================

label_encoder_area = LabelEncoder()
label_encoder_item = LabelEncoder()

df["Area"] = label_encoder_area.fit_transform(df["Area"])
df["Item"] = label_encoder_item.fit_transform(df["Item"])

print("\nFirst 5 Rows After Encoding:")
print(df.head())

# ==========================================
# Features and Target
# ==========================================

X = df.drop("hg/ha_yield", axis=1)
y = df["hg/ha_yield"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFirst 5 Features:")
print(X.head())

print("\nFirst 5 Target Values:")
print(y.head())

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

# ==========================================
# Import Regression Models
# ==========================================

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

# ==========================================
# Linear Regression
# ==========================================

print("\n========== Linear Regression ==========")

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("R2 Score :", round(r2_score(y_test, y_pred_lr), 4))
print("MAE      :", round(mean_absolute_error(y_test, y_pred_lr), 4))
print("RMSE     :", round(root_mean_squared_error(y_test, y_pred_lr), 4))


# ==========================================
# Random Forest Regressor
# ==========================================

print("\n========== Random Forest Regressor ==========")

rf = RandomForestRegressor(
    n_estimators=20,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("R2 Score :", round(r2_score(y_test, y_pred_rf), 4))
print("MAE      :", round(mean_absolute_error(y_test, y_pred_rf), 4))
print("RMSE     :", round(root_mean_squared_error(y_test, y_pred_rf), 4))

# ==========================================
# Gradient Boosting Regressor
# ==========================================

print("\n========== Gradient Boosting Regressor ==========")

gbr = GradientBoostingRegressor(random_state=42)

gbr.fit(X_train, y_train)

y_pred_gbr = gbr.predict(X_test)

print("R2 Score :", round(r2_score(y_test, y_pred_gbr), 4))
print("MAE      :", round(mean_absolute_error(y_test, y_pred_gbr), 4))
print("RMSE     :", round(root_mean_squared_error(y_test, y_pred_gbr), 4))

print("\n======================================")
print("        Model Comparison")
print("======================================")

print(f"Linear Regression       : {r2_score(y_test, y_pred_lr):.4f}")
print(f"Random Forest Regressor : {r2_score(y_test, y_pred_rf):.4f}")
print(f"Gradient Boosting       : {r2_score(y_test, y_pred_gbr):.4f}")

print("\nBest Model: Random Forest Regressor")


# ==========================================
# Save Best Model
# ==========================================

import joblib

joblib.dump(rf, "models/crop_yield_model.pkl")

# Save Label Encoders

joblib.dump(label_encoder_area, "models/area_encoder.pkl")
joblib.dump(label_encoder_item, "models/item_encoder.pkl")

print("Encoders Saved Successfully!")

print("\n======================================")
print("Best Model Saved Successfully!")
print("Model Name : Random Forest Regressor")
print("Location   : models/crop_yield_model.pkl")
print("======================================")