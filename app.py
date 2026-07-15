# ==========================================
# CROP YIELD PREDICTION APPLICATION
# ==========================================

import joblib
import pandas as pd

# Load Model and Encoders
model = joblib.load("models/crop_yield_model.pkl")
area_encoder = joblib.load("models/area_encoder.pkl")
item_encoder = joblib.load("models/item_encoder.pkl")

print("===================================")
print(" Crop Yield Prediction System")
print("===================================\n")

# User Input
area = input("Enter Area: ")
item = input("Enter Crop: ")
year = int(input("Enter Year: "))
rainfall = float(input("Enter Average Rainfall (mm/year): "))
pesticides = float(input("Enter Pesticides (tonnes): "))
temperature = float(input("Enter Average Temperature (°C): "))

# Encode Inputs
try:
    area_encoded = area_encoder.transform([area])[0]
    item_encoded = item_encoder.transform([item])[0]
except ValueError:
    print("\nInvalid Area or Crop Name!")
    print("Please enter values exactly as present in the dataset.")
    exit()

# Create DataFrame (removes sklearn warning)
input_data = pd.DataFrame({
    "Area": [area_encoded],
    "Item": [item_encoded],
    "Year": [year],
    "average_rain_fall_mm_per_year": [rainfall],
    "pesticides_tonnes": [pesticides],
    "avg_temp": [temperature]
})

# Prediction
prediction = model.predict(input_data)

print("\n===================================")
print("Estimated Crop Yield:", round(prediction[0], 2), "hg/ha")
print("===================================")