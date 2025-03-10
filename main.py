from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the trained model pipeline (StandardScaler + Lasso)
model_pipeline = joblib.load("best_model.joblib")

# Define request body structure
class InputFeatures(BaseModel):
    highest_value: int

def preprocessing(input_features: InputFeatures):
    """Applies the same preprocessing steps as the training pipeline."""
    # Convert input to DataFrame with correct column names
    processed_data = pd.DataFrame([[input_features.highest_value]], columns=['highest_value'])

    # Ensure feature order is correct
    processed_data = processed_data.reindex(columns=model_pipeline.feature_names_in_)

    return processed_data

# GET request (Test API)
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

# GET and POST request for items
@app.get("/items/{item_id}")
@app.post("/items/{item_id}")
def create_item(item_id: int):
    return {"item": item_id}

# POST request for prediction
@app.post("/predict")
async def predict(input_features: InputFeatures):
    try:
        # Preprocess the input
        data = preprocessing(input_features)

        # Debugging: Check if input is scaled before prediction
        print("Raw Input:", data.values)
        print("Scaled Input Before Prediction:", model_pipeline.named_steps['standardscaler'].transform(data))

        # Make prediction using the full pipeline (StandardScaler + Lasso)
        y_pred = model_pipeline.predict(data)

        return {"pred": y_pred.tolist()[0]}
    except Exception as e:
        print("❌ ERROR:", str(e))  # Print error in the terminal for debugging
        raise HTTPException(status_code=500, detail=str(e))
