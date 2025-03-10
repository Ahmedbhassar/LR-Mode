import uvicorn
import os
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the trained model pipeline (StandardScaler + Lasso)
try:
    model_pipeline = joblib.load("best_model.joblib")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model_pipeline = None  # Prevents crashing if model fails to load

# Define request body structure
class InputFeatures(BaseModel):
    highest_value: int

def preprocessing(input_features: InputFeatures):
    """Applies the same preprocessing steps as the training pipeline."""
    try:
        # Convert input to DataFrame with correct column names
        processed_data = pd.DataFrame([[input_features.highest_value]], columns=['highest_value'])

        # Ensure feature order matches the trained model
        processed_data = processed_data.reindex(columns=model_pipeline.feature_names_in_)

        return processed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

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
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Preprocess input
        data = preprocessing(input_features)

        # Make prediction
        y_pred = model_pipeline.predict(data)

        return {"pred": y_pred.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Explicitly define the entry point (fixes Render port issue)
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use Render's assigned port or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
