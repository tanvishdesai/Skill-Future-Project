from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import os
from contextlib import asynccontextmanager
from src.dataset_schema import HeartDiseaseInput, LiverDiseaseInput
from src.explanation import get_explanation

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Heart Model
    try:
        models['heart_model'] = joblib.load('models/heart_model.pkl')
        models['heart_scaler'] = joblib.load('models/heart_scaler.pkl')
        models['heart_features'] = joblib.load('models/heart_features.pkl')
        print("Heart model loaded.")
    except Exception as e:
        print(f"Error loading heart model: {e}")

    # Load Liver Model
    try:
        models['liver_model'] = joblib.load('models/liver_model.pkl')
        models['liver_scaler'] = joblib.load('models/liver_scaler.pkl')
        models['liver_features'] = joblib.load('models/liver_features.pkl')
        if os.path.exists('models/liver_gender_encoder.pkl'):
            models['liver_gender_encoder'] = joblib.load('models/liver_gender_encoder.pkl')
        print("Liver model loaded.")
    except Exception as e:
        print(f"Error loading liver model: {e}")
    
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Medical Disease Prediction API is running"}

@app.post("/predict/heart")
async def predict_heart(data: HeartDiseaseInput):
    if 'heart_model' not in models:
        raise HTTPException(status_code=503, detail="Heart model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # Preprocessing matching training
        # 1. Dummies
        df = pd.get_dummies(df)
        
        # 2. Reindex to match training features (fill missing with 0)
        expected_features = models['heart_features']
        df = df.reindex(columns=expected_features, fill_value=0)
        
        # 3. Scale
        X_scaled = models['heart_scaler'].transform(df)
        
        # Predict
        prediction = models['heart_model'].predict(X_scaled)[0]
        prob = models['heart_model'].predict_proba(X_scaled)[0][1] # Probability of class 1
        
        # Explanation
        explanation = get_explanation("Heart Disease", input_dict, prediction, prob)
        
        return {
            "prediction": int(prediction),
            "probability": float(prob),
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/liver")
async def predict_liver(data: LiverDiseaseInput):
    if 'liver_model' not in models:
        raise HTTPException(status_code=503, detail="Liver model not loaded")
    
    try:
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # Preprocessing
        # 1. Gender Encoding
        if 'liver_gender_encoder' in models:
            # Handle if gender string doesn't match
            try:
                df['Gender'] = models['liver_gender_encoder'].transform(df['Gender'])
            except:
                # Fallback or error? Let's map manually if encoder fails strictly
                df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0}) # Assumption: similar to LabelEncoder usually
        
        # 2. Reindex
        expected_features = models['liver_features']
        df = df.reindex(columns=expected_features, fill_value=0)
        
        # 3. Scale
        X_scaled = models['liver_scaler'].transform(df)
        
        # Predict
        prediction = models['liver_model'].predict(X_scaled)[0]
        prob = models['liver_model'].predict_proba(X_scaled)[0][1]
        
        # Explanation
        explanation = get_explanation("Liver Disease", input_dict, prediction, prob)
        
        return {
            "prediction": int(prediction),
            "probability": float(prob),
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
