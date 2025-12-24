from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import os
from contextlib import asynccontextmanager
from src.dataset_schema import HeartDiseaseInput, LiverDiseaseInput, ECGInput
from src.explanation import get_explanation
from src.ecg_inference import load_ecg_model, predict_ecg, get_sample_signals, ECG_CLASSES

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

    # Load ECG Model
    try:
        ecg_weights_path = 'models/ecg/mit_weights.keras'
        if os.path.exists(ecg_weights_path):
            models['ecg_model'] = load_ecg_model(ecg_weights_path)
            print("ECG model loaded.")
        else:
            print(f"ECG model not found at {ecg_weights_path}")
    except Exception as e:
        print(f"Error loading ECG model: {e}")
    
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

@app.post("/predict/ecg")
async def predict_ecg_endpoint(data: ECGInput):
    if 'ecg_model' not in models:
        raise HTTPException(status_code=503, detail="ECG model not loaded")
    
    try:
        # Get signal from input or use sample
        ecg_signal = data.ecg_signal
        
        if ecg_signal is None and data.sample_type:
            # Use sample signal
            samples = get_sample_signals()
            if data.sample_type in samples:
                ecg_signal = samples[data.sample_type]
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid sample_type. Available: {list(samples.keys())}"
                )
        
        if ecg_signal is None:
            raise HTTPException(
                status_code=400, 
                detail="Either ecg_signal or sample_type must be provided"
            )
        
        # Validate signal length
        if len(ecg_signal) != 187:
            raise HTTPException(
                status_code=400,
                detail=f"ECG signal must have exactly 187 points, got {len(ecg_signal)}"
            )
        
        # Run inference
        result = predict_ecg(models['ecg_model'], ecg_signal)
        
        # Prepare data for explanation
        explanation_data = {
            "class_code": result["class_code"],
            "class_name": result["class_name"],
            "all_probabilities": result["all_probabilities"]
        }
        
        # Get Gemini explanation
        explanation = get_explanation(
            "ECG Heartbeat Classification", 
            explanation_data, 
            result["predicted_class"], 
            result["confidence"]
        )
        
        return {
            "prediction": result["predicted_class"],
            "class_code": result["class_code"],
            "class_name": result["class_name"],
            "class_description": result["class_description"],
            "probability": result["confidence"],
            "all_probabilities": result["all_probabilities"],
            "explanation": explanation
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ecg/samples")
async def get_ecg_samples():
    """Get available sample ECG signals for testing"""
    samples = get_sample_signals()
    return {
        "available_samples": list(samples.keys()),
        "signal_length": 187
    }

@app.get("/ecg/classes")
async def get_ecg_classes():
    """Get ECG class information"""
    return ECG_CLASSES

