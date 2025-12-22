import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def train_heart_model():
    print("Training Heart Disease Model...")
    try:
        df = pd.read_csv('data/heart.csv')
        
        # Basic preprocessing
        # Handle categorical variables: Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope
        # We will use get_dummies for simplicity in this baseline, or LabelEncoder if we want to be more specific.
        # For a robust pipeline, OneHotEncoding in a ColumnTransformer is better, but for this task, let's keep it simple.
        
        # Identify categorical columns
        cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        # Check if these columns exist, if not, try to guess or skip
        existing_cat_cols = [c for c in cat_cols if c in df.columns]
        
        df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)
        
        X = df.drop('HeartDisease', axis=1)
        y = df['HeartDisease']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        print("Heart Model Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        
        # Save artifacts
        joblib.dump(model, 'models/heart_model.pkl')
        joblib.dump(scaler, 'models/heart_scaler.pkl')
        joblib.dump(X.columns.tolist(), 'models/heart_features.pkl') # Save feature names for inference
        print("Heart model saved.")
        
    except Exception as e:
        print(f"Error training heart model: {e}")

def train_liver_model():
    print("\nTraining Liver Disease Model...")
    try:
        df = pd.read_csv('data/indian_liver_patient.csv')
        
        # Rename dataset columns to be standard if needed, or stick to what's there.
        # This dataset often has a 'Dataset' column as target (1 or 2).
        # And 'Gender' which needs encoding.
        # Also 'Albumin_and_Globulin_Ratio' often has nulls.
        
        # Fill missing values
        df = df.fillna(df.median(numeric_only=True)) # Simple imputation
        
        # Encode Gender
        if 'Gender' in df.columns:
            le = LabelEncoder()
            df['Gender'] = le.fit_transform(df['Gender'])
            joblib.dump(le, 'models/liver_gender_encoder.pkl')
        
        # Target column is usually 'Dataset' where 1=Disease, 2=No Disease (typically). 
        # Let's check or assume standard ILPD format. 
        # Standard: 1=Liver Patient, 2=Non Liver Patient. Let's convert to 0 and 1.
        # 1 -> 1 (Disease), 2 -> 0 (No Disease)
        if 'Dataset' in df.columns:
            df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})
            target = 'Dataset'
        else:
            # Fallback if column name is different
            target = df.columns[-1]
            
        X = df.drop(target, axis=1)
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        print("Liver Model Accuracy:", accuracy_score(y_test, y_pred))
        
        # Save artifacts
        joblib.dump(model, 'models/liver_model.pkl')
        joblib.dump(scaler, 'models/liver_scaler.pkl')
        joblib.dump(X.columns.tolist(), 'models/liver_features.pkl')
        print("Liver model saved.")

    except Exception as e:
        print(f"Error training liver model: {e}")

if __name__ == "__main__":
    train_heart_model()
    train_liver_model()
