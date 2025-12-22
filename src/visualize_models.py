import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

os.makedirs('plots', exist_ok=True)

def visualize_heart_model():
    print("Visualizing Heart Model...")
    try:
        model = joblib.load('models/heart_model.pkl')
        scaler = joblib.load('models/heart_scaler.pkl')
        
        df = pd.read_csv('data/heart.csv')
        cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        existing_cat_cols = [c for c in cat_cols if c in df.columns]
        df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)
        
        X = df.drop('HeartDisease', axis=1)
        y = df['HeartDisease']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = scaler.transform(X_test)
        
        y_pred = model.predict(X_test_scaled)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Heart Disease Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('plots/heart_confusion_matrix.png')
        plt.close()
        
        # Feature Importance
        feature_names = X.columns
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Heart Disease Feature Importances")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
        plt.xlim([-1, X.shape[1]])
        plt.tight_layout()
        plt.savefig('plots/heart_feature_importance.png')
        plt.close()
        
        # SHAP
        # Use a sample for speed
        explainer = shap.TreeExplainer(model)
        # TreeExplainer expects raw features if model was trained on them, but we gathered scaled features.
        # Actually Random Forest doesn't strictly need scaling but we did it. 
        # SHAP with scaled data might be less interpretable for values, but fine for importance.
        shap_values = explainer.shap_values(X_test_scaled)
        
        plt.figure()
        # Summary plot (bar)
        shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, plot_type="bar", show=False)
        plt.savefig('plots/heart_shap_summary.png', bbox_inches='tight')
        plt.close()
        
        print("Heart plots saved.")
        
    except Exception as e:
        print(f"Error visualizing heart model: {e}")

def visualize_liver_model():
    print("Visualizing Liver Model...")
    try:
        model = joblib.load('models/liver_model.pkl')
        scaler = joblib.load('models/liver_scaler.pkl')
        
        df = pd.read_csv('data/indian_liver_patient.csv')
        df = df.fillna(df.median(numeric_only=True))
        
        if 'Gender' in df.columns:
            le = joblib.load('models/liver_gender_encoder.pkl')
            df['Gender'] = le.transform(df['Gender'])
            
        if 'Dataset' in df.columns:
            df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})
            target = 'Dataset'
        else:
            target = df.columns[-1]
            
        X = df.drop(target, axis=1)
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = scaler.transform(X_test)
        
        y_pred = model.predict(X_test_scaled)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
        plt.title('Liver Disease Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('plots/liver_confusion_matrix.png')
        plt.close()
        
        # Feature Importance
        feature_names = X.columns
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Liver Disease Feature Importances")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
        plt.xlim([-1, X.shape[1]])
        plt.tight_layout()
        plt.savefig('plots/liver_feature_importance.png')
        plt.close()

         # SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)
        
        plt.figure()
        shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, plot_type="bar", show=False)
        plt.savefig('plots/liver_shap_summary.png', bbox_inches='tight')
        plt.close()
        
        print("Liver plots saved.")

    except Exception as e:
        print(f"Error visualizing liver model: {e}")

if __name__ == "__main__":
    visualize_heart_model()
    visualize_liver_model()
