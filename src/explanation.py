import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=api_key)

# ECG class information for explanations
ECG_CLASS_CONTEXT = {
    "N": "Normal heartbeat - indicates healthy cardiac rhythm",
    "S": "Supraventricular premature beat - an early heartbeat originating above the ventricles",
    "V": "Premature ventricular contraction (PVC) - an early heartbeat originating in the ventricles",
    "F": "Fusion beat - combination of normal and ventricular beats occurring together",
    "Q": "Unclassifiable beat - irregular rhythm that doesn't fit standard categories"
}

def get_explanation(disease_type: str, input_data: dict, prediction, probability: float):
    try:
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        # Handle ECG-specific formatting
        if disease_type == "ECG Heartbeat Classification":
            class_code = input_data.get("class_code", "N")
            class_name = input_data.get("class_name", "Unknown")
            class_description = ECG_CLASS_CONTEXT.get(class_code, "Unknown classification")
            all_probs = input_data.get("all_probabilities", {})
            
            is_abnormal = class_code != "N"
            risk_level = "Abnormal Rhythm Detected" if is_abnormal else "Normal Rhythm"
            
            prompt = f"""
            Act as a helpful medical assistant explanation tool for ECG analysis.
            
            Context:
            The user's ECG heartbeat has been analyzed using a deep learning model.
            Classification Result: **{class_name}** ({class_code})
            Medical Context: {class_description}
            Confidence: **{probability:.1%}**
            
            Class Probabilities:
            - Normal (N): {all_probs.get('N', 0):.1%}
            - Supraventricular (S): {all_probs.get('S', 0):.1%}
            - Ventricular (V): {all_probs.get('V', 0):.1%}
            - Fusion (F): {all_probs.get('F', 0):.1%}
            - Unclassifiable (Q): {all_probs.get('Q', 0):.1%}
            
            Task:
            Explain this ECG classification result to the patient in simple, easy-to-understand language.
            - If normal, reassure while mentioning regular checkups are still important.
            - If abnormal, explain what the detected rhythm type means without causing alarm.
            - Mention the confidence level and what it indicates.
            Keep the explanation concise (3-4 sentences) + 1 disclaimer sentence.
            Always add a disclaimer that this is an AI estimation from a single heartbeat sample and they should consult a cardiologist for proper diagnosis.
            """
        else:
            # Original logic for Heart/Liver diseases
            risk_level = "High Risk" if prediction == 1 or prediction == "1" else "Low Risk"
            
            prompt = f"""
            Act as a helpful medical assistant explanation tool.
            
            Context:
            The user has been assessed for **{disease_type}**.
            Prediction: **{risk_level}**
            Probability: **{probability:.2f}**
            
            Patient Data Key Factors:
            {input_data}
            
            Task:
            Explain this result to the patient in simple, easy-to-understand language. 
            Highlight which specific values (based on standard medical norms) might have contributed to this risk level. 
            Do not give medical advice or diagnosis. Always add a disclaimer that this is an AI estimation and they should consult a doctor.
            Keep the explanation concise (max 3-4 sentences) + 1 disclaimer sentence.
            """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate explanation due to an error: {str(e)}"

