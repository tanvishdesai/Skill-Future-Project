import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=api_key)

def get_explanation(disease_type: str, input_data: dict, prediction: str, probability: float):
    try:
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
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
