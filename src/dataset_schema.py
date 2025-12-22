from pydantic import BaseModel, Field
from typing import Optional

class HeartDiseaseInput(BaseModel):
    Age: int
    Sex: str  # 'Male' or 'Female' (will be encoded)
    ChestPainType: str # 'ATA', 'NAP', 'ASY', 'TA'
    RestingBP: int
    Cholesterol: int
    FastingBS: int # 0 or 1
    RestingECG: str # 'Normal', 'ST', 'LVH'
    MaxHR: int
    ExerciseAngina: str # 'Y' or 'N'
    Oldpeak: float
    ST_Slope: str # 'Up', 'Flat', 'Down'

class LiverDiseaseInput(BaseModel):
    Age: int
    Gender: str # 'Male' or 'Female'
    Total_Bilirubin: float
    Direct_Bilirubin: float
    Alkaline_Phosphotase: int
    Alamine_Aminotransferase: int
    Aspartate_Aminotransferase: int
    Total_Protiens: float
    Albumin: float
    Albumin_and_Globulin_Ratio: float
