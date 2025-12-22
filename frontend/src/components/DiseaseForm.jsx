import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const HeartFields = () => (
  <>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div>
        <label>Age</label>
        <input type="number" name="Age" required min="1" max="120" defaultValue="50"/>
      </div>
      <div>
        <label>Sex</label>
        <select name="Sex" required defaultValue="Male">
          <option value="Male">Male</option>
          <option value="Female">Female</option>
        </select>
      </div>
      <div>
        <label>Chest Pain Type</label>
        <select name="ChestPainType" required defaultValue="ASY">
          <option value="TA">Typical Angina</option>
          <option value="ATA">Atypical Angina</option>
          <option value="NAP">Non-Anginal Pain</option>
          <option value="ASY">Asymptomatic</option>
        </select>
      </div>
      <div>
        <label>Resting BP (mm Hg)</label>
        <input type="number" name="RestingBP" required defaultValue="120"/>
      </div>
      <div>
        <label>Cholesterol (mm/dl)</label>
        <input type="number" name="Cholesterol" required defaultValue="200"/>
      </div>
      <div>
        <label>Fasting Blood Sugar  120 mg/dl?</label>
        <select name="FastingBS" required defaultValue="0">
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>
      <div>
        <label>Resting ECG</label>
        <select name="RestingECG" required defaultValue="Normal">
          <option value="Normal">Normal</option>
          <option value="ST">ST-T Wave Abnormality</option>
          <option value="LVH">Left Ventricular Hypertrophy</option>
        </select>
      </div>
      <div>
        <label>Max Heart Rate</label>
        <input type="number" name="MaxHR" required defaultValue="140"/>
      </div>
      <div>
        <label>Exercise Angina?</label>
        <select name="ExerciseAngina" required defaultValue="N">
          <option value="N">No</option>
          <option value="Y">Yes</option>
        </select>
      </div>
      <div>
        <label>Oldpeak (ST depression)</label>
        <input type="number" step="0.1" name="Oldpeak" required defaultValue="0.0"/>
      </div>
      <div>
        <label>ST Slope</label>
        <select name="ST_Slope" required defaultValue="Flat">
          <option value="Up">Upsloping</option>
          <option value="Flat">Flat</option>
          <option value="Down">Downsloping</option>
        </select>
      </div>
    </div>
  </>
);

const LiverFields = () => (
  <>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div>
        <label>Age</label>
        <input type="number" name="Age" required min="1" max="120" defaultValue="45"/>
      </div>
      <div>
        <label>Gender</label>
        <select name="Gender" required defaultValue="Male">
          <option value="Male">Male</option>
          <option value="Female">Female</option>
        </select>
      </div>
      <div>
        <label>Total Bilirubin</label>
        <input type="number" step="0.1" name="Total_Bilirubin" required defaultValue="0.9"/>
      </div>
      <div>
        <label>Direct Bilirubin</label>
        <input type="number" step="0.1" name="Direct_Bilirubin" required defaultValue="0.2"/>
      </div>
      <div>
        <label>Alkaline Phosphotase</label>
        <input type="number" name="Alkaline_Phosphotase" required defaultValue="200"/>
      </div>
      <div>
        <label>Alamine Aminotransferase</label>
        <input type="number" name="Alamine_Aminotransferase" required defaultValue="30"/>
      </div>
      <div>
        <label>Aspartate Aminotransferase</label>
        <input type="number" name="Aspartate_Aminotransferase" required defaultValue="30"/>
      </div>
      <div>
        <label>Total Proteins</label>
        <input type="number" step="0.1" name="Total_Protiens" required defaultValue="6.8"/>
      </div>
      <div>
        <label>Albumin</label>
        <input type="number" step="0.1" name="Albumin" required defaultValue="3.5"/>
      </div>
      <div>
        <label>Albumin/Globulin Ratio</label>
        <input type="number" step="0.1" name="Albumin_and_Globulin_Ratio" required defaultValue="1.0"/>
      </div>
    </div>
  </>
);

const DiseaseForm = ({ diseaseType, onSubmit }) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    // Type conversion
    for (const key in data) {
        if (!isNaN(data[key]) && data[key] !== '') {
            data[key] = data[key].includes('.') ? parseFloat(data[key]) : parseInt(data[key]);
        }
    }
    
    onSubmit(data);
  };

  return (
    <motion.form 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      onSubmit={handleSubmit} 
      className="glass-card p-8"
    >
      <h2 className="text-2xl font-bold mb-6 text-gray-800 border-b pb-2">
        {diseaseType} Disease Assessment
      </h2>
      
      {diseaseType === 'Heart' ? <HeartFields /> : <LiverFields />}
      
      <div className="mt-8">
        <button type="submit" className="btn-primary">
          Analyze Risk
        </button>
      </div>
    </motion.form>
  );
};

export default DiseaseForm;
