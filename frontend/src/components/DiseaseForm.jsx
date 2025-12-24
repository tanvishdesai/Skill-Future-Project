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

const ECGFields = () => (
  <>
    <div className="space-y-6">
      <div className="bg-blue-50 border-l-4 border-blue-400 p-4 mb-4">
        <p className="text-sm text-blue-700">
          <strong>ECG Signal Input:</strong> Select a sample ECG signal or provide your own 187-point signal data.
          The model analyzes single heartbeat waveforms to classify cardiac rhythm patterns.
        </p>
      </div>
      
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Select Sample ECG Signal</label>
        <select 
          name="sample_type" 
          className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          defaultValue="normal"
        >
          <option value="normal">Normal Heartbeat (N)</option>
          <option value="arrhythmia_ventricular">Ventricular Arrhythmia (V)</option>
        </select>
      </div>

      <div className="border-t pt-4">
        <p className="text-xs text-gray-500 mb-2">
          ℹ️ In production, this would integrate with ECG hardware or file upload for real patient data.
        </p>
      </div>
    </div>
  </>
);

const DiseaseForm = ({ diseaseType, onSubmit }) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    // Type conversion for Heart/Liver
    if (diseaseType !== 'ECG') {
      for (const key in data) {
          if (!isNaN(data[key]) && data[key] !== '') {
              data[key] = data[key].includes('.') ? parseFloat(data[key]) : parseInt(data[key]);
          }
      }
    }
    
    onSubmit(data);
  };

  const getFormTitle = () => {
    if (diseaseType === 'ECG') return 'ECG Heartbeat Analysis';
    return `${diseaseType} Disease Assessment`;
  };

  const renderFields = () => {
    switch(diseaseType) {
      case 'Heart':
        return <HeartFields />;
      case 'Liver':
        return <LiverFields />;
      case 'ECG':
        return <ECGFields />;
      default:
        return <HeartFields />;
    }
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
        {getFormTitle()}
      </h2>
      
      {renderFields()}
      
      <div className="mt-8">
        <button type="submit" className="btn-primary">
          {diseaseType === 'ECG' ? 'Analyze ECG' : 'Analyze Risk'}
        </button>
      </div>
    </motion.form>
  );
};

export default DiseaseForm;

