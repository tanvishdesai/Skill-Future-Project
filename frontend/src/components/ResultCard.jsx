import React from 'react';
import { motion } from 'framer-motion';

const ResultCard = ({ result, onReset }) => {
  // Handle ECG results differently
  const isECG = result.class_code !== undefined;
  
  let isHighRisk, colorClass, bgClass, borderClass, riskLabel;
  
  if (isECG) {
    // ECG: any non-normal class is considered abnormal
    isHighRisk = result.class_code !== 'N';
    colorClass = isHighRisk ? 'text-orange-600' : 'text-green-600';
    bgClass = isHighRisk ? 'bg-orange-50' : 'bg-green-50';
    borderClass = isHighRisk ? 'border-orange-200' : 'border-green-200';
    riskLabel = result.class_name || (isHighRisk ? 'ABNORMAL RHYTHM' : 'NORMAL RHYTHM');
  } else {
    // Heart/Liver disease
    isHighRisk = result.prediction === 1;
    colorClass = isHighRisk ? 'text-red-600' : 'text-green-600';
    bgClass = isHighRisk ? 'bg-red-50' : 'bg-green-50';
    borderClass = isHighRisk ? 'border-red-200' : 'border-green-200';
    riskLabel = isHighRisk ? 'HIGH RISK' : 'LOW RISK';
  }

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`glass-card p-8 border-2 ${borderClass} ${bgClass}`}
    >
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-800 mb-2">
          {isECG ? 'ECG Analysis Result' : 'Analysis Result'}
        </h2>
        <div className={`text-4xl font-extrabold ${colorClass} mb-2`}>
          {riskLabel}
        </div>
        <div className="text-gray-500 font-medium">
          Confidence: {(result.probability * 100).toFixed(1)}%
        </div>
        
        {/* ECG-specific: Show class description */}
        {isECG && result.class_description && (
          <div className="mt-2 text-sm text-gray-600 italic">
            {result.class_description}
          </div>
        )}
      </div>

      {/* ECG-specific: Show all class probabilities */}
      {isECG && result.all_probabilities && (
        <div className="bg-white p-4 rounded-xl shadow-inner mb-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Classification Breakdown</h3>
          <div className="grid grid-cols-5 gap-2 text-center text-xs">
            {Object.entries(result.all_probabilities).map(([code, prob]) => (
              <div 
                key={code}
                className={`p-2 rounded ${code === result.class_code ? 'bg-indigo-100 font-bold' : 'bg-gray-50'}`}
              >
                <div className="font-medium">{code}</div>
                <div className="text-gray-500">{(prob * 100).toFixed(1)}%</div>
              </div>
            ))}
          </div>
          <div className="mt-2 text-xs text-gray-400">
            N=Normal, S=Supraventricular, V=Ventricular, F=Fusion, Q=Unclassifiable
          </div>
        </div>
      )}

      <div className="bg-white p-6 rounded-xl shadow-inner mb-8">
        <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
          <span className="mr-2">🤖</span> AI Explanation
        </h3>
        <p className="text-gray-700 leading-relaxed whitespace-pre-line">
          {result.explanation}
        </p>
      </div>

      <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-8">
        <div className="flex">
          <div className="ml-3">
            <p className="text-sm text-yellow-700">
              <strong>Disclaimer:</strong> This is an AI-generated estimate for educational purposes only. 
              It is NOT a medical diagnosis. Please consult a qualified healthcare professional for advice.
            </p>
          </div>
        </div>
      </div>

      <button onClick={onReset} className="w-full py-3 px-6 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition">
        Start New Assessment
      </button>
    </motion.div>
  );
};

export default ResultCard;

