import React from 'react';
import { motion } from 'framer-motion';

const ResultCard = ({ result, onReset }) => {
  const isHighRisk = result.prediction === 1;
  const colorClass = isHighRisk ? 'text-red-600' : 'text-green-600';
  const bgClass = isHighRisk ? 'bg-red-50' : 'bg-green-50';
  const borderClass = isHighRisk ? 'border-red-200' : 'border-green-200';

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`glass-card p-8 border-2 ${borderClass} ${bgClass}`}
    >
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-800 mb-2">Analysis Result</h2>
        <div className={`text-4xl font-extrabold ${colorClass} mb-2`}>
          {isHighRisk ? 'HIGH RISK' : 'LOW RISK'}
        </div>
        <div className="text-gray-500 font-medium">
          Confidence: {(result.probability * 100).toFixed(1)}%
        </div>
      </div>

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
