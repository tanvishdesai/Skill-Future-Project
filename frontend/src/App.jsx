import React, { useState } from 'react';
import axios from 'axios';
import DiseaseForm from './components/DiseaseForm';
import ResultCard from './components/ResultCard';

function App() {
  const [activeTab, setActiveTab] = useState('Heart'); // 'Heart', 'Liver', or 'ECG'
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async (data) => {
    setLoading(true);
    setError(null);
    try {
      let endpoint = 'http://localhost:8000/predict/heart';
      if (activeTab === 'Liver') {
        endpoint = 'http://localhost:8000/predict/liver';
      } else if (activeTab === 'ECG') {
        endpoint = 'http://localhost:8000/predict/ecg';
      }
      
      const response = await axios.post(endpoint, data);
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError("Failed to get prediction. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  const switchTab = (tab) => {
    setActiveTab(tab);
    handleReset();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 py-10 px-4">
      <div className="max-w-4xl mx-auto">
        <header className="text-center mb-10">
          <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600 mb-4">
            Dr.AI
          </h1>
          <p className="text-lg text-gray-600">
            Advanced Disease Risk Assessment using Machine Learning & Gemini AI
          </p>
        </header>

        {/* Tab Switcher */}
        <div className="flex justify-center mb-8">
          <div className="bg-white p-1 rounded-xl shadow-md inline-flex">
            {['Heart', 'Liver', 'ECG'].map((tab) => (
              <button
                key={tab}
                onClick={() => switchTab(tab)}
                className={`px-8 py-3 rounded-lg text-sm font-semibold transition-all duration-200 ${
                  activeTab === tab
                    ? 'bg-indigo-600 text-white shadow-lg'
                    : 'text-gray-500 hover:text-indigo-600'
                }`}
              >
                {tab === 'ECG' ? 'ECG Heartbeat' : `${tab} Disease`}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="relative">
          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-6">
              {error}
            </div>
          )}

          {loading ? (
            <div className="glass-card p-12 text-center">
              <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-600 mb-4"></div>
              <p className="text-xl text-gray-600 font-medium">Analyzing parameters...</p>
              <p className="text-sm text-gray-500 mt-2">Consulting Gemini AI for explanation...</p>
            </div>
          ) : result ? (
            <ResultCard result={result} onReset={handleReset} />
          ) : (
            <DiseaseForm diseaseType={activeTab} onSubmit={handlePredict} />
          )}
        </div>
        
        <footer className="text-center mt-12 text-gray-400 text-sm">
          &copy; 2024 MediPredict AI. Powered by Google Gemini.
        </footer>
      </div>
    </div>
  );
}

export default App;
