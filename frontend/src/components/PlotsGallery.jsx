import React, { useState } from 'react';

const PlotsGallery = () => {
    const [activeSection, setActiveSection] = useState('Heart');

    const plots = {
        Heart: [
            { title: 'Confusion Matrix', src: '/plots/heart_confusion_matrix.png' },
            { title: 'Feature Importance', src: '/plots/heart_feature_importance.png' },
            { title: 'SHAP Summary', src: '/plots/heart_shap_summary.png' },
        ],
        Liver: [
            { title: 'Confusion Matrix', src: '/plots/liver_confusion_matrix.png' },
            { title: 'Feature Importance', src: '/plots/liver_feature_importance.png' },
            { title: 'SHAP Summary', src: '/plots/liver_shap_summary.png' },
        ],
        ECG: [
            { title: 'PTB Distribution', src: '/plots/ecg plots/01_ptb_distribution.png' },
            { title: 'MIT-BIH Distribution', src: '/plots/ecg plots/02_mitbih_distribution.png' },
            { title: 'PTB Loss', src: '/plots/ecg plots/03_ptb_loss_full.png' },
            { title: 'PTB Accuracy', src: '/plots/ecg plots/05_ptb_accuracy_full.png' },
            { title: 'PTB Confusion Matrix', src: '/plots/ecg plots/07_ptb_model_confusion.png' },
            { title: 'MIT-BIH Loss', src: '/plots/ecg plots/08_mitbih_loss_full.png' },
            { title: 'MIT-BIH Accuracy', src: '/plots/ecg plots/10_mitbih_accuracy_full.png' },
            { title: 'MIT-BIH Confusion Matrix', src: '/plots/ecg plots/12_mit_bih_model_confusion.png' },
            { title: 'Transfer MIT to PTB Loss', src: '/plots/ecg plots/13_transfer_mit_to_ptb_loss_full.png' },
            { title: 'Transfer MIT to PTB Accuracy', src: '/plots/ecg plots/15_transfer_mit_to_ptb_accuracy_full.png' },
        ],
    };

    return (
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-8 animate-fade-in-up">
            <h2 className="text-3xl font-bold text-gray-800 mb-6 border-b pb-4">Model Performance Analysis</h2>
            
            <div className="flex flex-wrap gap-4 mb-8 justify-center">
                {Object.keys(plots).map((section) => (
                    <button
                        key={section}
                        onClick={() => setActiveSection(section)}
                        className={`px-6 py-2 rounded-full font-semibold transition-all duration-300 ${
                            activeSection === section
                                ? 'bg-indigo-600 text-white shadow-lg transform scale-105'
                                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                        }`}
                    >
                        {section} Models
                    </button>
                ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {plots[activeSection].map((plot, index) => (
                    <div key={index} className="group relative bg-gray-50 rounded-xl overflow-hidden shadow-sm hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
                        <div className="aspect-w-16 aspect-h-12 overflow-hidden bg-gray-200">
                            <img
                                src={plot.src}
                                alt={plot.title}
                                className="object-cover w-full h-full transform group-hover:scale-110 transition-transform duration-500"
                                loading="lazy"
                            />
                        </div>
                        <div className="p-4 bg-white">
                            <h3 className="text-lg font-bold text-gray-800 group-hover:text-indigo-600 transition-colors">
                                {plot.title}
                            </h3>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default PlotsGallery;
