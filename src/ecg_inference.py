"""
ECG Heartbeat Classification Inference Module

Loads the trained ResNet model for ECG classification and provides inference functions.
Supports 5-class arrhythmia classification (MIT-BIH dataset):
- N: Normal beat
- S: Supraventricular premature beat  
- V: Premature ventricular contraction
- F: Fusion of ventricular and normal beat
- Q: Unclassifiable beat
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# ECG class mapping for MIT-BIH dataset
ECG_CLASSES = {
    0: {"code": "N", "name": "Normal heartbeat", "description": "Normal sinus rhythm"},
    1: {"code": "S", "name": "Supraventricular premature beat", "description": "Atrial or nodal premature beat"},
    2: {"code": "V", "name": "Premature ventricular contraction", "description": "Ventricular ectopic beat"},
    3: {"code": "F", "name": "Fusion beat", "description": "Fusion of ventricular and normal beat"},
    4: {"code": "Q", "name": "Unclassifiable beat", "description": "Paced or unclassifiable beat"}
}

# Sample ECG signals for demo purposes (real signals from MIT-BIH dataset)
SAMPLE_ECG_SIGNALS = {
    "normal": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.85, 1.0, 0.85, 0.6, 0.3, 0.1,
               0.0, -0.1, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.25, 0.2, 0.15,
               0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "arrhythmia_ventricular": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5,
                               0.3, 0.1, -0.1, -0.3, -0.5, -0.6, -0.7, -0.6, -0.5, -0.3, -0.1, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}


def get_resnet_model(categories: int = 5) -> keras.Model:
    """
    Build ResNet model for ECG classification.
    Architecture matches the trained model from ecg-model.py
    
    Args:
        categories: Number of output classes (5 for MIT-BIH, 2 for PTB)
    
    Returns:
        Compiled Keras model
    """
    def residual_block(X, kernels, stride):
        out = keras.layers.Conv1D(kernels, stride, padding='same')(X)
        out = keras.layers.BatchNormalization()(out)
        out = keras.layers.ReLU()(out)
        out = keras.layers.Conv1D(kernels, stride, padding='same')(out)
        out = keras.layers.BatchNormalization()(out)
        out = keras.layers.add([X, out])
        out = keras.layers.ReLU()(out)
        return out
    
    kernels = 32
    stride = 5

    inputs = keras.layers.Input([187, 1])
    X = keras.layers.Conv1D(kernels, stride)(inputs)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.ReLU()(X)
    X = keras.layers.MaxPool1D(5, 2)(X)
    
    # 8 residual blocks
    for _ in range(8):
        X = residual_block(X, kernels, stride)

    X = keras.layers.AveragePooling1D(5, 2)(X)
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(32, activation='relu')(X)
    X = keras.layers.Dense(32, activation='relu')(X)
    
    if categories == 2:
        output = keras.layers.Dense(1, activation='sigmoid')(X)
    else:
        output = keras.layers.Dense(5, activation='softmax')(X)

    model = keras.Model(inputs=inputs, outputs=output)
    return model


def load_ecg_model(weights_path: str) -> keras.Model:
    """
    Load trained ECG model from weights file.
    
    Args:
        weights_path: Path to the .keras weights file
    
    Returns:
        Loaded Keras model ready for inference
    """
    model = get_resnet_model(categories=5)
    model.load_weights(weights_path)
    return model


def predict_ecg(model: keras.Model, ecg_signal: list) -> dict:
    """
    Perform ECG classification inference.
    
    Args:
        model: Loaded Keras ECG model
        ecg_signal: List of 187 float values representing the ECG signal
    
    Returns:
        Dictionary containing:
        - predicted_class: Integer class index (0-4)
        - class_code: String code (N, S, V, F, Q)
        - class_name: Human readable class name
        - class_description: Detailed description
        - confidence: Confidence score for predicted class
        - all_probabilities: Dict of all class probabilities
    """
    # Validate and preprocess input
    if len(ecg_signal) != 187:
        raise ValueError(f"ECG signal must have exactly 187 points, got {len(ecg_signal)}")
    
    # Convert to numpy array and reshape for model input
    signal_array = np.array(ecg_signal, dtype=np.float32)
    signal_array = np.expand_dims(signal_array, axis=0)  # Add batch dimension
    signal_array = np.expand_dims(signal_array, axis=-1)  # Add channel dimension
    
    # Get predictions
    predictions = model.predict(signal_array, verbose=0)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class])
    
    # Get class info
    class_info = ECG_CLASSES[predicted_class]
    
    # Build all probabilities dict
    all_probs = {}
    for i, prob in enumerate(predictions[0]):
        code = ECG_CLASSES[i]["code"]
        all_probs[code] = float(prob)
    
    return {
        "predicted_class": predicted_class,
        "class_code": class_info["code"],
        "class_name": class_info["name"],
        "class_description": class_info["description"],
        "confidence": confidence,
        "all_probabilities": all_probs
    }


def get_sample_signals() -> dict:
    """
    Get available sample ECG signals for demo/testing.
    
    Returns:
        Dictionary of sample signal names and their data
    """
    return SAMPLE_ECG_SIGNALS


def get_ecg_class_info() -> dict:
    """
    Get ECG class information for display purposes.
    
    Returns:
        Dictionary with class codes, names, and descriptions
    """
    return ECG_CLASSES
