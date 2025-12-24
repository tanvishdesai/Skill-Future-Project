from src.ecg_inference import SAMPLE_ECG_SIGNALS

for name, signal in SAMPLE_ECG_SIGNALS.items():
    print(f"{name}: {len(signal)}")
