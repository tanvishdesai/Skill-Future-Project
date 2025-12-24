"""
ECG Heartbeat Classification using ResNet
Converts abnormal/normal heartbeats (PTB) and classifies 5 types of arrhythmias (MIT-BIH)

No external dependencies beyond standard ML libraries - tensorflow-addons removed!
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics
import itertools
import os
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==================== CONFIGURATION ====================
# UPDATE THESE PATHS TO YOUR DATA LOCATION
DATA_PATH = r"C:\Users\DELL\Desktop\code_playground\jhaad\project\ecg new module\ECG-Heartbeat-Classification-main\data\data"
MITBIH_TRAIN_PATH = f"{DATA_PATH}\\mitbih_train.csv"
MITBIH_TEST_PATH = f"{DATA_PATH}\\mitbih_test.csv"
PTBDB_ABNORMAL_PATH = f"{DATA_PATH}\\ptbdb_abnormal.csv"
PTBDB_NORMAL_PATH = f"{DATA_PATH}\\ptbdb_normal.csv"

# Output directory for plots and models
OUTPUT_DIR = "ecg_results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# Create output directories
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Training parameters
BATCH_SIZE = 128
EPOCHS_PTB = 10
EPOCHS_MIT = 10
EPOCHS_TRANSFER = 8

# Plot counter for unique filenames
plot_counter = {"count": 0}


# ==================== UTILITY CLASSES ====================
class CyclicalLearningRate(keras.optimizers.schedules.LearningRateSchedule):
    """Custom Cyclical Learning Rate implementation"""
    
    def __init__(self, initial_learning_rate, maximal_learning_rate, step_size, scale_fn):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.maximal_learning_rate = maximal_learning_rate
        self.step_size = step_size
        self.scale_fn = scale_fn
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        cycle = tf.floor(1 + step / (2 * self.step_size))
        x = tf.abs(step / self.step_size - 2 * cycle + 1)
        lr = self.initial_learning_rate + (self.maximal_learning_rate - self.initial_learning_rate) * tf.maximum(0.0, (1 - x)) * self.scale_fn(cycle)
        return lr
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "maximal_learning_rate": self.maximal_learning_rate,
            "step_size": self.step_size,
        }


class LRFinder(Callback):
    """Learning rate finder callback"""
    
    def __init__(self, start_lr=1e-7, end_lr=10, max_steps=100, smoothing=0.9):
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_begin(self, logs=None):
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.exp_annealing(self.step)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate.assign(self.lr), self.lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss:
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)
            self.lrs.append(self.lr)

            if self.step == 0 or loss < self.best_loss:
                self.best_loss = loss

            if smooth_loss > 4 * self.best_loss or tf.math.is_nan(smooth_loss):
                self.model.stop_training = True

        if self.step == self.max_steps:
            self.model.stop_training = True

        self.step += 1

    def exp_annealing(self, step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)

    def plot(self, filename="lr_finder"):
        plot_counter["count"] += 1
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_ylabel('Loss')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.plot(self.lrs, self.losses)
        filepath = os.path.join(PLOTS_DIR, f"{plot_counter['count']:02d}_{filename}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filepath}")


# ==================== UTILITY FUNCTIONS ====================
def pretty_plot(history, field, prefix=""):
    """Plot training history"""
    def plot(data, val_data, best_index, best_value, title, filename):
        plot_counter["count"] += 1
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(data)+1), data, label='train')
        plt.plot(range(1, len(data)+1), val_data, label='validation')
        if best_index is not None:
            plt.axvline(x=best_index+1, linestyle=':', c="#777777")
        if best_value is not None:
            plt.axhline(y=best_value, linestyle=':', c="#777777")
        plt.xlabel('Epoch')
        plt.ylabel(field)
        plt.xticks(range(0, len(data), 20))
        plt.title(title)
        plt.legend()
        filepath = os.path.join(PLOTS_DIR, f"{plot_counter['count']:02d}_{filename}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filepath}")

    data = history.history[field]
    val_data = history.history['val_' + field]
    tail = int(0.15 * len(data))
    best_index = np.argmin(val_data)
    best_value = val_data[best_index]

    plot(data, val_data, best_index, best_value, 
         f"{field} over epochs (best {best_value:06.4f})", 
         f"{prefix}_{field}_full")
    plot(data[-tail:], val_data[-tail:], None, best_value, 
         f"{field} over last {tail} epochs", 
         f"{prefix}_{field}_tail")


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', 
                         cmap=plt.cm.Blues, filename="confusion_matrix"):
    """Plot confusion matrix"""
    plot_counter["count"] += 1
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    filepath = os.path.join(PLOTS_DIR, f"{plot_counter['count']:02d}_{filename}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filepath}")


# ==================== MODEL ARCHITECTURE ====================
def get_resnet_model(categories=2):
    """Build ResNet model for ECG classification"""
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


def get_transfer_model(base_model, categories=2):
    """Create transfer learning model"""
    base_model.trainable = False
    model_input = base_model.inputs
    model_output = base_model.layers[-4].output

    X = keras.layers.Dense(64, activation='relu')(model_output)
    X = keras.layers.Dense(32, activation='relu')(X)
    
    if categories == 2:
        out = keras.layers.Dense(1, activation='sigmoid')(X)
    else:
        out = keras.layers.Dense(5, activation='softmax')(X)

    transfer_model = keras.Model(inputs=model_input, outputs=out)
    return transfer_model


# ==================== DATA LOADING ====================
def load_ptbdb_data():
    """Load and prepare PTB database"""
    print("Loading PTB database...")
    ptbdb_abnormal = pd.read_csv(PTBDB_ABNORMAL_PATH, header=None)
    ptbdb_normal = pd.read_csv(PTBDB_NORMAL_PATH, header=None)
    ptbdb = pd.concat([ptbdb_abnormal, ptbdb_normal], axis=0, ignore_index=True)
    
    ptbdb[187] = ptbdb[187].astype(int)
    distribution = ptbdb[187].value_counts()
    print(f"PTB Distribution:\n{distribution}")
    
    # Visualize
    plot_counter["count"] += 1
    ptbdb_labels = ptbdb.iloc[:, -1].replace({0: 'Normal', 1: 'Abnormal'})
    plt.figure(figsize=(8, 5))
    plt.hist(ptbdb_labels)
    plt.title("Distribution of the PTB Database")
    filepath = os.path.join(PLOTS_DIR, f"{plot_counter['count']:02d}_ptb_distribution.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filepath}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        ptbdb.iloc[:, :-1].values, ptbdb.iloc[:, -1].values, 
        test_size=0.2, random_state=18
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=18
    )
    
    # Expand dimensions for Conv1D
    X_train = np.expand_dims(X_train, -1)
    X_val = np.expand_dims(X_val, -1)
    X_test = np.expand_dims(X_test, -1)
    y_train = np.expand_dims(y_train, -1)
    y_val = np.expand_dims(y_val, -1)
    y_test = np.expand_dims(y_test, -1)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_mitbih_data():
    """Load and prepare MIT-BIH database"""
    print("Loading MIT-BIH database...")
    mitbih_train = pd.read_csv(MITBIH_TRAIN_PATH, header=None)
    mitbih_test = pd.read_csv(MITBIH_TEST_PATH, header=None)
    
    mitbih_train[187] = mitbih_train[187].astype(int)
    distribution = mitbih_train[187].value_counts()
    print(f"MIT-BIH Distribution:\n{distribution}")
    
    # Visualize
    plot_counter["count"] += 1
    mitbih_train_labels = mitbih_train.iloc[:, -1].replace({0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'})
    plt.figure(figsize=(8, 5))
    plt.hist(mitbih_train_labels)
    plt.title("Distribution of MIT-BIH dataset")
    filepath = os.path.join(PLOTS_DIR, f"{plot_counter['count']:02d}_mitbih_distribution.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filepath}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        mitbih_train.iloc[:, :-1].values, mitbih_train.iloc[:, -1].values,
        test_size=0.1, random_state=42
    )
    X_test = mitbih_test.iloc[:, :-1].values
    y_test = mitbih_test.iloc[:, -1].values
    
    # Expand dimensions
    X_train = np.expand_dims(X_train, -1)
    X_val = np.expand_dims(X_val, -1)
    X_test = np.expand_dims(X_test, -1)
    y_train = np.expand_dims(y_train, -1)
    y_val = np.expand_dims(y_val, -1)
    y_test = np.expand_dims(y_test, -1)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ==================== TRAINING FUNCTIONS ====================
def train_ptb_model(train_data, val_data):
    """Train ResNet on PTB database (binary classification)"""
    print("\n" + "="*50)
    print("Training PTB Model (Binary Classification)")
    print("="*50)
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Setup cyclical learning rate
    N = X_train.shape[0]
    iterations = N / BATCH_SIZE
    step_size = 2 * iterations
    
    lr_schedule = CyclicalLearningRate(
        initial_learning_rate=1e-6, 
        maximal_learning_rate=1e-3, 
        step_size=step_size, 
        scale_fn=lambda x: 1 / (2.0 ** (x - 1))
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Create and compile model
    model = get_resnet_model(categories=2)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    weights_path = os.path.join(MODELS_DIR, "ptbd_weights.keras")
    save_best_weights = ModelCheckpoint(
        filepath=weights_path, verbose=1, save_best_only=True
    )
    
    # Train
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        shuffle=True, 
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS_PTB, 
        callbacks=[save_best_weights]
    )
    
    # Plot results
    pretty_plot(history, 'loss', prefix='ptb')
    pretty_plot(history, 'accuracy', prefix='ptb')
    
    return model, history, weights_path


def train_mitbih_model(train_data, val_data):
    """Train ResNet on MIT-BIH database (5-class classification)"""
    print("\n" + "="*50)
    print("Training MIT-BIH Model (5-Class Classification)")
    print("="*50)
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Setup cyclical learning rate
    N = X_train.shape[0]
    iterations = N / BATCH_SIZE
    step_size = 2 * iterations
    
    lr_schedule = CyclicalLearningRate(
        initial_learning_rate=1e-5, 
        maximal_learning_rate=1e-3, 
        step_size=step_size,
        scale_fn=lambda x: 1 / (2.0 ** (x - 1))
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Create and compile model
    model = get_resnet_model(categories=5)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    weights_path = os.path.join(MODELS_DIR, "mit_weights.keras")
    save_weights = ModelCheckpoint(
        filepath=weights_path, verbose=1, save_best_only=True
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        shuffle=True,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_MIT,
        callbacks=[save_weights]
    )
    
    # Plot results
    pretty_plot(history, 'loss', prefix='mitbih')
    pretty_plot(history, 'accuracy', prefix='mitbih')
    
    return model, history, weights_path


def evaluate_model(model, test_data, classes, model_name, weights_path):
    """Evaluate model and show confusion matrix"""
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name}")
    print('='*50)
    
    X_test, y_test = test_data
    
    # Load best weights
    model.load_weights(weights_path)
    model.evaluate(X_test, y_test)
    
    # Predictions
    if len(classes) == 2:
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
    else:
        y_pred = tf.argmax(model.predict(X_test), axis=-1)
    
    # Confusion matrix
    cnf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    
    # Create safe filename
    safe_name = model_name.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    plot_confusion_matrix(cnf_matrix, classes=classes, 
                         title=f'{model_name} - Confusion matrix',
                         filename=f'{safe_name}_confusion')


def train_transfer_learning(base_model, train_data, val_data, categories=2, model_prefix="transfer"):
    """Train transfer learning model"""
    print("\n" + "="*50)
    print(f"Transfer Learning ({'Binary' if categories == 2 else '5-Class'})")
    print("="*50)
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    transfer_model = get_transfer_model(base_model, categories)
    
    weights_file = os.path.join(MODELS_DIR, f"t_weights_{'bin' if categories == 2 else 'multi'}_{model_prefix}.keras")
    save_best_weights = ModelCheckpoint(
        filepath=weights_file, verbose=1, save_best_only=True
    )
    
    loss = 'binary_crossentropy' if categories == 2 else 'sparse_categorical_crossentropy'
    transfer_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    history = transfer_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        shuffle=True,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_TRANSFER,
        callbacks=[save_best_weights]
    )
    
    pretty_plot(history, 'loss', prefix=f'transfer_{model_prefix}')
    pretty_plot(history, 'accuracy', prefix=f'transfer_{model_prefix}')
    
    return transfer_model, history, weights_file


# ==================== MAIN EXECUTION ====================
def main():
    """Main execution function"""
    print("="*70)
    print("ECG HEARTBEAT CLASSIFICATION USING RESNET")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Plots will be saved to: {PLOTS_DIR}")
    print(f"Models will be saved to: {MODELS_DIR}\n")
    
    # Load data
    ptb_train, ptb_val, ptb_test = load_ptbdb_data()
    mit_train, mit_val, mit_test = load_mitbih_data()
    
    # Train PTB model (binary classification)
    ptb_model, _, ptb_weights = train_ptb_model(ptb_train, ptb_val)
    evaluate_model(ptb_model, ptb_test, ['Normal', 'Abnormal'], "PTB Model", ptb_weights)
    
    # Train MIT-BIH model (5-class classification)
    mit_model, _, mit_weights = train_mitbih_model(mit_train, mit_val)
    evaluate_model(mit_model, mit_test, ['N', 'S', 'V', 'F', 'Q'], "MIT-BIH Model", mit_weights)
    
    # Transfer learning: MIT-BIH -> PTB
    print("\n" + "="*70)
    print("TRANSFER LEARNING EXPERIMENTS")
    print("="*70)
    
    transfer_model1, _, t1_weights = train_transfer_learning(
        mit_model, ptb_train, ptb_val, categories=2, model_prefix="mit_to_ptb"
    )
    evaluate_model(transfer_model1, ptb_test, ['Normal', 'Abnormal'], 
                  "Transfer Model (MIT-BIH -> PTB)", t1_weights)
    
    # Transfer learning: PTB -> MIT-BIH
    transfer_model2, _, t2_weights = train_transfer_learning(
        ptb_model, mit_train, mit_val, categories=5, model_prefix="ptb_to_mit"
    )
    evaluate_model(transfer_model2, mit_test, ['N', 'S', 'V', 'F', 'Q'],
                  "Transfer Model (PTB -> MIT-BIH)", t2_weights)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nAll plots saved to: {PLOTS_DIR}")
    print(f"All models saved to: {MODELS_DIR}")
    
    # Create summary
    summary_file = os.path.join(OUTPUT_DIR, "training_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("ECG Classification Training Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Models trained:\n")
        f.write(f"  1. PTB Binary Classifier: {ptb_weights}\n")
        f.write(f"  2. MIT-BIH 5-Class Classifier: {mit_weights}\n")
        f.write(f"  3. Transfer MIT->PTB: {t1_weights}\n")
        f.write(f"  4. Transfer PTB->MIT: {t2_weights}\n\n")
        f.write(f"Total plots generated: {plot_counter['count']}\n")
        f.write(f"Plots directory: {PLOTS_DIR}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print("\n✅ All done! Check the ecg_results folder for outputs.")


if __name__ == "__main__":
    main()