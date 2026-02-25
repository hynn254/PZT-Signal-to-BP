import numpy as np
import pandas as pd
import glob
import os
import random
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import HeNormal


# ============================================================
# setting
# ============================================================
FS = 400
SEGMENT_SECONDS = 10
STRIDE_SECONDS = 5

POINTS_PER_SEGMENT = FS * SEGMENT_SECONDS
POINTS_PER_STRIDE  = FS * STRIDE_SECONDS

SIGNAL_FOLDER = "/home/park/Downloads/PZT_signal_to_BP_code/data/processed/ref/Mn8/20260219/"
BP_FILE = "/home/park/Downloads/PZT_signal_to_BP_code/data/raw/GT/20260219.csv"


# ============================================================
# File grouping
# ============================================================
def group_files(folder):
    files = glob.glob(os.path.join(folder, "*.csv"))
    groups = {}
    for f in files:
        subject = os.path.basename(f).split("_")[0]
        groups.setdefault(subject, []).append(f)

    for s in groups:
        groups[s] = sorted(groups[s])

    return groups


# ============================================================
# Created in 65 seconds
# ============================================================
def load_full_signal(file_list):
    sigs = []
    for f in file_list:
        df = pd.read_csv(f)
        sigs.append(df.iloc[:, 0].values)
    return np.concatenate(sigs)


# ============================================================
# segmentation
# ============================================================
def segment_signal(signal):
    segments = []
    start = 0
    while start + POINTS_PER_SEGMENT <= len(signal):
        segments.append(signal[start:start+POINTS_PER_SEGMENT])
        start += POINTS_PER_STRIDE
    return np.array(segments)


# ============================================================
# normalization
# ============================================================
def normalize_segments(X):
    X_norm = []
    for seg in X:
        min_val = np.min(seg)
        max_val = np.max(seg)
        seg_norm = (seg - min_val) / (max_val - min_val)
        X_norm.append(seg_norm)
    return np.array(X_norm)[..., np.newaxis]


# ============================================================
# CNN (Figure 5c structure)
# ============================================================
def build_model(input_length):
    he = HeNormal()

    inputs = Input(shape=(input_length, 1))

    x = Conv1D(64, 7, padding='same', activation='relu', kernel_initializer=he)(inputs)
    x = MaxPooling1D(5, strides=5)(x)

    x = Conv1D(128, 5, padding='same', activation='relu', kernel_initializer=he)(x)
    x = MaxPooling1D(5, strides=5)(x)

    x = Conv1D(256, 3, padding='same', activation='relu', kernel_initializer=he)(x)
    x = MaxPooling1D(5, strides=5)(x)

    x = Flatten()(x)

    x = Dense(512, activation='relu', kernel_initializer=he)(x)
    x = Dense(256, activation='relu', kernel_initializer=he)(x)
    x = Dense(128, activation='relu', kernel_initializer=he)(x)

    outputs = Dense(2)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.0005), loss='mae', metrics=['mae'])

    return model

# ============================================================
# plot structure
# ============================================================
def save_figure_5d(y_true, y_pred, save_path="figure_5d.png"):

    sbp_true = y_true[:, 0]
    dbp_true = y_true[:, 1]

    sbp_pred = y_pred[:, 0]
    dbp_pred = y_pred[:, 1]

    sbp_mae = np.mean(np.abs(sbp_true - sbp_pred))
    dbp_mae = np.mean(np.abs(dbp_true - dbp_pred))

    plt.figure(figsize=(10, 4))

    # ---------------- SBP ----------------
    plt.subplot(1, 2, 1)

    plt.scatter(sbp_true, sbp_pred, s=50)
    min_sbp = min(sbp_true.min(), sbp_pred.min())
    max_sbp = max(sbp_true.max(), sbp_pred.max())

    plt.plot([min_sbp, max_sbp],
             [min_sbp, max_sbp],
             'r--', linewidth=1.5)

    plt.xlabel("True SBP (mmHg)")
    plt.ylabel("Predicted SBP (mmHg)")
    plt.title("SBP")

    plt.text(min_sbp, max_sbp,
             f"MAE = {sbp_mae:.2f} mmHg",
             verticalalignment='top')

    # ---------------- DBP ----------------
    plt.subplot(1, 2, 2)

    plt.scatter(dbp_true, dbp_pred, s=50)
    min_dbp = min(dbp_true.min(), dbp_pred.min())
    max_dbp = max(dbp_true.max(), dbp_pred.max())

    plt.plot([min_dbp, max_dbp],
             [min_dbp, max_dbp],
             'r--', linewidth=1.5)

    plt.xlabel("True DBP (mmHg)")
    plt.ylabel("Predicted DBP (mmHg)")
    plt.title("DBP")

    plt.text(min_dbp, max_dbp,
             f"MAE = {dbp_mae:.2f} mmHg",
             verticalalignment='top')

    plt.tight_layout()

    # 🔥 저장 (해상도 300dpi → 논문용)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figure saved to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    # ============================================================
    # data load
    # ============================================================
    bp_df = pd.read_csv(BP_FILE)
    groups = group_files(SIGNAL_FOLDER)

    subjects = list(groups.keys())
    train_subject = subjects[0]
    test_subject  = subjects[1]

    print("Train set:", train_subject)
    print("Test set:", test_subject)

    # ---------------- Train ----------------
    train_signal = load_full_signal(groups[train_subject])
    train_segments = segment_signal(train_signal)

    sbp_train = bp_df[bp_df["name"] == train_subject]["SBP"].values[0]
    dbp_train = bp_df[bp_df["name"] == train_subject]["DBP"].values[0]

    y_train = np.tile([sbp_train, dbp_train], (len(train_segments), 1))

    # Flip augmentation (train only)
    train_segments_flip = -train_segments
    train_segments = np.concatenate([train_segments, train_segments_flip])
    y_train = np.concatenate([y_train, y_train])

    # ---------------- Test ----------------
    test_signal = load_full_signal(groups[test_subject])
    test_segments = segment_signal(test_signal)

    sbp_test = bp_df[bp_df["name"] == test_subject]["SBP"].values[0]
    dbp_test = bp_df[bp_df["name"] == test_subject]["DBP"].values[0]

    # Select only 1 segment like a thesis
    idx = random.randint(0, len(test_segments)-1)
    X_test = test_segments[idx:idx+1]
    y_test = np.array([[sbp_test, dbp_test]])

    # ---------------- Normalization ----------------
    X_train = normalize_segments(train_segments)
    X_test  = normalize_segments(X_test)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)


    # ============================================================
    # Train
    # ============================================================
    model = build_model(POINTS_PER_SEGMENT)

    early_stop = EarlyStopping(monitor='val_mae',
                            patience=300,
                            restore_best_weights=True)

    model.fit(X_train, y_train,
            validation_split=0.2,
            epochs=2000,
            batch_size=256,
            callbacks=[early_stop],
            verbose=1)

    # ============================================================
    # Prediction
    # ============================================================
    pred = model.predict(X_test)

    print("\nTrue:", y_test[0])
    print("Pred:", pred[0])

    print("SBP MAE:", abs(y_test[0][0] - pred[0][0]))
    print("DBP MAE:", abs(y_test[0][1] - pred[0][1]))

    save_figure_5d(y_test, pred, save_path="Figure_5d_result.png")

