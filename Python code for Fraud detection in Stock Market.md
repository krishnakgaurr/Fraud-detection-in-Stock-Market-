# ==========================================
# Fraud Detection in Stock Market
# Isolation Forest + LSTM Autoencoder
# ==========================================

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras import layers, models

import random
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIG
# ==========================================
TICKER = "AAPL"
START = "2020-01-01"
END = "2023-12-31"
WINDOWS = [5, 10, 20]
SEED = 42

np.random.seed(SEED)
random.seed(SEED)

# ==========================================
# FETCH DATA
# ==========================================
def fetch_data():
    df = yf.download(TICKER, start=START, end=END, progress=False)

    if df.empty:
        raise ValueError("No data fetched!")

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.dropna(inplace=True)

    return df

# ==========================================
# FEATURE ENGINEERING
# ==========================================
def feature_engineering(df):
    df = df.copy()

    # returns
    df['return'] = df['Close'].pct_change().fillna(0)
    df['log_return'] = np.log1p(df['return'].clip(-0.999999, None))

    for w in WINDOWS:
        df[f'roll_mean_{w}'] = df['return'].rolling(w).mean().fillna(0)
        df[f'roll_std_{w}'] = df['return'].rolling(w).std().fillna(0)

        vol_mean = df['Volume'].rolling(w).mean().replace(0, 1)
        vol_std = df['Volume'].rolling(w).std().replace(0, 1)

        df[f'vol_z_{w}'] = (df['Volume'] - vol_mean) / vol_std

    df.fillna(0, inplace=True)
    return df

# ==========================================
# SYNTHETIC FRAUD INJECTION
# ==========================================
def inject_anomalies(df, n_events=10, magnitude=5):
    df = df.copy()
    labels = pd.Series(0, index=df.index)

    indices = list(range(10, len(df)-10))
    chosen = random.sample(indices, n_events)

    for i in chosen:
        direction = 1 if random.random() > 0.5 else -1
        pct = magnitude * 0.01 * direction

        df.iloc[i, df.columns.get_loc('Close')] *= (1 + pct)
        df.iloc[i, df.columns.get_loc('Volume')] *= magnitude

        labels.iloc[i] = 1

    return df, labels

# ==========================================
# PREPARE DATA
# ==========================================
def prepare_data(df):
    feature_cols = [c for c in df.columns if c not in ['Open','High','Low','Close','Volume']]

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, feature_cols

# ==========================================
# ISOLATION FOREST
# ==========================================
def train_isolation_forest(X):
    model = IsolationForest(contamination=0.01, random_state=SEED)
    model.fit(X)
    scores = -model.decision_function(X)
    return model, scores

# ==========================================
# LSTM AUTOENCODER
# ==========================================
def build_lstm_autoencoder(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32, return_sequences=False),
        layers.RepeatVector(input_shape[0]),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(64, return_sequences=True),
        layers.TimeDistributed(layers.Dense(input_shape[1]))
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def create_sequences(X, seq_len=10):
    seqs = []
    for i in range(len(X) - seq_len):
        seqs.append(X[i:i+seq_len])
    return np.array(seqs)

# ==========================================
# EVALUATION
# ==========================================
def evaluate(true, scores):
    threshold = np.percentile(scores, 99)
    preds = (scores > threshold).astype(int)

    precision = precision_score(true, preds, zero_division=0)
    recall = recall_score(true, preds, zero_division=0)
    f1 = f1_score(true, preds, zero_division=0)

    return precision, recall, f1

# ==========================================
# PLOT
# ==========================================
def plot_results(df, labels, scores):
    plt.figure(figsize=(12,6))

    plt.plot(df.index, df['Close'], label='Close Price')

    anomalies = df[labels == 1]
    plt.scatter(anomalies.index, anomalies['Close'], color='red', label='Fraud')

    plt.legend()
    plt.title("Fraud Detection")
    plt.show()

# ==========================================
# MAIN
# ==========================================
def main():

    # 1. Data
    df = fetch_data()
    df = feature_engineering(df)

    # 2. Inject fraud
    df, labels = inject_anomalies(df)

    # 3. Split
    split = int(len(df) * 0.7)
    df_train, df_test = df[:split], df[split:]
    labels_train, labels_test = labels[:split], labels[split:]

    # 4. Prepare
    X_train, scaler, cols = prepare_data(df_train)
    X_test = scaler.transform(df_test[cols])

    # ======================================
    # Isolation Forest
    # ======================================
    iso_model, train_scores = train_isolation_forest(X_train)
    test_scores = -iso_model.decision_function(X_test)

    p, r, f = evaluate(labels_test.values, test_scores)

    print("\nIsolation Forest Results:")
    print(f"Precision: {p:.3f}, Recall: {r:.3f}, F1: {f:.3f}")

    # ======================================
    # LSTM Autoencoder
    # ======================================
    seq_len = 10
    X_train_seq = create_sequences(X_train, seq_len)
    X_test_seq = create_sequences(X_test, seq_len)

    model = build_lstm_autoencoder((seq_len, X_train.shape[1]))

    model.fit(
        X_train_seq, X_train_seq,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # Reconstruction error
    recon = model.predict(X_test_seq)
    mse = np.mean(np.square(recon - X_test_seq), axis=(1,2))

    # Align labels
    labels_seq = labels_test[seq_len:].values

    p, r, f = evaluate(labels_seq, mse)

    print("\nLSTM Autoencoder Results:")
    print(f"Precision: {p:.3f}, Recall: {r:.3f}, F1: {f:.3f}")

    # Plot
    plot_results(df_test, labels_test, test_scores)


if __name__ == "__main__":
    main()
