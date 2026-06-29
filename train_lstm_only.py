"""
Standalone script to train and save ONLY the Bidirectional LSTM model.
All other models and preprocessing artifacts are already saved.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import os
import re
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

DATASET_PATH = r"d:\download\nlpProject\Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
ARTIFACTS_DIR = r"d:\download\nlpProject\models"

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\{\{.*?\}\}', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    print("="*60)
    print("TRAINING BIDIRECTIONAL LSTM MODEL ONLY")
    print("="*60)

    df = pd.read_csv(DATASET_PATH)
    data = df[['instruction', 'intent', 'response', 'category']].copy()
    data.dropna(subset=['instruction', 'intent', 'response'], inplace=True)
    data['clean_instruction'] = data['instruction'].apply(clean_text)
    data = data[data['clean_instruction'].str.len() > 0]

    X = data['clean_instruction']
    y = data['intent']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    with open(os.path.join(ARTIFACTS_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "dl_config.pkl"), "rb") as f:
        dl_config = pickle.load(f)

    MAX_LEN = dl_config["MAX_LEN"]
    VOCAB_SIZE = dl_config["VOCAB_SIZE"]
    NUM_CLASSES = dl_config["NUM_CLASSES"]

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    y_train_onehot = to_categorical(y_train_encoded, num_classes=NUM_CLASSES)
    y_test_onehot = to_categorical(y_test_encoded, num_classes=NUM_CLASSES)

    print("\nBuilding & Training Bidirectional LSTM Model...")
    lstm_model = Sequential(name='Bidirectional_LSTM')
    lstm_model.add(Input(shape=(MAX_LEN,)))
    lstm_model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=128))
    lstm_model.add(SpatialDropout1D(0.2))
    lstm_model.add(Bidirectional(LSTM(64, return_sequences=True)))
    lstm_model.add(Bidirectional(LSTM(32, return_sequences=False)))
    lstm_model.add(Dense(64, activation='relu'))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(Dense(NUM_CLASSES, activation='softmax'))
    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop_lstm = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    # Using batch_size=128 to optimize memory allocation and speed on Windows CPU
    lstm_model.fit(X_train_pad, y_train_onehot, epochs=15, batch_size=128, validation_split=0.1, callbacks=[early_stop_lstm], verbose=2)

    loss_lstm, acc_lstm = lstm_model.evaluate(X_test_pad, y_test_onehot, verbose=0)
    print(f"--> Bidirectional LSTM Test Accuracy: {acc_lstm*100:.2f}%")
    lstm_model.save(os.path.join(ARTIFACTS_DIR, "lstm_model.keras"))
    print("Saved lstm_model.keras successfully!")

if __name__ == "__main__":
    main()
