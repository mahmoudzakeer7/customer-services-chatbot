"""
Standalone Training & Artifact Generation Script for Customer Service Chatbot
Trains Logistic Regression, Simple RNN, and Bidirectional LSTM models and saves them
for quick loading in the Streamlit deployment app.
"""

import sys
# Configure UTF-8 encoding to prevent Windows console encoding errors with Keras progress bars
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, SimpleRNN, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Set seeds for reproducibility
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
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print("="*60)
    print("STARTING CHATBOT MODEL TRAINING & ARTIFACT GENERATION")
    print("="*60)

    # 1. Load & Clean Data
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    data = df[['instruction', 'intent', 'response', 'category']].copy()
    data.dropna(subset=['instruction', 'intent', 'response'], inplace=True)
    
    print("Cleaning text instructions...")
    data['clean_instruction'] = data['instruction'].apply(clean_text)
    data = data[data['clean_instruction'].str.len() > 0]
    print(f"Dataset ready: {len(data):,} valid rows.")

    # Build intent -> response dictionary
    intent_responses = {}
    for intent in data['intent'].unique():
        responses = data[data['intent'] == intent]['response'].dropna().tolist()
        intent_responses[intent] = list(set(responses))
    
    with open(os.path.join(ARTIFACTS_DIR, "intent_responses.pkl"), "wb") as f:
        pickle.dump(intent_responses, f)
    print("Saved intent_responses.pkl")

    # 2. Train / Test Split
    print("\n[2/6] Splitting train/test data (80/20)...")
    X = data['clean_instruction']
    y = data['intent']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Baseline: Logistic Regression + TF-IDF
    print("\n[3/6] Training Baseline Logistic Regression model...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    lr_model = LogisticRegression(C=5, max_iter=1000, solver='lbfgs', n_jobs=-1)
    lr_model.fit(X_train_tfidf, y_train)
    
    acc_lr = lr_model.score(X_test_tfidf, y_test)
    print(f"--> Logistic Regression Test Accuracy: {acc_lr*100:.2f}%")

    with open(os.path.join(ARTIFACTS_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(ARTIFACTS_DIR, "lr_model.pkl"), "wb") as f:
        pickle.dump(lr_model, f)
    print("Saved vectorizer.pkl and lr_model.pkl")

    # 4. Deep Learning Preprocessing (Tokenization & Padding)
    print("\n[4/6] Tokenizing and padding sequences for Deep Learning...")
    VOCAB_SIZE = 10000
    OOV_TOKEN = '<OOV>'
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    seq_lengths = [len(seq) for seq in X_train_seq]
    MAX_LEN = int(np.percentile(seq_lengths, 95))
    print(f"Chosen MAX_LEN = {MAX_LEN}")

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    NUM_CLASSES = len(label_encoder.classes_)

    y_train_onehot = to_categorical(y_train_encoded, num_classes=NUM_CLASSES)
    y_test_onehot = to_categorical(y_test_encoded, num_classes=NUM_CLASSES)

    with open(os.path.join(ARTIFACTS_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(ARTIFACTS_DIR, "dl_config.pkl"), "wb") as f:
        pickle.dump({"MAX_LEN": MAX_LEN, "VOCAB_SIZE": VOCAB_SIZE, "EMBEDDING_DIM": 128, "NUM_CLASSES": NUM_CLASSES}, f)
    print("Saved tokenizer.pkl, label_encoder.pkl, and dl_config.pkl")

    # 5. Simple RNN Model
    print("\n[5/6] Building & Training Simple RNN Model...")
    rnn_model = Sequential(name='Simple_RNN')
    rnn_model.add(Input(shape=(MAX_LEN,)))
    rnn_model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=128))
    rnn_model.add(SpatialDropout1D(0.2))
    rnn_model.add(SimpleRNN(units=64, return_sequences=False))
    rnn_model.add(Dropout(0.3))
    rnn_model.add(Dense(NUM_CLASSES, activation='softmax'))
    rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    # Use verbose=2 (one line per epoch) to avoid Unicode encoding errors on Windows console
    rnn_model.fit(X_train_pad, y_train_onehot, epochs=15, batch_size=64, validation_split=0.1, callbacks=[early_stop], verbose=2)
    
    loss_rnn, acc_rnn = rnn_model.evaluate(X_test_pad, y_test_onehot, verbose=0)
    print(f"--> Simple RNN Test Accuracy: {acc_rnn*100:.2f}%")
    rnn_model.save(os.path.join(ARTIFACTS_DIR, "rnn_model.keras"))
    print("Saved rnn_model.keras")

    # 6. Bidirectional LSTM Model
    print("\n[6/6] Building & Training Bidirectional LSTM Model...")
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
    # Use verbose=2 (one line per epoch) and batch_size=128 to avoid memory or encoding errors on Windows console
    lstm_model.fit(X_train_pad, y_train_onehot, epochs=15, batch_size=128, validation_split=0.1, callbacks=[early_stop_lstm], verbose=2)

    loss_lstm, acc_lstm = lstm_model.evaluate(X_test_pad, y_test_onehot, verbose=0)
    print(f"--> Bidirectional LSTM Test Accuracy: {acc_lstm*100:.2f}%")
    lstm_model.save(os.path.join(ARTIFACTS_DIR, "lstm_model.keras"))
    print("Saved lstm_model.keras")

    print("\n" + "="*60)
    print("ALL MODELS TRAINED & ARTIFACTS SAVED SUCCESSFULLY TO:")
    print(ARTIFACTS_DIR)
    print("="*60)

if __name__ == "__main__":
    main()
