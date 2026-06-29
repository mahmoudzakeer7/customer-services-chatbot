# 🤖 AI Customer Service Chatbot & Ensemble Intent Classification Dashboard

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

An enterprise-grade **Customer Support Chatbot** powered by a **Real-Time Dynamic Ensemble Selection** engine. This system compares three distinct Natural Language Processing (NLP) and Deep Learning architectures on every user query, autonomously selecting the model with the highest classification confidence and visually explaining the decision through an interactive analytical dashboard.

---

## 🌟 Key Features

- **⚡ Real-Time Ensemble Selection:** Simultaneously evaluates user inputs across **Logistic Regression**, **Simple RNN**, and **Bidirectional LSTM** models, choosing the winner based on the maximum softmax probability score.
- **📊 Interactive Analytical Dashboard:** Visualizes *why* the winning model was chosen using rich Altair bar charts, confidence comparisons, top-5 probability breakdowns, and side-by-side diagnostic tables.
- **💎 Premium UI/UX Aesthetics:** Designed with custom gradient banners, sleek dark-mode styling, glassmorphism cards, and responsive metric KPIs.
- **🛡️ Auto-Fallback Training Pipeline:** Built-in caching (`@st.cache_resource`) that detects missing model artifacts on startup and automatically initializes background training with an interactive progress spinner.

---

## 🏗️ System Architecture & Model Performance

The project trains and compares three models on the **Bitext 27K Customer Support Dataset** (~27,000 real-world customer instructions categorized into **27 distinct intent classes**).

| Model Architecture | Feature Representation | Key Layers / Parameters | Test Accuracy | Role in Ensemble |
| :--- | :--- | :--- | :---: | :--- |
| **Logistic Regression** | TF-IDF (1-2 n-grams, 5K max features) | `C=5`, L-BFGS Solver | **99.50%** | Fast baseline; excels at exact keyword & phrase matching |
| **Simple RNN** | Word Tokenization (10K vocab) | 128D Embedding + `SpatialDropout1D(0.2)` + `SimpleRNN(64)` | **98.70%** | Recurrent memory capturing sequential word order |
| **Bidirectional LSTM** | Word Tokenization (10K vocab) | 128D Embedding + Stacked `Bi-LSTM(64/32)` + `Dense(64)` | **98.96%** | Deep forward/backward sequence modeling for complex phrasing |

---

## 📂 Project Structure

```text
d:\download\nlpProject\
│
├── Customer_Service_Chatbot.ipynb # Original exploratory Jupyter Notebook with step-by-step NLP walkthrough
├── train_models.py                # Unified standalone script to train all 3 models and export pre-trained artifacts
├── train_lstm_only.py             # Optimized script for training Bidirectional LSTM (batch_size=128)
├── app.py                         # Production Streamlit web application & ensemble analytical dashboard
│
└── models/                        # Cached pre-trained models and preprocessors (auto-generated)
    ├── intent_responses.pkl       # Mapping of 27 intent classes to curated customer support responses
    ├── vectorizer.pkl             # Fitted TfidfVectorizer (5,000 vocabulary n-grams)
    ├── lr_model.pkl               # Trained Logistic Regression classifier
    ├── tokenizer.pkl              # Fitted Keras Tokenizer (10,000 vocabulary)
    ├── label_encoder.pkl          # Scikit-learn LabelEncoder for 27 intent strings
    ├── dl_config.pkl              # Configuration metadata (e.g., MAX_LEN=13, VOCAB_SIZE=10000)
    ├── rnn_model.keras            # Saved Keras 3 Simple RNN neural network
    └── lstm_model.keras           # Saved Keras 3 Bidirectional LSTM neural network
```

---

## ⚙️ How the Ensemble Engine Works

```mermaid
graph TD
    A[User Input Message] --> B[Text Cleaning & Normalization]
    
    B --> C[TF-IDF Vectorization]
    B --> D[Tokenization & Sequence Padding]
    
    C --> E[Logistic Regression Model]
    D --> F[Simple RNN Model]
    D --> G[Bidirectional LSTM Model]
    
    E -->|Intent & Prob (%)| H[Ensemble Confidence Evaluator]
    F -->|Intent & Prob (%)| H
    G -->|Intent & Prob (%)| H
    
    H -->|Select max confidence| I[🏆 Winning Model Selected]
    I --> J[Render Chat Response & Analytical Dashboard]
```

1. **Text Normalization:** Lowercases text, removes special punctuation, and strips template placeholders like `{{Order Number}}`.
2. **Parallel Inference:** Passes the normalized query through TF-IDF transformations for Scikit-Learn and Sequence Padding (`MAX_LEN=13`) for TensorFlow.
3. **Softmax Extraction:** Retrieves the exact classification probability array across all 27 classes for each model.
4. **Winner Selection:** Selects the model where `max(probability)` is highest. If Logistic Regression is 95% confident but Bi-LSTM is 99.4% confident, Bi-LSTM wins the round and handles the user query.

---

## 🚀 Getting Started & Installation

### 1. Prerequisites
Ensure you have Python 3.9+ installed along with the required data science libraries:

```powershell
pip install streamlit tensorflow scikit-learn pandas numpy altair matplotlib
```

### 2. Training the Models (Optional / Standalone)
If you wish to re-train all models from scratch and regenerate the `.pkl` and `.keras` artifacts, run:

```powershell
python train_models.py
```
*(Note: The script is optimized with UTF-8 console stream wrapping and `verbose=2` to ensure 100% stability on Windows environments.)*

### 3. Launching the Streamlit Deployment
Start the web dashboard locally:

```powershell
streamlit run app.py
```

Open your browser to `http://localhost:8501` to interact with the chatbot, test sample queries, and explore the visual comparison metrics!

---

## 💡 Sample Queries to Try in the Dashboard

- *"I want to cancel my order immediately"* (Usually won by Bi-LSTM or Logistic Regression with >99% confidence)
- *"Where is my refund? I haven't received the money yet"* (Tests sequence context understanding)
- *"How do I change my shipping address on an active purchase?"* (Demonstrates intent classification separation across top 5 probabilities)
- *"Can I speak to a human agent please?"* (Triggers customer support escalation intents)
