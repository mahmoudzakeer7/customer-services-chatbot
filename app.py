"""
Streamlit Web Application for Customer Service Chatbot
Ensemble Intent Classification Dashboard comparing Logistic Regression, Simple RNN, and LSTM.
Automatically selects the best model based on prediction confidence and visualizes the reasoning.
"""

import os
import re
import random
import pickle
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Set page configuration with rich title and layout
st.set_page_config(
    page_title="AI Customer Service Chatbot & Ensemble Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = r"d:\download\nlpProject"
ARTIFACTS_DIR = os.path.join(BASE_DIR, "models")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_models.py")

# =====================================================================
# CUSTOM CSS STYLING FOR PREMIUM AESTHETICS (DYNAMIC & WOW FACTOR)
# =====================================================================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Gradient Banner */
    .hero-banner {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.3);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 400;
    }
    
    /* Winner Card Styling */
    .winner-card {
        background: rgba(16, 185, 129, 0.1);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.15);
    }
    .winner-badge {
        background-color: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Chat Response Box */
    .chat-response {
        background: rgba(30, 41, 59, 0.8);
        border-left: 4px solid #6366f1;
        padding: 1.25rem;
        border-radius: 0 12px 12px 0;
        font-size: 1.15rem;
        line-height: 1.6;
        margin: 1rem 0;
    }
    
    /* Dashboard section cards */
    .dash-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# MODEL LOADING & AUTO-TRAINING FALLBACK
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_all_artifacts():
    """Loads pre-trained models and preprocessing artifacts. Trains if missing."""
    required_files = [
        "intent_responses.pkl", "vectorizer.pkl", "lr_model.pkl",
        "tokenizer.pkl", "label_encoder.pkl", "dl_config.pkl",
        "rnn_model.keras", "lstm_model.keras"
    ]
    
    missing = [f for f in required_files if not os.path.exists(os.path.join(ARTIFACTS_DIR, f))]
    
    if missing:
        with st.status("⚙️ Pre-trained models not found. Initializing AI training pipeline... (approx. 2-3 mins)", expanded=True) as status:
            st.write("Loading 27K customer responses dataset...")
            st.write("Training Logistic Regression baseline...")
            st.write("Training Simple RNN & Bidirectional LSTM neural networks...")
            res = subprocess.run(["python", TRAIN_SCRIPT], capture_output=True, text=True)
            if res.returncode != 0:
                st.error(f"Training failed:\n{res.stderr}")
                st.stop()
            status.update(label="✅ All AI Models Trained & Saved Successfully!", state="complete", expanded=False)
    
    # Import tensorflow inside cache to avoid reloading
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    
    with open(os.path.join(ARTIFACTS_DIR, "intent_responses.pkl"), "rb") as f:
        intent_responses = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "lr_model.pkl"), "rb") as f:
        lr_model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "dl_config.pkl"), "rb") as f:
        dl_config = pickle.load(f)
        
    rnn_model = load_model(os.path.join(ARTIFACTS_DIR, "rnn_model.keras"))
    lstm_model = load_model(os.path.join(ARTIFACTS_DIR, "lstm_model.keras"))
    
    return intent_responses, vectorizer, lr_model, tokenizer, label_encoder, dl_config, rnn_model, lstm_model

# Clean text helper
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\{\{.*?\}\}', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# =====================================================================
# SIDEBAR NAVIGATION & INFO
# =====================================================================
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1531482615713-2afd69097998?auto=format&fit=crop&w=600&q=80", use_container_width=True)
    st.markdown("### ⚙️ System Architecture")
    st.markdown("""
    This chatbot utilizes a **Dynamic Ensemble Selection** mechanism evaluating three distinct architectures in real-time:
    
    1. **Logistic Regression**: TF-IDF n-gram baseline model. Fast and highly accurate on standard phrasing.
    2. **Simple RNN**: Recurrent Neural Network capturing sequential context across 128D embeddings.
    3. **Bidirectional LSTM**: State-of-the-art Deep Learning model processing sequences forward and backward for nuanced intent extraction.
    """)
    st.divider()
    st.markdown("### 📊 Dataset Specs")
    st.info("**Training Corpus**: Bitext 27K Customer Support Dataset\n\n**Intent Classes**: 27 Distinct Categories\n\n**Vocabulary Cap**: 10,000 Words")


# =====================================================================
# MAIN APP HEADER
# =====================================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">AI Customer Service Chatbot</div>
    <div class="hero-subtitle">Powered by Real-Time Ensemble Confidence Scoring & Deep Learning Analytics</div>
</div>
""", unsafe_allow_html=True)

# Load artifacts
with st.spinner("Loading ensemble models into memory..."):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    intent_responses, vectorizer, lr_model, tokenizer, label_encoder, dl_config, rnn_model, lstm_model = load_all_artifacts()

MAX_LEN = dl_config["MAX_LEN"]

# =====================================================================
# USER INPUT SECTION
# =====================================================================
st.markdown("### 💬 Customer Message")
sample_queries = [
    "I want to cancel my order immediately",
    "Where is my refund? I haven't received the money yet",
    "How do I change my shipping address on an active purchase?",
    "Can I speak to a human agent please?",
    "I received a damaged item and need a replacement"
]

col_input, col_sample = st.columns([3, 1])
with col_sample:
    selected_sample = st.selectbox("Or try a sample query:", ["-- Select Sample --"] + sample_queries)

default_text = selected_sample if selected_sample != "-- Select Sample --" else ""
user_input = st.text_input("Type how we can assist you today:", value=default_text, placeholder="e.g., I need help tracking my latest order...")


# =====================================================================
# ENSEMBLE PREDICTION & CONFIDENCE EVALUATION
# =====================================================================
if user_input.strip():
    cleaned = clean_text(user_input)
    
    if not cleaned:
        st.warning("⚠️ Please enter a valid text query containing words or numbers.")
    else:
        # --- Model 1: Logistic Regression ---
        vec = vectorizer.transform([cleaned])
        lr_probs = lr_model.predict_proba(vec)[0]
        lr_class_idx = np.argmax(lr_probs)
        lr_intent = lr_model.classes_[lr_class_idx]
        lr_conf = float(lr_probs[lr_class_idx])
        lr_resp = random.choice(intent_responses.get(lr_intent, ["I'm not sure how to help with that."]))
        lr_resp = re.sub(r'\{\{.*?\}\}', '[INFO]', lr_resp)
        
        # --- Model 2 & 3 Preprocessing (DL) ---
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # --- Model 2: Simple RNN ---
        rnn_probs = rnn_model.predict(padded, verbose=0)[0]
        rnn_class_idx = np.argmax(rnn_probs)
        rnn_intent = label_encoder.inverse_transform([rnn_class_idx])[0]
        rnn_conf = float(rnn_probs[rnn_class_idx])
        rnn_resp = random.choice(intent_responses.get(rnn_intent, ["I'm not sure how to help with that."]))
        rnn_resp = re.sub(r'\{\{.*?\}\}', '[INFO]', rnn_resp)
        
        # --- Model 3: Bidirectional LSTM ---
        lstm_probs = lstm_model.predict(padded, verbose=0)[0]
        lstm_class_idx = np.argmax(lstm_probs)
        lstm_intent = label_encoder.inverse_transform([lstm_class_idx])[0]
        lstm_conf = float(lstm_probs[lstm_class_idx])
        lstm_resp = random.choice(intent_responses.get(lstm_intent, ["I'm not sure how to help with that."]))
        lstm_resp = re.sub(r'\{\{.*?\}\}', '[INFO]', lstm_resp)
        
        # --- Ensemble Decision (Highest Confidence) ---
        models_data = [
            {"name": "Bidirectional LSTM", "intent": lstm_intent, "conf": lstm_conf, "resp": lstm_resp, "probs": lstm_probs, "classes": label_encoder.classes_},
            {"name": "Simple RNN", "intent": rnn_intent, "conf": rnn_conf, "resp": rnn_resp, "probs": rnn_probs, "classes": label_encoder.classes_},
            {"name": "Logistic Regression", "intent": lr_intent, "conf": lr_conf, "resp": lr_resp, "probs": lr_probs, "classes": lr_model.classes_}
        ]
        
        # Sort by confidence descending
        models_data.sort(key=lambda x: x["conf"], reverse=True)
        winner = models_data[0]
        
        # Display Winner Chat Response
        st.markdown(f"""
        <div class="winner-card">
            <span class="winner-badge">⚡ SELECTED ENSEMBLE MODEL: {winner['name'].upper()}</span>
            <div style="margin-top: 1rem; font-size: 1.25rem; font-weight: 600; color: #10b981;">
                Predicted Intent: <span style="color: white; background: #334155; padding: 0.2rem 0.6rem; border-radius: 6px;">{winner['intent']}</span> 
                <span style="font-size: 1rem; color: #94a3b8; margin-left: 1rem;">Confidence: <b>{winner['conf']*100:.2f}%</b></span>
            </div>
            <div class="chat-response">
                💬 <b>Chatbot Response:</b><br>{winner['resp']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # =====================================================================
        # ANALYTICAL DASHBOARD: WHY THE SYSTEM CHOSE THIS MODEL
        # =====================================================================
        st.markdown("## 📈 Analytical Dashboard: Why Did the System Choose This Model?")
        st.markdown("The system compares real-time softmax probability outputs across all three AI models. The architecture demonstrating the **highest classification certainty** is autonomously selected to guarantee the most accurate support response.")
        
        # KPI Cards
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric("🏆 Winning Architecture", winner["name"], delta=f"+{(winner['conf'] - models_data[1]['conf'])*100:.1f}% vs runner-up")
        with kpi2:
            st.metric("🎯 Prediction Confidence", f"{winner['conf']*100:.2f}%", delta="High Certainty" if winner['conf'] > 0.85 else "Moderate")
        with kpi3:
            st.metric("🏷️ Classified Intent", winner["intent"])
            
        st.write("")
        
        # Charts Row
        col_chart1, col_chart2 = st.columns([1, 1])
        
        with col_chart1:
            st.markdown("#### ⚖️ Confidence Comparison Across Models")
            conf_df = pd.DataFrame({
                "Model": [m["name"] for m in models_data],
                "Confidence (%)": [m["conf"] * 100 for m in models_data],
                "Status": ["Winner ⭐" if i == 0 else "Runner-up" for i in range(len(models_data))]
            })
            
            chart = alt.Chart(conf_df).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
                x=alt.X('Model:N', sort='-y', axis=alt.Axis(labelAngle=0, title=None)),
                y=alt.Y('Confidence (%):Q', scale=alt.Scale(domain=[0, 100])),
                color=alt.Color('Status:N', scale=alt.Scale(domain=['Winner ⭐', 'Runner-up'], range=['#10b981', '#6366f1'])),
                tooltip=['Model', alt.Tooltip('Confidence (%):Q', format='.2f'), 'Status']
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)
            
        with col_chart2:
            st.markdown(f"#### 🔍 Top 5 Intent Probabilities ({winner['name']})")
            # Extract top 5 probabilities for winning model
            top_indices = np.argsort(winner["probs"])[::-1][:5]
            top_intents = [winner["classes"][i] for i in top_indices]
            top_probs = [float(winner["probs"][i]) * 100 for i in top_indices]
            
            prob_df = pd.DataFrame({
                "Intent": top_intents,
                "Probability (%)": top_probs
            })
            
            chart2 = alt.Chart(prob_df).mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8, color='#a855f7').encode(
                y=alt.Y('Intent:N', sort='-x', axis=alt.Axis(title=None)),
                x=alt.X('Probability (%):Q', scale=alt.Scale(domain=[0, 100])),
                tooltip=['Intent', alt.Tooltip('Probability (%):Q', format='.2f')]
            ).properties(height=300)
            
            st.altair_chart(chart2, use_container_width=True)
            
        # Side-by-Side Table
        st.markdown("#### 📋 Detailed Side-by-Side Model Diagnostics")
        diag_df = pd.DataFrame({
            "AI Model": [m["name"] for m in models_data],
            "Predicted Intent": [m["intent"] for m in models_data],
            "Confidence Score": [f"{m['conf']*100:.2f}%" for m in models_data],
            "Selection Status": ["🏆 Winner (Selected)" if i == 0 else "🥈 Runner-up" for i in range(len(models_data))],
            "Generated Response Snippet": [m["resp"][:80] + "..." if len(m["resp"]) > 80 else m["resp"] for m in models_data]
        })
        st.dataframe(diag_df, use_container_width=True, hide_index=True)
