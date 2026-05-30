import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Epilepsy Seizure Detector",
    page_icon="🧠",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

    * { font-family: 'Rajdhani', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a0a1a 100%);
        color: #e0e0e0;
    }

    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.8rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00d4ff, #7b2fff, #00d4ff);
        background-size: 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s infinite;
        padding: 1rem 0 0.2rem 0;
        letter-spacing: 3px;
    }

    @keyframes shimmer {
        0% { background-position: 0% }
        100% { background-position: 200% }
    }

    .subtitle {
        text-align: center;
        color: #7ecfff;
        font-size: 1.1rem;
        letter-spacing: 2px;
        margin-bottom: 2rem;
    }

    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    .result-seizure {
        background: linear-gradient(135deg, rgba(255,50,50,0.2), rgba(180,0,0,0.3));
        border: 2px solid #ff3333;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #ff5555;
        animation: pulse-red 1.5s infinite;
    }

    .result-normal {
        background: linear-gradient(135deg, rgba(0,255,150,0.15), rgba(0,180,100,0.2));
        border: 2px solid #00ff96;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #00ff96;
        animation: pulse-green 1.5s infinite;
    }

    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 15px rgba(255,50,50,0.4); }
        50% { box-shadow: 0 0 35px rgba(255,50,50,0.8); }
    }

    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 15px rgba(0,255,150,0.3); }
        50% { box-shadow: 0 0 35px rgba(0,255,150,0.7); }
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7b2fff);
        color: white;
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 2px;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,212,255,0.4);
    }

    .stTextArea > div > textarea {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 10px;
        color: #e0e0e0;
        font-family: 'Rajdhani', monospace;
    }

    .stRadio > div {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 0.5rem;
    }

    .stat-box {
        background: rgba(0,212,255,0.08);
        border: 1px solid rgba(0,212,255,0.25);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .stat-number {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 900;
        color: #00d4ff;
    }

    .stat-label {
        font-size: 0.85rem;
        color: #88aacc;
        letter-spacing: 1px;
    }

    .section-title {
        font-family: 'Orbitron', monospace;
        font-size: 1rem;
        color: #00d4ff;
        letter-spacing: 2px;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid rgba(0,212,255,0.2);
    }

    .info-tag {
        display: inline-block;
        background: rgba(123,47,255,0.2);
        border: 1px solid rgba(123,47,255,0.5);
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        color: #bb88ff;
        margin: 0.2rem;
    }

    .feature-label {
        font-size: 0.8rem;
        color: #7ecfff;
        margin-bottom: 0.2rem;
        letter-spacing: 1px;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 1px dashed rgba(0,212,255,0.3);
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ────────────────────────────────────────────────────────────────
# Top 10 feature indices selected by SelectKBest (ANOVA F-test) from 178 EEG features
# Corresponding to original columns: X9, X10, X11, X12, X26, X27, X43, X44, X94, X95
TOP_FEATURE_INDICES = [8, 9, 10, 11, 25, 26, 42, 43, 93, 94]
TOP_FEATURE_NAMES   = ["X9", "X10", "X11", "X12", "X26", "X27", "X43", "X44", "X94", "X95"]

@st.cache_resource
def load_model():
    model  = pickle.load(open('epilepsy_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl',         'rb'))
    # Load custom feature indices if saved separately, else use hardcoded
    if os.path.exists('top_indices.pkl'):
        indices = pickle.load(open('top_indices.pkl', 'rb'))
    else:
        indices = TOP_FEATURE_INDICES
    return model, scaler, indices

try:
    model, scaler, top_indices = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error   = str(e)


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 EPILEPSY SEIZURE DETECTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">⚡ AI-POWERED EEG ANALYSIS SYSTEM ⚡</div>', unsafe_allow_html=True)

if model_loaded:
    st.success("✅ Model Loaded Successfully")
else:
    st.error(f"❌ Model not found! Please place `epilepsy_model.pkl`, `scaler.pkl`, and `top_indices.pkl` in the same folder as app.py\n\nError: {load_error}")
    st.stop()

st.markdown("---")

# ─── Stats Row ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-box"><div class="stat-number">10</div><div class="stat-label">EEG FEATURES</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-box"><div class="stat-number">2</div><div class="stat-label">CLASSES</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-box"><div class="stat-number">SVC</div><div class="stat-label">MODEL USED</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="stat-box"><div class="stat-number">~95%</div><div class="stat-label">ACCURACY</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Main Layout ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    st.markdown('<div class="section-title">📡 INPUT METHOD</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎛️ Individual Fields", "✍️ Comma Input", "📁 Upload CSV"])

    # ── Tab 1: Individual number fields ────────────────────────────────────────
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("Enter the **10 most important EEG signal values**:")
        st.markdown("<br>", unsafe_allow_html=True)

        feat_values = []
        cols_row1 = st.columns(5)
        cols_row2 = st.columns(5)
        for i, (col, fname) in enumerate(zip(cols_row1 + cols_row2, TOP_FEATURE_NAMES)):
            with col:
                st.markdown(f'<div class="feature-label">{fname}</div>', unsafe_allow_html=True)
                val = st.number_input(
                    label=fname,
                    value=0.0,
                    step=0.1,
                    format="%.2f",
                    label_visibility="collapsed",
                    key=f"feat_{fname}"
                )
                feat_values.append(val)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn_fields = st.button("🔍  ANALYZE EEG SIGNAL", key="fields_predict")
        st.markdown('</div>', unsafe_allow_html=True)

        if predict_btn_fields:
            arr_10 = np.array(feat_values).reshape(1, -1)
            # Build a full 178-feature zero vector, fill in the 10 selected positions
            arr_full = np.zeros((1, 178))
            for j, idx in enumerate(top_indices):
                arr_full[0, idx] = arr_10[0, j]
            arr_scaled = scaler.transform(arr_full)
            # Keep only the selected features for prediction
            arr_selected = arr_scaled[:, top_indices]
            result = model.predict(arr_selected)[0]

            st.markdown("<br>", unsafe_allow_html=True)
            if result == 1:
                st.markdown(
                    '<div class="result-seizure">🚨 SEIZURE DETECTED<br><small style="font-size:0.7rem;font-weight:400;color:#ffaaaa;">IMMEDIATE ATTENTION REQUIRED</small></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="result-normal">✅ NO SEIZURE<br><small style="font-size:0.7rem;font-weight:400;color:#aaffcc;">NORMAL EEG PATTERN</small></div>',
                    unsafe_allow_html=True
                )

    # ── Tab 2: Comma-separated manual input ────────────────────────────────────
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        feature_list = ", ".join(TOP_FEATURE_NAMES)
        st.markdown(f"Enter **10 EEG values** separated by commas — in this order:  \n`{feature_list}`")

        input_data = st.text_area(
            label="EEG Values",
            placeholder="e.g.  14, -2, 0, 5, 3, -1, 7, 2, 10, -4",
            height=100,
            label_visibility="collapsed"
        )

        predict_btn = st.button("🔍  ANALYZE EEG SIGNAL", key="manual_predict")
        st.markdown('</div>', unsafe_allow_html=True)

        if predict_btn:
            if not input_data.strip():
                st.warning("⚠️ Please enter EEG values first.")
            else:
                try:
                    values = [float(v.strip()) for v in input_data.split(',')]
                    if len(values) != 10:
                        st.error(f"❌ Expected 10 values — you entered {len(values)}")
                    else:
                        arr_10    = np.array(values).reshape(1, -1)
                        arr_full  = np.zeros((1, 178))
                        for j, idx in enumerate(top_indices):
                            arr_full[0, idx] = arr_10[0, j]
                        arr_scaled   = scaler.transform(arr_full)
                        arr_selected = arr_scaled[:, top_indices]
                        result       = model.predict(arr_selected)[0]

                        st.markdown("<br>", unsafe_allow_html=True)
                        if result == 1:
                            st.markdown(
                                '<div class="result-seizure">🚨 SEIZURE DETECTED<br><small style="font-size:0.7rem;font-weight:400;color:#ffaaaa;">IMMEDIATE ATTENTION REQUIRED</small></div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                '<div class="result-normal">✅ NO SEIZURE<br><small style="font-size:0.7rem;font-weight:400;color:#aaffcc;">NORMAL EEG PATTERN</small></div>',
                                unsafe_allow_html=True
                            )
                except ValueError:
                    st.error("❌ Invalid input — please enter numeric values only.")

    # ── Tab 3: Batch CSV upload ─────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        feature_list = ", ".join(TOP_FEATURE_NAMES)
        st.markdown(f"Upload a CSV with these **10 columns** (in any order):  \n`{feature_list}`  \nOR upload the full 178-column dataset — the app will auto-select the right columns.")
        uploaded_file = st.file_uploader("Choose CSV", type=["csv"], label_visibility="collapsed")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.info(f"📊 Loaded: **{df.shape[0]} rows × {df.shape[1]} columns**")

        batch_btn = st.button("🔍  ANALYZE BATCH", key="batch_predict")
        st.markdown('</div>', unsafe_allow_html=True)

        if batch_btn:
            if uploaded_file is None:
                st.warning("⚠️ Please upload a CSV file first.")
            else:
                try:
                    # If CSV has named columns matching TOP_FEATURE_NAMES, use those
                    named_cols_present = all(c in df.columns for c in TOP_FEATURE_NAMES)
                    if named_cols_present:
                        raw = df[TOP_FEATURE_NAMES].values
                        arr_full = np.zeros((raw.shape[0], 178))
                        for j, idx in enumerate(top_indices):
                            arr_full[:, idx] = raw[:, j]
                    elif df.shape[1] >= 178:
                        # Full dataset — use column positions 1..178 (skip index col)
                        start = 1 if df.shape[1] > 178 else 0
                        arr_full = df.iloc[:, start:start+178].values
                    elif df.shape[1] == 10:
                        # Assume columns are exactly the 10 features in order
                        raw = df.iloc[:, :10].values
                        arr_full = np.zeros((raw.shape[0], 178))
                        for j, idx in enumerate(top_indices):
                            arr_full[:, idx] = raw[:, j]
                    else:
                        st.error(f"❌ CSV has {df.shape[1]} columns. Expected either 10 (top features) or 178+ (full dataset).")
                        st.stop()

                    scaled = scaler.transform(arr_full)
                    selected = scaled[:, top_indices]
                    preds = model.predict(selected)

                    df['🔴 Prediction'] = ['🚨 SEIZURE' if p == 1 else '✅ NORMAL' for p in preds]
                    seizure_count = sum(preds == 1)
                    normal_count  = sum(preds == 0)

                    c1, c2 = st.columns(2)
                    c1.metric("🚨 Seizure Cases", seizure_count)
                    c2.metric("✅ Normal Cases",  normal_count)

                    st.dataframe(df[['🔴 Prediction']].join(df.drop(columns=['🔴 Prediction'])), use_container_width=True)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Results", csv, "seizure_results.csv", "text/csv")

                except Exception as e:
                    st.error(f"❌ Error: {e}")


with right_col:
    st.markdown('<div class="section-title">ℹ️ ABOUT THIS SYSTEM</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <b style="color:#00d4ff;">📌 What is this?</b><br>
        This system uses Machine Learning to detect epileptic seizures from EEG (Electroencephalogram) brain signal data.
        <br><br>
        <b style="color:#00d4ff;">🔬 Dataset</b><br>
        Trained on the <b>Epileptic Seizure Recognition Dataset</b> with 11,500 patient records and 178 EEG time-series features.
        <br><br>
        <b style="color:#00d4ff;">🤖 Models Trained</b><br>
        <span class="info-tag">Logistic Regression</span>
        <span class="info-tag">SVM</span>
        <span class="info-tag">LinearSVC</span>
        <span class="info-tag">KNN</span>
        <span class="info-tag">ANN</span>
        <br><br>
        <b style="color:#00d4ff;">🏆 Best Model</b><br>
        Support Vector Machine (SVC) — trained on the <b>top 10 selected features</b> using ANOVA F-test (SelectKBest), achieving ~95% accuracy.
        <br><br>
        <b style="color:#00d4ff;">⚡ Feature Reduction</b><br>
        From 178 → <b>10 features</b> with zero accuracy loss. Selected features: <b>X9, X10, X11, X12, X26, X27, X43, X44, X94, X95</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📋 OUTPUT CLASSES</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div style="display:flex; align-items:center; margin-bottom:1rem; gap:1rem;">
            <div style="font-size:2rem;">🚨</div>
            <div>
                <div style="color:#ff5555; font-family:'Orbitron',monospace; font-weight:700;">SEIZURE (Class 1)</div>
                <div style="color:#aaa; font-size:0.85rem;">Epileptic brain activity detected in EEG signal</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:1rem;">
            <div style="font-size:2rem;">✅</div>
            <div>
                <div style="color:#00ff96; font-family:'Orbitron',monospace; font-weight:700;">NORMAL (Class 0)</div>
                <div style="color:#aaa; font-size:0.85rem;">No epileptic activity — normal EEG pattern</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">⚠️ DISCLAIMER</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="border-color:rgba(255,200,0,0.3);">
        <span style="color:#ffcc00;">⚠️</span> This tool is for <b>educational and research purposes only</b>.
        It is <b>NOT</b> a substitute for professional medical diagnosis.
        Always consult a qualified neurologist for medical advice.
    </div>
    """, unsafe_allow_html=True)


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#445566; font-size:0.85rem; letter-spacing:1px;">
     -><b style="color:#7b2fff;"></b> · ML Epilepsy Detection Project · Powered by Scikit-Learn & Streamlit
</div>
""", unsafe_allow_html=True)
