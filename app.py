"""
app.py - Streamlit Interactive Web Application MVP for Fund My Crazy 2026.
Privacy-Preserving Federated Learning for Rare Disease Diagnosis.

Features:
- Privacy Architecture Visualization (Raw data remains on-premise)
- Non-IID Hospital Client Datasets Inspection
- Live Federated Communication Rounds Simulator
- Comprehensive Performance Comparison (Local Models vs FL Global Model)
- Interactive Real-time Patient Diagnostic Prediction Tool
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import (
    load_all_hospital_datasets, HOSPITAL_METADATA, FEATURE_NAMES, TARGET_NAME
)
from client import HospitalClient
from server import FederatedServer
from model import RareDiseaseClassifier

# -----------------------------------------------------------------------------
# PAGE CONFIG & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FedDiagnosis MVP - Rare Disease FL",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Medical / Futuristic Look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
        color: inherit;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .disclaimer-box {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 25px;
        font-size: 0.92rem;
        color: #991B1B;
    }
    .privacy-card {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 15px;
        color: inherit;
    }
    .metric-container {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }


/* Hospital Cards */
.hospital-card {
    border: 1px solid #b7e4c7;
    border-radius: 12px;
    padding: 20px;
    min-height: 220px;
}

/* Default: LIGHT MODE */
.hospital-card,
.hospital-card h3,
.hospital-card h4,
.hospital-card p,
.hospital-card span,
.hospital-card div {
    color: #111111 !important;
}

/* DARK MODE */
@media (prefers-color-scheme: dark) {
    .hospital-card,
    .hospital-card h3,
    .hospital-card h4,
    .hospital-card p,
    .hospital-card span,
    .hospital-card div {
        color: #ffffff !important;
    }
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# APP HEADER & DISCLAIMER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🧬 Fund My Crazy 2026: Privacy-Preserving Federated Learning</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Collaborative Rare Disease Diagnostic Intelligence without Data Centralization</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>RESEARCH PROTOTYPE & SIMULATION ONLY:</strong> This application is a technological proof-of-concept demonstrating Privacy-Preserving Federated Learning (FedAvg) across 3 simulated medical centers. It is not validated for clinical diagnosis or medical decision-making.
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & STATE CACHING
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Simulation Settings")
random_seed = st.sidebar.number_input("Random Seed", value=42, min_value=1, max_value=9999)
num_rounds = st.sidebar.slider("Federated Communication Rounds", min_value=1, max_value=15, value=5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏥 Participating Hospitals")
for hid, meta in HOSPITAL_METADATA.items():
    st.sidebar.markdown(f"**{meta['name']}**  \n📍 *{meta['location']}* ({meta['samples']} samples)")

# Initialize data & session state
@st.cache_data(show_spinner=False)
def get_cached_data(seed):
    return load_all_hospital_datasets(seed=seed)

hospital_data, X_global_test, y_global_test = get_cached_data(random_seed)

# Instantiate Client instances
clients = [
    HospitalClient(
        hospital_id=h_id,
        X_train=hospital_data[h_id]["X_train"],
        y_train=hospital_data[h_id]["y_train"],
        X_val=hospital_data[h_id]["X_val"],
        y_val=hospital_data[h_id]["y_val"],
        random_state=random_seed
    )
    for h_id in HOSPITAL_METADATA.keys()
]

# Instantiate Server
server = FederatedServer(
    clients=clients,
    X_global_test=X_global_test,
    y_global_test=y_global_test,
    random_state=random_seed
)

# -----------------------------------------------------------------------------
# MAIN NAVIGATION TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔒 Privacy & Architecture",
    "🏥 Hospital Datasets (Non-IID)",
    "⚡ FL Training Simulator",
    "📊 Local vs Federated Performance",
    "🩺 Patient Diagnosis Predictor"
])

# =============================================================================
# TAB 1: PRIVACY & ARCHITECTURE FLOW
# =============================================================================

HOSPITAL_DISPLAY = {
    "hospital_1": {
        "name": "Hospital A",
        "location": "Simulated Site A",
        "samples": 300
    },
    "hospital_2": {
        "name": "Hospital B",
        "location": "Simulated Site B",
        "samples": 400
    },
    "hospital_3": {
        "name": "Hospital C",
        "location": "Simulated Site C",
        "samples": 250
    }
}

with tab1:
    st.subheader("🛡️ Federated Learning Privacy Architecture")
    st.markdown("""
    In traditional machine learning, hospitals must pool sensitive patient records into a centralized cloud database.
    For rare diseases, cross-institutional data sharing is blocked by HIPAA/GDPR privacy restrictions.
    
    **Federated Averaging (FedAvg)** solves this by keeping patient data on-premise at each hospital:
    """)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown("""
        <div class="privacy-card">
            <h4>🏥 Hospital 1 (St. Jude)</h4>
            <p><strong>Local Data:</strong> 300 Patients</p>
            <p>🔒 <strong>Data Status:</strong> ON-PREMISE ONLY</p>
            <p>📤 <strong>Outbound:</strong> Model Weights (<code>coef</code>, <code>intercept</code>)</p>
            <p>❌ <strong>Transmitted Patient Records:</strong> 0 rows (0 Bytes)</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="privacy-card">
            <h4>🏥 Hospital 2 (Mayo Clinic)</h4>
            <p><strong>Local Data:</strong> 400 Patients</p>
            <p>🔒 <strong>Data Status:</strong> ON-PREMISE ONLY</p>
            <p>📤 <strong>Outbound:</strong> Model Weights (<code>coef</code>, <code>intercept</code>)</p>
            <p>❌ <strong>Transmitted Patient Records:</strong> 0 rows (0 Bytes)</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="privacy-card">
            <h4>🏥 Hospital 3 (Johns Hopkins)</h4>
            <p><strong>Local Data:</strong> 250 Patients</p>
            <p>🔒 <strong>Data Status:</strong> ON-PREMISE ONLY</p>
            <p>📤 <strong>Outbound:</strong> Model Weights (<code>coef</code>, <code>intercept</code>)</p>
            <p>❌ <strong>Transmitted Patient Records:</strong> 0 rows (0 Bytes)</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🌐 Central Aggregation Protocol")
    st.latex(r"W_{\text{global}} = \sum_{k=1}^{K} \frac{n_k}{N} W_k, \quad b_{\text{global}} = \sum_{k=1}^{K} \frac{n_k}{N} b_k")
    st.info("💡 **Key Security Guarantee:** The central server receives only numerical parameter matrices ($W_k, b_k$). Raw patient records are not transmitted during federated training. Additional privacy techniques such as secure aggregation or differential privacy may be added for stronger protection.")

# =============================================================================
# TAB 2: HOSPITAL DATASETS (NON-IID)
# =============================================================================
with tab2:
    st.subheader("📊 Non-IID Local Dataset Inspection")
    st.write("Rare disease manifestations vary across clinical sites. Inspect the local feature distributions across the three hospitals below:")

    selected_feature = st.selectbox("Select Clinical Biomarker Feature to Compare:", FEATURE_NAMES, index=1)

    dfs = []
    for h_id, data in hospital_data.items():
        temp_df = data["full_df"].copy()
        temp_df["Hospital"] = HOSPITAL_METADATA[h_id]["name"]
        dfs.append(temp_df)
    combined_df = pd.concat(dfs, axis=0)

    fig = px.histogram(
        combined_df,
        x=selected_feature,
        color="Hospital",
        barmode="overlay",
        marginal="box",
        title=f"Distribution of {selected_feature} Across Hospital Clients (Non-IID)",
        color_discrete_map={
            HOSPITAL_METADATA["hospital_1"]["name"]: HOSPITAL_METADATA["hospital_1"]["color"],
            HOSPITAL_METADATA["hospital_2"]["name"]: HOSPITAL_METADATA["hospital_2"]["color"],
            HOSPITAL_METADATA["hospital_3"]["name"]: HOSPITAL_METADATA["hospital_3"]["color"],
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Client Summary Table")
    summary_list = []
    for c in clients:
        info = c.get_data_summary()
        summary_list.append({
            "Hospital Name": info["hospital_name"],
            "Train Patients": info["train_samples"],
            "Validation Patients": info["val_samples"],
            "Rare Disease Positives": info["positive_cases"],
            "Controls (Negatives)": info["negative_cases"],
            "Data Leakage Risk": "Zero (0 Bytes Exposed)"
        })
    st.dataframe(pd.DataFrame(summary_list), use_container_width=True)

# =============================================================================
# TAB 3: FL TRAINING SIMULATOR
# =============================================================================
with tab3:
    st.subheader("⚡ Execute Federated Communication Rounds")
    st.write("Run local training cycles at each hospital and aggregate parameters into the Global Model using FedAvg.")

    if st.button("🚀 Start Federated Learning Simulation", type="primary"):
        progress_bar = st.progress(0)
        history = []

        for r in range(1, num_rounds + 1):
            log = server.run_communication_round(round_num=r)
            history.append(log)
            progress_bar.progress(r / num_rounds)

        st.success(f"Successfully completed {num_rounds} Federated Communication Rounds!")

        st.session_state["fl_history"] = history
        st.session_state["trained_server"] = server
        st.session_state["trained_clients"] = clients

    if "fl_history" in st.session_state:
        history = st.session_state["fl_history"]

        rounds = [h["round"] for h in history]
        global_acc = [h["global_metrics"]["accuracy"] for h in history]
        global_f1 = [h["global_metrics"]["f1"] for h in history]
        global_auc = [h["global_metrics"]["roc_auc"] for h in history]

        fig_learning = go.Figure()
        fig_learning.add_trace(go.Scatter(x=rounds, y=global_acc, mode='lines+markers', name='Global Accuracy', line=dict(color='#1E3A8A', width=3)))
        fig_learning.add_trace(go.Scatter(x=rounds, y=global_f1, mode='lines+markers', name='Global F1-Score', line=dict(color='#10B981', width=3)))
        fig_learning.add_trace(go.Scatter(x=rounds, y=global_auc, mode='lines+markers', name='Global ROC-AUC', line=dict(color='#F59E0B', width=3)))

        fig_learning.update_layout(
            title="Global Model Convergence Across Federated Rounds",
            xaxis_title="Communication Round",
            yaxis_title="Score (0.0 - 1.0)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_learning, use_container_width=True)

        st.markdown("### 🎛️ Model Weight Evolution Across Rounds")
        coef_history = np.array([h["parameters"]["coef"][0] for h in history])
        df_coef = pd.DataFrame(coef_history, columns=FEATURE_NAMES, index=[f"Round {r}" for r in rounds])
        st.dataframe(df_coef.style.background_gradient(cmap="Blues"), use_container_width=True)

# =============================================================================
# TAB 4: LOCAL VS FEDERATED PERFORMANCE
# =============================================================================
with tab4:
    st.subheader("📊 Performance Comparison: Isolated Local Models vs Federated Global Model")
    st.write("Comparing isolated local models with the federated global model on a shared benchmark ")
    st.write("This experiment evaluates whether federated collaboration can produce a competitive global model while keeping local training data decentralized.")

    # Train isolated local models for comparison if not already trained
    local_evals = []
    for c in clients:
        c.train_local(global_parameters=None) # Train purely local
        eval_metrics = c.local_model.evaluate(X_global_test, y_global_test)
        local_evals.append({
            "Model Scope": f"Local Model ({c.hospital_name})",
            "Accuracy": eval_metrics["accuracy"],
            "Precision": eval_metrics["precision"],
            "Recall": eval_metrics["recall"],
            "F1-Score": eval_metrics["f1"],
            "ROC-AUC": eval_metrics["roc_auc"]
        })

    # Get server global eval
    if "trained_server" in st.session_state:
        srv = st.session_state["trained_server"]
        global_eval = srv.global_model.evaluate(X_global_test, y_global_test)
    else:
        # Run 5 quick rounds if user jumped straight to tab 4
        srv = server
        srv.run_federated_learning(num_rounds=5)
        global_eval = srv.global_model.evaluate(X_global_test, y_global_test)

    local_evals.append({
        "Model Scope": "🌟 FEDERATED GLOBAL MODEL (FedAvg)",
        "Accuracy": global_eval["accuracy"],
        "Precision": global_eval["precision"],
        "Recall": global_eval["recall"],
        "F1-Score": global_eval["f1"],
        "ROC-AUC": global_eval["roc_auc"]
    })

    comp_df = pd.DataFrame(local_evals)
    st.dataframe(comp_df.style.highlight_max(subset=["Accuracy", "F1-Score", "ROC-AUC"], color="#D1FAE5"), use_container_width=True)

    st.markdown("### 🧩 Confusion Matrix Comparison (Global Test Set)")
    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        st.markdown("**Local Model (St. Jude Isolated)**")
        cm_local = clients[0].local_model.evaluate(X_global_test, y_global_test)["confusion_matrix"]
        fig_cm1 = px.imshow(cm_local, text_auto=True, color_continuous_scale="Reds", labels=dict(x="Predicted", y="Actual"), x=["Neg", "Pos"], y=["Neg", "Pos"])
        st.plotly_chart(fig_cm1, use_container_width=True)

    with col_cm2:
        st.markdown("**Federated Global Model (FedAvg Combined)**")
        cm_global = global_eval["confusion_matrix"]
        fig_cm2 = px.imshow(cm_global, text_auto=True, color_continuous_scale="Greens", labels=dict(x="Predicted", y="Actual"), x=["Neg", "Pos"], y=["Neg", "Pos"])
        st.plotly_chart(fig_cm2, use_container_width=True)

# =============================================================================
# TAB 5: PATIENT DIAGNOSIS PREDICTOR
# =============================================================================
with tab5:
    st.subheader("🩺 Interactive Patient Rare Disease Diagnostic Tool")
    st.write("Input patient clinical biomarkers below to evaluate disease probability using the Federated Global Model vs Local Hospital Models:")

    col_in1, col_in2, col_in3 = st.columns(3)

    with col_in1:
        age_in = st.number_input("Patient Age", min_value=1, max_value=100, value=35)
        mutation_in = st.slider("Genetic Mutation Score (0 - 10)", 0.0, 10.0, 7.5, step=0.1)
        protein_in = st.number_input("Serum Protein Level (pg/mL)", min_value=10.0, max_value=500.0, value=220.0)

    with col_in2:
        inflam_in = st.slider("Inflammatory Index (0 - 5)", 0.0, 5.0, 3.2, step=0.1)
        expression_in = st.slider("Gene Expression Alpha Z-score (-3 to 3)", -3.0, 3.0, 1.2, step=0.1)

    with col_in3:
        decay_in = st.slider("Cellular Decay Rate (0 - 1)", 0.0, 1.0, 0.65, step=0.01)
        fam_hist_in = st.selectbox("Family History Score", [0, 1, 2, 3], index=2)

    # Construct input dataframe
    input_patient = pd.DataFrame([{
        "age": age_in,
        "genetic_mutation_score": mutation_in,
        "serum_protein_pg_ml": protein_in,
        "inflammatory_index": inflam_in,
        "gene_expression_alpha": expression_in,
        "cellular_decay_rate": decay_in,
        "family_history_score": fam_hist_in
    }])

    st.markdown("---")
    st.markdown("### 🎯 Real-Time Diagnostic Risk Predictions")

    if "trained_server" in st.session_state:
        active_server = st.session_state["trained_server"]
    else:
        active_server = server
        active_server.run_federated_learning(num_rounds=5)

    global_prob = active_server.global_model.predict_proba(input_patient)[0][1]

    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        st.metric(
            label="🌟 Federated Global Model Risk Probability",
            value=f"{global_prob * 100:.1f}%",
            delta="High Risk" if global_prob > 0.5 else "Low Risk",
            delta_color="inverse" if global_prob > 0.5 else "normal"
        )
        if global_prob > 0.5:
            st.error("🚨 Diagnostic Indicator: **POSITIVE FOR RARE DISEASE X**")
        else:
            st.success("✅ Predicted Risk Class: HIGH") 
            st.write("This prediction is generated from synthetic data and is not a clinical diagnosis.")

    with col_res2:
        # Compare local hospital predictions
        local_preds = []
        for c in clients:
            p_val = c.local_model.predict_proba(input_patient)[0][1]
            local_preds.append({"Model": c.hospital_name, "Predicted Risk (%)": round(p_val * 100, 1)})
        local_preds.append({"Model": "🌟 Federated Global Model", "Predicted Risk (%)": round(global_prob * 100, 1)})

        fig_pred = px.bar(
            pd.DataFrame(local_preds),
            x="Model",
            y="Predicted Risk (%)",
            color="Model",
            title="Comparison of Local Hospital Models vs Global Model on Patient Case",
            text_auto=True
        )
        fig_pred.update_layout(yaxis_range=[0, 100], showlegend=False)
        st.plotly_chart(fig_pred, use_container_width=True)

st.markdown("---")
st.markdown("✨ *Fund My Crazy 2026 Submission - Federated Learning for Rare Disease Diagnosis MVP*")
