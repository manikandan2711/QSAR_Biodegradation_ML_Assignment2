from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / "metadata.json").read_text())
FEATURES = META["feature_columns"]
MODEL_FILES = META["model_files"]

st.set_page_config(page_title="QSAR Biodegradation Classifier", page_icon="🧪", layout="wide")
st.title("🧪 QSAR Biodegradation - Classification Model Evaluation")
st.caption("Evaluate five pre-trained classifiers on QSAR test data. Models are loaded from disk and are not retrained on uploaded data.")

with st.sidebar:
    st.header("Evaluation Controls")
    model_name = st.selectbox("Select model", list(MODEL_FILES.keys()))
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    st.info("Use the included test_data.csv. It contains 41 input features plus target/class columns.")

st.subheader("Dataset and Target")
st.write("Target: **RB (1) = Ready Biodegradable**, **NRB (0) = Not Ready Biodegradable**.")
st.write("The project uses 41 molecular-descriptor input features and a fixed 80/20 stratified train/test split (random_state=42).")

if uploaded is None:
    st.warning("Upload test_data.csv from the repository to evaluate a saved model.")
    st.stop()

try:
    data = pd.read_csv(uploaded)
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        st.error(f"Missing required feature columns: {', '.join(missing[:8])}" + (" ..." if len(missing)>8 else ""))
        st.stop()
    if "target" in data.columns:
        y = data["target"].astype(int)
    elif "class" in data.columns:
        y = data["class"].astype(str).str.strip().map({"RB":1, "NRB":0})
        if y.isna().any(): raise ValueError("class must contain only RB or NRB")
    else:
        st.error("Uploaded CSV must contain target or class column."); st.stop()
    X = data[FEATURES]
    model = joblib.load(ROOT / "model" / MODEL_FILES[model_name])
    pred = model.predict(X)
    prob = model.predict_proba(X)[:,1]
    metrics = {
        "Accuracy": accuracy_score(y,pred), "AUC": roc_auc_score(y,prob),
        "Precision": precision_score(y,pred,zero_division=0), "Recall": recall_score(y,pred,zero_division=0),
        "F1 Score": f1_score(y,pred,zero_division=0), "MCC": matthews_corrcoef(y,pred)
    }
    st.subheader(f"Results - {model_name}")
    cols = st.columns(6)
    for col,(label,val) in zip(cols,metrics.items()): col.metric(label,f"{val:.4f}")
    left,right=st.columns(2)
    with left:
        st.subheader("Confusion Matrix")
        fig,ax=plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay(confusion_matrix(y,pred),display_labels=["NRB","RB"]).plot(ax=ax,cmap="Blues",colorbar=False); st.pyplot(fig)
    with right:
        st.subheader("Classification Report")
        report=classification_report(y,pred,target_names=["NRB","RB"],output_dict=True,zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().round(4),use_container_width=True)
    st.subheader("Uploaded Test Data Preview")
    st.dataframe(data.head(10),use_container_width=True)
except Exception as exc:
    st.error(f"Evaluation failed: {exc}")
