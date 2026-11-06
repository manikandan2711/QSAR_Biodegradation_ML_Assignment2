
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent

META = json.loads((ROOT / "metadata.json").read_text())
FEATURES = META["feature_columns"]
MODEL_FILES = META["model_files"]

st.set_page_config(
    page_title="QSAR Biodegradation Classifier",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 QSAR Biodegradation - Classification Model Evaluation")

st.caption(
    "Evaluate five pre-trained classifiers on QSAR test data. "
    "Models are loaded from disk and are not retrained on uploaded data."
)

# Sidebar controls
with st.sidebar:
    st.header("Evaluation Controls")

    model_name = st.selectbox(
        "Select model",
        list(MODEL_FILES.keys())
    )

    uploaded = st.file_uploader(
        "Upload test CSV",
        type=["csv"]
    )

    st.info(
        "Upload a CSV containing 41 molecular descriptor features. "
        "Actual target labels are optional."
    )

# Dataset information
st.subheader("Dataset and Target")

st.write(
    "Target: **RB (1) = Ready Biodegradable**, "
    "**NRB (0) = Not Ready Biodegradable**."
)

st.write(
    "The project uses 41 molecular-descriptor input features "
    "and a fixed 80/20 stratified train/test split (random_state=42)."
)

if uploaded is None:
    st.warning("Upload a test CSV file to evaluate a saved model.")
    st.stop()

try:
    # Read uploaded CSV
    data = pd.read_csv(uploaded)

    # Validate feature columns
    missing = [c for c in FEATURES if c not in data.columns]

    if missing:
        st.error(
            f"Missing required feature columns: {', '.join(missing[:8])}"
            + (" ..." if len(missing) > 8 else "")
        )
        st.stop()

    if data.empty:
        st.error("Uploaded CSV contains no records.")
        st.stop()

    # Extract feature columns in the correct order
    X = data[FEATURES].copy()

    # Validate numeric feature values
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        st.error("Feature columns contain missing or non-numeric values.")
        st.stop()

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        st.error("Feature columns contain infinite values.")
        st.stop()

    # Load selected pre-trained model
    model = joblib.load(ROOT / "model" / MODEL_FILES[model_name])

    # Generate predictions
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    st.subheader(f"Results - {model_name}")

    # Prediction summary
    total = len(pred)
    rb_count = int((pred == 1).sum())
    nrb_count = int((pred == 0).sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", total)
    col2.metric("Predicted RB", rb_count)
    col3.metric("Predicted NRB", nrb_count)

    # Prediction output
    predictions = pd.DataFrame({
        "Predicted Class": pred,
        "Predicted Label": np.where(pred == 1, "RB", "NRB"),
        "RB Probability (%)": np.round(prob * 100, 2),
    })

    st.subheader("Prediction Results")
    st.dataframe(predictions.head(20), use_container_width=True)

    # Download predictions
    download_data = pd.concat(
        [data.reset_index(drop=True), predictions],
        axis=1
    )

    st.download_button(
        label="Download Predictions CSV",
        data=download_data.to_csv(index=False).encode("utf-8"),
        file_name="qsar_predictions.csv",
        mime="text/csv",
    )

    # Check whether actual labels are available
    if "target" in data.columns:
        y = pd.to_numeric(data["target"], errors="coerce")

    elif "class" in data.columns:
        y = data["class"].astype(str).str.strip().map({
            "RB": 1,
            "NRB": 0
        })

    else:
        y = None

    if y is not None:

        if y.isna().any() or not y.isin([0, 1]).all():
            st.error("Actual target labels must contain only 0/1 or RB/NRB.")
            st.stop()

        y = y.astype(int)

        if y.nunique() < 2:
            st.warning(
                "Only one actual class is present. "
                "AUC cannot be calculated, but other classification metrics "
                "can still be evaluated."
            )

        # Calculate evaluation metrics
        accuracy = accuracy_score(y, pred)
        auc = roc_auc_score(y, prob) if y.nunique() == 2 else None
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        mcc = matthews_corrcoef(y, pred)

        st.subheader("Evaluation Metrics")

        cols = st.columns(6)

        metrics = {
            "Accuracy": accuracy,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC": mcc,
        }

        for col, (label, value) in zip(cols, metrics.items()):
            col.metric(
                label,
                f"{value:.4f}" if value is not None else "N/A"
            )

        # Confusion matrix
        left, right = st.columns(2)

        with left:
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y, pred, labels=[0, 1])

            fig, ax = plt.subplots(figsize=(5, 4))

            ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["NRB", "RB"]
            ).plot(
                ax=ax,
                cmap="Blues",
                colorbar=False
            )

            st.pyplot(fig)
            plt.close(fig)

        # Classification report
        with right:
            st.subheader("Classification Report")

            report = classification_report(
                y,
                pred,
                labels=[0, 1],
                target_names=["NRB", "RB"],
                output_dict=True,
                zero_division=0
            )

            st.dataframe(
                pd.DataFrame(report).transpose().round(4),
                use_container_width=True
            )

        st.success(
            "Evaluation completed using the actual labels in the uploaded CSV."
        )

    else:
        st.info(
            "No actual target labels found. Predictions were generated "
            "successfully, but evaluation metrics and confusion matrix "
            "cannot be calculated without ground-truth labels."
        )

except Exception as exc:
    st.error(f"Evaluation failed: {exc}")
