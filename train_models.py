"""Train and save the five classifiers required for ML Assignment 2."""
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent
FEATURES = [f"feature_{i:02d}" for i in range(1, 42)]
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

def load_data():
    df = pd.read_csv(ROOT / "biodeg.csv", sep=";", header=None, names=FEATURES + ["class"])
    df["target"] = (df["class"].str.strip() == "RB").astype(int)
    return df

def main():
    df = load_data()
    X, y = df[FEATURES], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=42))]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]),
        "Naive Bayes": GaussianNB(),
        "SVM": Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="rbf", probability=True, random_state=42))]),
    }
    filenames = {
        "Logistic Regression": "logistic_regression.joblib", "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib", "Naive Bayes": "naive_bayes.joblib", "SVM": "svm.joblib"
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        results.append({
            "ML Model Name": name, "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, prob), "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred), "F1": f1_score(y_test, pred),
            "MCC": matthews_corrcoef(y_test, pred),
        })
        joblib.dump(model, MODEL_DIR / filenames[name])

    pd.DataFrame(results).to_csv(ROOT / "model_metrics.csv", index=False, float_format="%.4f")
    test = X_test.copy(); test["target"] = y_test.values; test["class"] = test["target"].map({1:"RB", 0:"NRB"})
    test.to_csv(ROOT / "test_data.csv", index=False)
    metadata = {"feature_columns": FEATURES, "model_files": filenames, "positive_class":"RB", "negative_class":"NRB", "random_state":42, "test_size":0.20}
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(results).round(4).to_string(index=False))

if __name__ == "__main__":
    main()
