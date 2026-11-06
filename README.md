# Machine Learning Assignment 2 - QSAR Biodegradation Classification

## a. Problem Statement
The objective is to build and compare five supervised classification algorithms for predicting whether a chemical is **Ready Biodegradable (RB)** or **Not Ready Biodegradable (NRB)** from molecular descriptors. The same dataset and train/test split are used for every model. The trained models are exposed through a Streamlit application that evaluates uploaded test data without retraining.

## b. Dataset Description
**Dataset:** QSAR Biodegradation  
**Source:** UCI Machine Learning Repository  
**Problem type:** Binary classification  
**Instances:** 1,055  
**Input features:** 41 molecular descriptors  
**Classes:** 356 RB and 699 NRB  
**Missing values:** None in the supplied dataset  

For this implementation, RB is encoded as `1` (positive class) and NRB as `0`. A stratified 80/20 train/test split with `random_state=42` is used, producing 844 training rows and 211 test rows. Standardization is included inside the saved pipelines for Logistic Regression, KNN and SVM.

Dataset page: https://archive.ics.uci.edu/dataset/254/qsar+biodegradation

## c. GitHub Repository Link
**Replace before submission:** `YOUR_GITHUB_REPOSITORY_URL`

## Live Streamlit Application
**Replace before submission:** `YOUR_STREAMLIT_APP_URL`

## d. Models Used and Evaluation Metrics
The following five models are implemented on the same dataset:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Model Comparison
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.8626** | 0.9136 | **0.8000** | 0.7887 | **0.7943** | **0.6912** |
| Decision Tree | 0.8152 | 0.7844 | 0.7424 | 0.6901 | 0.7153 | 0.5796 |
| KNN | 0.8531 | 0.8973 | 0.7703 | 0.8028 | 0.7862 | 0.6747 |
| Naive Bayes | 0.6967 | 0.8650 | 0.5276 | **0.9437** | 0.6768 | 0.4972 |
| SVM | 0.8483 | **0.9189** | 0.7910 | 0.7465 | 0.7681 | 0.6562 |

### Observations About Model Performance
| ML Model | Observation |
|---|---|
| Logistic Regression | Best overall balance. It achieved the highest accuracy (0.8626), precision (0.8000), F1 score (0.7943), and MCC (0.6912), while also obtaining a strong AUC of 0.9136. |
| Decision Tree | Produced interpretable nonlinear decisions but had the weakest AUC (0.7844) and lower recall/F1 than LR and KNN, suggesting poorer generalization on this split. |
| KNN | Very competitive: second-highest accuracy (0.8531) and F1 (0.7862), with recall (0.8028) slightly higher than LR. Scaling is important because KNN is distance-based. |
| Naive Bayes | Achieved the highest recall (0.9437), so it identified most RB cases, but precision (0.5276) and accuracy (0.6967) were much lower. This indicates many false-positive RB predictions. |
| SVM | Achieved the highest AUC (0.9189), showing excellent ranking/separation ability. Its accuracy/F1 were slightly below LR and KNN on the selected threshold and split. |
| **Overall Winner** | **Logistic Regression**, because it leads four of the six requested metrics (Accuracy, Precision, F1 and MCC) and remains close to the best AUC/Recall results. |

### Metric Interpretation
- **Accuracy:** proportion of all predictions that are correct.
- **AUC:** ability to rank/separate RB from NRB across classification thresholds.
- **Precision:** among predicted RB cases, the proportion actually RB.
- **Recall:** among actual RB cases, the proportion correctly identified.
- **F1:** harmonic mean of precision and recall.
- **MCC:** balanced correlation measure using all four confusion-matrix outcomes; useful when classes are imbalanced.

## Streamlit Application Features
The application provides:
- Test CSV upload
- Dropdown for selecting any of the five saved models
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Test-data preview

The application loads the persisted model selected by the user and **does not retrain** it using uploaded test data.

## Repository Structure
```text
QSAR_Biodegradation_ML_Assignment2/
|-- app.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- biodeg.csv
|-- test_data.csv
|-- model_metrics.csv
|-- metadata.json
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   `-- svm.joblib
`-- SUBMISSION_CHECKLIST.txt
```

## How to Run on BITS Virtual Lab
```bash
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```
Open the Streamlit URL shown in the terminal, upload `test_data.csv`, select each model, and verify the displayed results. Capture one screenshot on the BITS Virtual Lab showing successful assignment execution for the final PDF.

## Streamlit Community Cloud Deployment
1. Push all repository files to your GitHub repository.
2. Sign in to Streamlit Community Cloud using GitHub.
3. Create a new app and select the repository and branch.
4. Set the application file to `app.py`.
5. Deploy and verify that `test_data.csv` works with all five models.
6. Copy the public app URL into this README and the final submission PDF.

## Reproducibility
The supplied `train_models.py` regenerates the persisted models, `test_data.csv`, and `model_metrics.csv` using the fixed split (`test_size=0.20`, `random_state=42`, stratified by class). Results can vary if package versions, hyperparameters or the split are changed.

## BITS Virtual Lab Screenshot
**Insert your own BITS Virtual Lab execution screenshot in the final submitted PDF.**
