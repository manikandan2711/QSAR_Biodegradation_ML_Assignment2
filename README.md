\# QSAR Biodegradation Classification Using Machine Learning



\*\*BITS Pilani – Work Integrated Learning Programmes\*\*  

\*\*M.Tech in Artificial Intelligence and Machine Learning\*\*  

\*\*Machine Learning – Assignment 2\*\*



\## 1. Problem Statement



The objective of this project is to build and compare five supervised machine learning classification algorithms to predict whether a chemical compound is \*\*Ready Biodegradable (RB)\*\* or \*\*Not Ready Biodegradable (NRB)\*\* based on its molecular descriptors.



This is a binary classification problem. The project evaluates five classification algorithms using six standard performance metrics. A Streamlit web application is developed to allow users to evaluate labelled test datasets and generate predictions for datasets containing only input features.



The application uses previously trained and saved models without retraining them during prediction or evaluation.



\## 2. Dataset Description



| Attribute | Description |

|---|---|

| Dataset | QSAR Biodegradation |

| Source | UCI Machine Learning Repository |

| Problem Type | Binary Classification |

| Total Instances | 1,055 |

| Input Features | 41 molecular descriptors |

| Classes | RB and NRB |

| Missing Values | None in the supplied dataset |



\*\*Class distribution:\*\*

\- Ready Biodegradable (RB): 356

\- Not Ready Biodegradable (NRB): 699



The target variable is encoded as:

\- `1` – Ready Biodegradable (RB)

\- `0` – Not Ready Biodegradable (NRB)



A stratified 80:20 train-test split is performed using `random\_state=42`.



\- Training dataset: 844 instances

\- Testing dataset: 211 instances



Stratification maintains the class distribution across the training and testing datasets.



StandardScaler is incorporated into the saved pipelines for Logistic Regression, KNN and SVM. This ensures consistent preprocessing during training and inference.



\*\*Dataset source:\*\*  

https://archive.ics.uci.edu/dataset/254/qsar+biodegradation



\## 3. Project Links



\*\*GitHub Repository:\*\*  

https://github.com/manikandan2711/QSAR\_Biodegradation\_ML\_Assignment2



\*\*Live Streamlit Application:\*\*  

https://qsarbiodegradationmlassignment2-4uxkwa7blhwx4bfyczqjvb.streamlit.app/



\## 4. Machine Learning Models



The following five supervised classification algorithms are implemented:



1\. Logistic Regression

2\. Decision Tree Classifier

3\. K-Nearest Neighbors (KNN)

4\. Gaussian Naive Bayes

5\. Support Vector Machine (SVM)



All models are trained and evaluated using the same stratified training and testing partitions to ensure a consistent comparison.



\## 5. Model Evaluation and Comparison



The models are evaluated using six mandatory classification metrics.



| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |

|---|---:|---:|---:|---:|---:|---:|

| Logistic Regression | 0.8626 | 0.9136 | 0.8000 | 0.7887 | 0.7943 | 0.6912 |

| Decision Tree | 0.8152 | 0.7844 | 0.7424 | 0.6901 | 0.7153 | 0.5796 |

| KNN | 0.8531 | 0.8973 | 0.7703 | 0.8028 | 0.7862 | 0.6747 |

| Naive Bayes | 0.6967 | 0.8650 | 0.5276 | 0.9437 | 0.6768 | 0.4972 |

| SVM | 0.8483 | 0.9189 | 0.7910 | 0.7465 | 0.7681 | 0.6562 |



\### 5.1 Performance Observations



\*\*Logistic Regression:\*\* Achieved the highest accuracy (0.8626), precision (0.8000), F1 score (0.7943) and MCC (0.6912). It demonstrated the strongest overall balance among the evaluated metrics.



\*\*Decision Tree:\*\* Provided interpretable nonlinear decision rules but achieved the lowest AUC (0.7844), indicating comparatively weaker class discrimination on the selected test split.



\*\*K-Nearest Neighbors:\*\* Achieved the second-highest accuracy (0.8531) and F1 score (0.7862). Its recall of 0.8028 was higher than Logistic Regression. Feature scaling is important because KNN relies on distance calculations.



\*\*Gaussian Naive Bayes:\*\* Achieved the highest recall (0.9437), identifying a large proportion of actual RB compounds. However, its precision (0.5276) and accuracy (0.6967) were lower, indicating a higher number of false-positive predictions.



\*\*Support Vector Machine:\*\* Achieved the highest AUC (0.9189), demonstrating strong discrimination between the two classes across classification thresholds. It used an RBF kernel with probability estimation enabled.



\### 5.2 Overall Model Assessment



Logistic Regression is the strongest overall model on this test split because it leads in four of the six evaluation metrics: Accuracy, Precision, F1 and MCC.



SVM achieved the highest AUC, while Naive Bayes achieved the highest Recall. The most suitable model ultimately depends on the evaluation objective and the relative importance of false-positive and false-negative predictions.



\## 6. Evaluation Metrics – Definitions



\- \*\*Accuracy:\*\* Proportion of all predictions that are correct.

\- \*\*AUC:\*\* Area under the ROC curve, measuring the model's ability to distinguish between the two classes across different thresholds.

\- \*\*Precision:\*\* Proportion of predicted positive cases that are actually positive.

\- \*\*Recall:\*\* Proportion of actual positive cases correctly identified.

\- \*\*F1 Score:\*\* Harmonic mean of precision and recall.

\- \*\*MCC:\*\* Matthews Correlation Coefficient, a balanced measure of classification quality based on all four confusion matrix outcomes. Its value ranges from -1 to +1.



For this project, RB is treated as the positive class.



\## 7. Streamlit Application



The project includes an interactive web application developed using Streamlit.



\*\*Live application:\*\*  

https://qsarbiodegradationmlassignment2-4uxkwa7blhwx4bfyczqjvb.streamlit.app/



\### Application Features



\- CSV file upload.

\- Dropdown selection of any of the five trained models.

\- Support for labelled test datasets.

\- Support for features-only datasets.

\- Display of Accuracy, AUC, Precision, Recall, F1 and MCC when actual labels are available.

\- Confusion matrix and classification report for labelled datasets.

\- Predicted class and predicted label (RB/NRB) for features-only datasets.

\- RB prediction probability.

\- Downloadable prediction results.



\### Input Dataset Scenarios



\*\*Scenario 1 – Labelled CSV\*\*



When the uploaded CSV contains actual target labels (`target` or `class`), the application generates predictions and evaluates the selected model using the six performance metrics, confusion matrix and classification report.



\*\*Scenario 2 – Features-only CSV\*\*



When the uploaded CSV contains only the 41 molecular descriptor features, the application generates predictions, predicted labels and RB probabilities. Evaluation metrics are not displayed because actual target values are unavailable.



The application loads the selected pre-trained model from the repository. It does not retrain the model using uploaded test data.



\## 8. GitHub Repository Structure



```text

QSAR\_Biodegradation\_ML\_Assignment2/

|

|-- app.py

|-- train\_models.py

|-- requirements.txt

|-- README.md

|-- biodeg.csv

|-- test\_data.csv

|-- model\_metrics.csv

|-- metadata.json

|

|-- model/

|   |-- logistic\_regression.joblib

|   |-- decision\_tree.joblib

|   |-- knn.joblib

|   |-- naive\_bayes.joblib

|   |-- svm.joblib

|

`-- SUBMISSION\_CHECKLIST.txt

```



\### Important Files



\- `app.py`: Streamlit application for model selection, evaluation and prediction.

\- `train\_models.py`: Script for preprocessing, training, evaluating and saving the five models.

\- `biodeg.csv`: Original QSAR Biodegradation dataset.

\- `test\_data.csv`: Held-out test dataset containing features and actual labels.

\- `model\_metrics.csv`: Model performance comparison results.

\- `metadata.json`: Project metadata and model information.

\- `model/`: Directory containing the five saved machine learning models.

\- `requirements.txt`: Python dependencies required to run the project.



\## 9. Installation and Execution



Clone the GitHub repository and navigate to the project directory.



Install the dependencies:



```bash

python -m pip install -r requirements.txt

```



To regenerate the models and evaluation results:



```bash

python train\_models.py

```



To launch the Streamlit application:



```bash

streamlit run app.py

```



Open the local URL displayed in the terminal to access the application.



Upload `test\_data.csv` to evaluate the models. For features-only prediction, upload a CSV containing the required molecular descriptor columns without the target label.



\## 10. Streamlit Community Cloud Deployment



The application is deployed using Streamlit Community Cloud and connected to the GitHub repository.



Deployment steps:



1\. Push the project files to GitHub.

2\. Sign in to Streamlit Community Cloud.

3\. Create a new application and select the GitHub repository.

4\. Select the `main` branch and `app.py` as the application entry point.

5\. Deploy and verify the application.

6\. Test both labelled and features-only CSV uploads.



The deployed application is publicly accessible using the Streamlit link provided in Section 3.



\## 11. Reproducibility



The `train\_models.py` script regenerates the saved model artifacts, `test\_data.csv` and `model\_metrics.csv` using a stratified 80:20 train-test split with `random\_state=42`.



The same split is used for all five algorithms to support a fair performance comparison.



The Streamlit application uses the persisted model artifacts for inference and evaluation without retraining on uploaded test data.



Results may vary if the dataset, preprocessing, hyperparameters, random seed or package versions are modified.



\## 12. Conclusion



This project demonstrates the implementation and comparison of five supervised machine learning algorithms for QSAR biodegradation classification.



Among the evaluated models, Logistic Regression demonstrated the strongest overall balance across accuracy, precision, F1 score and MCC. SVM achieved the highest AUC, while Gaussian Naive Bayes achieved the highest recall.



The project also demonstrates deployment of saved machine learning models through a Streamlit application, supporting both labelled dataset evaluation and features-only prediction.



The implementation provides a reproducible workflow covering dataset preprocessing, model training, performance evaluation, model persistence and application deployment.



\## 13. BITS Virtual Lab Execution Evidence



The required BITS Virtual Lab execution screenshot is included in the final assignment submission PDF as evidence of the lab activity.





