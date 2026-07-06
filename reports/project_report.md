# Medical Risk Prediction System using Machine Learning

## Abstract

This project presents a classical machine learning system for predicting
the risk of heart disease based on tabular clinical data.  The system
loads a heart disease dataset, preprocesses the features, trains and
compares multiple models, evaluates their performance and exposes the
selected model via a simple web application.  The goal is to provide
students with an end‑to‑end example of a supervised learning pipeline.

## Introduction

Heart disease remains one of the leading causes of morbidity and
mortality worldwide.  Data‑driven methods can assist clinicians by
highlighting high‑risk patients.  This project explores whether
classical machine learning algorithms can effectively classify patients
into high‑risk or low‑risk categories based on routinely collected
features.  The project is intended for educational purposes and does
not constitute medical advice.

## Dataset

The dataset used in this project is the publicly available UCI Heart
Disease dataset.  It contains 303 instances with 14 key attributes
including age, sex, chest pain type, resting blood pressure, serum
cholesterol, fasting blood sugar, resting electrocardiographic results,
maximum heart rate achieved, exercise‑induced angina, ST depression
(`oldpeak`), slope, number of major vessels (`ca`), thalassemia and a
binary target indicating the presence of heart disease【505570677987423†L90-L130】.

## Methodology

1. **Data Loading:**  The CSV file is read into a pandas `DataFrame`.
   A helper function detects the target column based on common names such
   as `target` or `HeartDisease`.  Basic validation ensures the dataset
   contains at least one feature and a target.

2. **Preprocessing:**  Numeric features are imputed using the median and
   scaled with a `StandardScaler`.  Categorical features are imputed
   using the most frequent value and one‑hot encoded.  These steps are
   combined in a `ColumnTransformer` to avoid data leakage.

3. **Model Training:**  Three baseline models are compared: Logistic
   Regression, Random Forest and Gradient Boosting.  Each model is
   combined with the preprocessing pipeline in a single scikit‑learn
   `Pipeline`.  The data is split into training and test sets using a
   stratified 80/20 split.

4. **Evaluation:**  Models are evaluated on the test set using accuracy,
   precision, recall, F1‑score and ROC‑AUC.  The best model is selected
   primarily by F1‑score because recall is crucial in medical risk
   prediction, with ROC‑AUC used as a secondary criterion.

5. **Persistence and Demo:**  The selected model is serialised with
   `joblib`.  A Streamlit application provides an interactive interface
   for entering feature values and obtaining a risk prediction.

## Preprocessing

Preprocessing is a critical step to prepare heterogeneous clinical
features for model consumption.  Missing numeric values are replaced by
the median of each feature; this is robust to outliers.  Numeric
features are then standardised to zero mean and unit variance.  Categorical
variables are encoded into binary indicator columns via one‑hot
encoding.  The `ColumnTransformer` ensures that these operations are
applied only to their respective column subsets.

## Models

Three classical algorithms were evaluated:

1. **Logistic Regression:**  A linear classifier that models the log
   odds of the positive class.  It is fast and provides probabilities
   but may underfit complex relationships.
2. **Random Forest:**  An ensemble of decision trees trained on
   bootstrap samples.  It captures non‑linearities and interactions but
   can be less interpretable.
3. **Gradient Boosting:**  Sequentially adds trees to correct errors of
   previous trees.  It often achieves strong performance on tabular
   data with limited tuning.

## Evaluation Metrics

Evaluation metrics quantify how well each model performs on unseen data.

| Metric        | Description |
|---------------|-------------|
| **Accuracy**  | Fraction of correct predictions |
| **Precision** | Fraction of predicted positives that are true positives |
| **Recall**    | Fraction of true positives detected (sensitivity) |
| **F1‑score**  | Harmonic mean of precision and recall |
| **ROC‑AUC**   | Area under the ROC curve, summarising the trade‑off between sensitivity and specificity |

In the context of heart disease prediction, recall is emphasised because
a false negative (failing to identify a high‑risk patient) could have
more severe consequences than a false positive.  F1‑score balances
precision and recall, making it a suitable primary metric.

## Results

Because the project is intended as a template and the raw dataset may not
be available at runtime, placeholder results are shown below.  After
training on the real dataset you should replace these values with
actual numbers.

| Model                | F1‑score | ROC‑AUC |
|----------------------|---------|--------|
| Logistic Regression  | –       | –      |
| Random Forest        | –       | –      |
| Gradient Boosting    | –       | –      |

## Limitations

This project uses a relatively small dataset and simple models.  No
hyperparameter tuning or cross‑validation is performed.  The model
performance may therefore be sub‑optimal.  Additionally, the model
has not been validated on external populations; it must not be used in
clinical practice.  The feature set is limited to variables available in
the UCI dataset and may omit other important risk factors.

## Future Work

Future improvements could include:

* Collecting more diverse and up‑to‑date data and addressing class
  imbalance
* Performing systematic hyperparameter optimisation
* Incorporating explainable AI methods such as SHAP values to
  understand feature importance
* Deploying the model as a REST API or web service
* Enhancing the Streamlit app with better visualisations and input
  validation

## Conclusion

This project demonstrates an end‑to‑end classical machine learning
pipeline for predicting heart disease risk from tabular clinical data.
The modular structure separates concerns such as data loading,
preprocessing, training, evaluation and inference, making the codebase
easy to understand and extend.  While the system is educational and
should not be used clinically, it provides a solid foundation for
students and early‑career practitioners to build upon.