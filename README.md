# Medical Risk Prediction System using Machine Learning

## Overview

This repository contains a complete endâ€‘toâ€‘end machine learning project that
predicts whether a patient is at risk of heart disease using commonly
available clinical measurements.  The project is built with Python and
scikitâ€‘learn and demonstrates data loading, preprocessing, model training,
evaluation, inference and a simple web demo.  It is intended for
educational use only and must **not** be presented as a real medical
diagnostic tool.

## Problem Statement

Cardiovascular disease remains one of the leading causes of death worldwide.
Early identification of patients at risk can help clinicians prioritise
treatment and lifestyle interventions.  This project frames the problem as
a binary classification task where the goal is to predict the presence
(`1`) or absence (`0`) of heart disease given a set of patient features.

## Why This Project?

An endâ€‘toâ€‘end machine learning pipeline is a great way to showcase
fundamental data science skills.  This project includes data loading,
preprocessing, model selection, evaluation, persistence of the best model
and a Streamlit demo app.  It is designed to be understandable to junior
students yet sufficiently robust for inclusion in a GitHub portfolio or
discussion in an internship interview.

## Dataset

The project expects a CSV file named `heart_disease.csv` to be placed at
`data/raw/heart_disease.csv`.  This file should contain a tabular heart
disease dataset with a binary target column indicating whether heart
disease is present.  Several public datasets follow this format, including
the UCI Heart Disease data.  The UCI dataset uses 14 attributes such as
age, sex, chest pain type, resting blood pressure, serum cholesterol,
fasting blood sugar, resting electrocardiographic results, maximum heart
rate achieved, exerciseâ€‘induced angina, ST depression (`oldpeak`), slope of
the peak exercise ST segment, number of vessels coloured by fluoroscopy
(`ca`), and the `thal` variable.  The target column
is sometimes called `num` or `target`; our code automatically detects
common names.

**Important:** This dataset is for educational use only.  It does not
contain personally identifiable information, and the trained model **must
not** be used for real medical decisions.

### Expected File Layout

Place your raw CSV file here:

```
medical-risk-prediction-ml/
â””â”€â”€ data/
    â””â”€â”€ raw/
        â””â”€â”€ heart_disease.csv  â† put your dataset here
```

If the file is missing, running the training script will print a clear
message telling you where to place it.

## Project Pipeline

The project follows a simple machine learning workflow:

1. **Data Loading:** Read the CSV file and detect the target column.
2. **Preprocessing:** Impute missing values, scale numeric features and
   oneâ€‘hot encode categorical features using a `ColumnTransformer`.
3. **Model Training:** Train and compare several classical models
   (Logistic Regression, Random Forest, Gradient Boosting) on the
   preprocessed data.
4. **Evaluation:** Compute metrics such as accuracy, precision, recall,
   F1â€‘score and ROCâ€‘AUC.  Plot a confusion matrix and a model comparison
   bar chart.
5. **Model Saving:** Persist the best model and its evaluation metrics
   under the `models/` directory.
6. **Streamlit Demo:** Provide a simple user interface for entering
   feature values and obtaining a risk prediction.

## Technologies Used

* Python 3.10+
* pandas
* numpy
* scikitâ€‘learn
* matplotlib
* seaborn
* joblib
* Streamlit
* pytest

## Repository Structure

The repository uses a clear and modular layout:

```
medical-risk-prediction-ml/
â”œâ”€â”€ README.md                 â† this file
â”œâ”€â”€ requirements.txt          â† Python dependencies
â”œâ”€â”€ .gitignore                â† ignores large/temporary files
â”œâ”€â”€ LICENSE                   â† MIT license
â”‚
â”œâ”€â”€ app.py                    â† Streamlit demo
â”œâ”€â”€ train_model.py            â† training entry point
â”œâ”€â”€ predict.py                â† simple commandâ€‘line inference
â”‚
â”œâ”€â”€ data/                     â† data storage
â”‚   â”œâ”€â”€ README.md             â† dataset instructions
â”‚   â”œâ”€â”€ raw/                  â† raw data (CSV goes here)
â”‚   â”‚   â””â”€â”€ .gitkeep
â”‚   â””â”€â”€ processed/            â† optional processed data
â”‚       â””â”€â”€ .gitkeep
â”‚
â”œâ”€â”€ src/                      â† reusable Python modules
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ config.py             â† centralised paths
â”‚   â”œâ”€â”€ data_loader.py        â† loading & validation
â”‚   â”œâ”€â”€ preprocessing.py      â† preprocessing pipeline
â”‚   â”œâ”€â”€ model_training.py     â† model selection & training
â”‚   â”œâ”€â”€ evaluation.py         â† metrics & plotting
â”‚   â””â”€â”€ inference.py          â† model loading & prediction
â”‚
â”œâ”€â”€ models/                   â† trained models & metrics
â”‚   â””â”€â”€ .gitkeep
â”‚
â”œâ”€â”€ reports/                  â† reports and figures
â”‚   â”œâ”€â”€ project_report.md     â† universityâ€‘style report
â”‚   â””â”€â”€ figures/
â”‚       â””â”€â”€ .gitkeep
â”‚
â”œâ”€â”€ tests/                    â† lightweight unit tests
â”‚   â”œâ”€â”€ test_preprocessing.py
â”‚   â””â”€â”€ test_inference.py
â”‚
â””â”€â”€ assets/                   â† assets for README/demo
    â””â”€â”€ .gitkeep
```

## Models Used

Three classical machine learning models are compared:

1. **Logistic Regression** â€“ a linear model suitable for baseline
   classification tasks.
2. **Random Forest Classifier** â€“ an ensemble of decision trees that
   handles nonâ€‘linear relationships and interactions.
3. **Gradient Boosting Classifier** â€“ a boosting ensemble that often
   provides strong performance on tabular data.

## Evaluation Metrics

To assess model performance we compute the following metrics on the
validation set:

| Metric        | Description |
|---------------|-------------|
| **Accuracy**  | Proportion of correct predictions |
| **Precision** | Proportion of positive predictions that are correct |
| **Recall**    | Proportion of actual positives that are correctly identified |
| **F1â€‘score**  | Harmonic mean of precision and recall |
| **ROCâ€‘AUC**   | Area under the ROC curve, measuring the tradeâ€‘off between true positive rate and false positive rate |

In medical risk prediction recall is particularly important, because
incorrectly labelling a highâ€‘risk patient as low risk (a false negative)
could have serious consequences.  A high recall ensures that most
highâ€‘risk patients are detected, even if it comes at the cost of some false
positives.

## How to Run

To reproduce the project locally:

1. **Clone the repository:**

   ```bash
   git clone <repoâ€‘url>
   cd medical-risk-prediction-ml
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the dataset:**  Place your `heart_disease.csv` file in
   `data/raw/`.  The file should contain the target column (`target` or
   similar) and feature columns.  If you do not have a dataset yet you can
   download the UCI Heart Disease dataset or any other public heart
   disease dataset with similar columns .

5. **Train the model:**

   ```bash
   python train_model.py
   ```

   If the dataset is missing the script will stop and print a clear
   instruction telling you where to place the CSV file.

6. **Run the demo app:**

   ```bash
   streamlit run app.py
   ```

7. **Run the tests:**

   ```bash
   pytest
   ```

## Model Results

After training the model on the heart disease dataset, the best model achieved the following evaluation results:

| Metric | Score |
|---|---:|
| Accuracy | 0.8361 |
| Precision | 0.7805 |
| Recall | 0.9697 |
| F1-score | 0.8649 |
| ROC-AUC | 0.9091 |

The high recall score is important in a medical risk prediction task because it means the model is able to identify most high-risk cases.

> Note: This project is for educational and portfolio purposes only. It is not intended for real medical diagnosis.

## Demo

The project includes a Streamlit web app for entering patient information and receiving a heart disease risk prediction.

![Streamlit Demo](assets/demo_screenshot.png)

## Interview Explanation

When discussing this project during an internship interview you can say
something like:

> â€œThis project is an endâ€‘toâ€‘end classical machine learning system for
> predicting heart disease risk from tabular clinical data.  I handled
> data loading, preprocessing, model training, evaluation, model saving
> and built a Streamlit demo app.  I compared multiple models and
> selected the best one based on F1â€‘score and ROCâ€‘AUC, because
> accuracy alone can be misleading in medical risk prediction.â€

## Limitations

This project is for educational purposes only.  It uses public datasets
that may be small, imbalanced or outdated, and the model has not been
validated on external cohorts.  As such it should **never** be used for
real clinical decision making.  Real deployment would require extensive
data, rigorous validation, hyperparameter tuning and model explainability.

## Future Improvements

Possible future enhancements include:

* Collecting more data and addressing class imbalance
* Performing proper hyperparameter tuning
* Adding explainability tools such as SHAP values
* Deploying the model as an API
* Improving the Streamlit UI

