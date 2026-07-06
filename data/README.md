# Dataset Instructions

This project expects a single CSV file named `heart_disease.csv` in the
`data/raw/` directory.  The file should contain a binary target column
indicating the presence of heart disease and several feature columns.  The
code will automatically detect common target column names such as
`target`, `HeartDisease`, `heart_disease`, `output` or `disease`.

## Required Location

```
medical-risk-prediction-ml/
└── data/
    └── raw/
        └── heart_disease.csv
```

If the file is missing, running `python train_model.py` will display a
clear message telling you where to place it.  The repository includes
`.gitkeep` files to ensure that empty directories are tracked; do not
delete them.

## Typical Columns

Public heart disease datasets often include the following columns【505570677987423†L90-L130】:

| Column    | Description                                    |
|-----------|------------------------------------------------|
| age       | Age in years                                  |
| sex       | Sex (e.g. 1 = male, 0 = female)               |
| cp        | Chest pain type (categorical)                 |
| trestbps  | Resting blood pressure (mm Hg)                |
| chol      | Serum cholesterol (mg/dl)                     |
| fbs       | Fasting blood sugar > 120 mg/dl (1 = true)    |
| restecg   | Resting electrocardiographic results          |
| thalach   | Maximum heart rate achieved                   |
| exang     | Exercise‑induced angina (1 = yes, 0 = no)      |
| oldpeak   | ST depression induced by exercise             |
| slope     | Slope of the peak exercise ST segment         |
| ca        | Number of major vessels colored by fluoroscopy|
| thal      | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect) |
| target    | Diagnosis of heart disease (0 = no, 1 = yes)   |

Your file may use slightly different column names; as long as the target
column is named according to one of the accepted names above, the
training script will locate it automatically.

## Do Not Commit Data

Never commit large or private medical data to version control.  The
`.gitignore` file is configured to ignore CSV files in the `data/raw/`
and `data/processed/` directories.