# AI-Powered Li-Ion Battery RUL Prediction System

## Project Overview

This project predicts the Remaining Useful Life (RUL) of lithium-ion batteries using machine learning and provides a natural language interface powered by an LLM.

## Dataset
Lithium-Ion Battery Degradation Dataset
**Source:**
https://www.kaggle.com/datasets/programmer3/lithium-ion-battery-degradation-dataset?select=Battery_dataset.csv

This dataset is modeled after the NASA Ames Prognostics Center of Excellence lithium-ion battery degradation dataset. It simulates the charge-discharge behavior and aging process of lithium-ion batteries across multiple cycles, capturing realistic trends in battery health over time.

The dataset features three virtual battery cells B0005 (B5), B0006 (B6) and B0007 (B7) and includes average values per cycle for key parameters such as:

Charging/Discharging Current

Charging/Discharging Voltage

Charging/Discharging Temperature

Battery Capacity (BCt)

State of Health (SOH)

Remaining Useful Life (RUL)


## Project Goals

- Train ML models for battery RUL prediction
- Track experiments using MLflow
- Build an LLM-powered interface
- Deploy a production-style ML application


## Data Preprocessing:

The preprocessing pipeline was developed to inspect and prepare the battery dataset before model training.

The following preprocessing steps were applied:

- Loaded the raw dataset using pandas
- Inspected dataset shape, column names, and data types
- Checked dataset information and summary statistics
- Checked for missing values and duplicate records
- Cleaned and organized the dataset for further analysis

The preprocessing workflow was implemented in:

```plaintext
src/preprocess.py
```

## Model Training


Three models were trained and compared:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Random Forest performed best because it captured nonlinear battery degradation behavior and achieved lower RMSE compared to the other models.

## Model Evaluation

The following metrics were used:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

These metrics evaluate prediction accuracy and model generalization.

** Identifier columns such as `Battery_ID` were removed before model training because they do not provide meaningful predictive information for Remaining Useful Life estimation.**

## Best Model Selection

Three machine learning models were evaluated for battery Remaining Useful Life (RUL) prediction:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The models were compared using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

 Model                        | MAE   | RMSE  | R²    |
| Linear Regression           | 16.88 | 18.00 | 0.931 |
| Random Forest Regressor     | 16.55 | 18.59 | 0.926 |
| Gradient Boosting Regressor | 16.55 | 18.81 | 0.925 |

Based on the evaluation metrics, Linear Regression was selected as the best-performing model because it achieved:

- the highest R² score
- the lowest RMSE
- strong overall generalization performance

These results indicate that the dataset exhibits relatively linear relationships between battery measurements and Remaining Useful Life (RUL).