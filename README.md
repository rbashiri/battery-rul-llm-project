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

## Feature Importance Analysis
Feature importance analysis was performed using the Random Forest Regressor model because tree-based models can quantify the contribution of each feature toward prediction performance.

Feature importance analysis was performed using the Random Forest Regressor model to identify the variables that most strongly influence battery Remaining Useful Life (RUL) prediction.

The results showed that the most influential features were:

| Feature | Importance |
=======================
| BCt     |  0.423     |
| SOH     |  0.322     |
| cycle   |  0.211     |

All remaining features had relatively small importance values (< 0.01).

### Interpretation

- **BCt** was identified as the most important predictor, contributing approximately 42% of the model importance. This suggests that BCt has a strong relationship with battery degradation and Remaining Useful Life.

- **SOH (State of Health)** was the second most influential feature, contributing about 32% importance. This indicates that battery health strongly affects RUL prediction accuracy.

- **cycle** also showed significant influence (~21%), confirming that the number of charge/discharge cycles is an important degradation indicator.

- Other variables such as `chI`, `disV`, `chT`, `disI`, `chV`, and `disT` contributed minimally to the prediction model.

Overall, the feature importance analysis indicates that battery degradation behavior in this dataset is primarily driven by BCt, SOH, and cycle-related characteristics.

## MLflow Experiment Tracking

This project uses MLflow to track and compare battery Remaining Useful Life (RUL) prediction experiments.

Each experiment run logs:
- dataset information
- model hyperparameters
- evaluation metrics
- trained model artifacts

The following metrics were tracked:
- MAE
- MSE
- RMSE
- R²

Five experiment runs were completed using different regression models and hyperparameter configurations:
- Linear Regression
- Random Forest Regressor with different tree depths
- Gradient Boosting Regressor with different learning rates

MLflow was also used to programmatically compare all experiment runs using `mlflow.search_runs()`.

The best experiment run was automatically identified based on the lowest RMSE value.

### Best MLflow Run

| Metric | Value |
|---|---|
| Best Model | Linear Regression |
| RMSE | 18.0005 |
| MAE | 16.8801 |
| R² | 0.9309 |

The best trained model was saved locally as:

```text
models/best_model.pkl