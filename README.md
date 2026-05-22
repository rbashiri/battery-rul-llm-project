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

# Stage 3: LLM Interface for Battery RUL Prediction

## Overview

This stage implements a Streamlit-based battery-health assistant that connects natural language user input with the trained machine learning Remaining Useful Life (RUL) prediction model.

The application allows users to describe battery conditions in conversational language. The system extracts the required battery features, validates the inputs, invokes the trained machine learning model, and generates an engineering explanation of the prediction.

---

# Stage 3 Requirements Coverage

| Reviewer Requirement | Implementation in This Project |
|---|---|
| LLM correctly parses natural language input into model features | The application extracts battery features such as voltage, current, temperature, cycle number, capacity time, and SOH from conversational text |
| Trained model is loaded and invoked with parsed features | The trained `best_model.pkl` model is loaded using `joblib` and used for inference |
| Response is clear, contextual, and includes the prediction | The application explains battery degradation and Remaining Useful Life in engineering context |
| Edge cases handled gracefully | Missing values, incomplete inputs, and ambiguous queries are detected before prediction |
| Interface is functional and easy to use | The project uses a Streamlit web application for user interaction |

---

# Application Workflow

```text
User Natural Language Query
            ↓
Input Parsing
            ↓
Feature Validation
            ↓
Model Prediction
            ↓
Engineering Explanation
            ↓
Prediction Display
```

---

# 1. Natural Language Input Parsing

The application accepts conversational battery descriptions from the user.

## Example User Input

```text
Predict RUL for cycle 120, charge current 1.5,
charge voltage 4.1, charge temperature 30,
discharge current 1.2, discharge voltage 3.7,
discharge temperature 28, battery capacity time 250,
and SOH 85.
```

The system extracts structured battery features required by the machine learning model.

## Parsed Features

```python
{
    "cycle": 120,
    "chI": 1.5,
    "chV": 4.1,
    "chT": 30,
    "disI": 1.2,
    "disV": 3.7,
    "disT": 28,
    "BCt": 250,
    "SOH": 85
}
```

This demonstrates that the system converts conversational text into structured machine learning inputs.

---

# 2. Model Invocation

The trained battery Remaining Useful Life prediction model is stored in:

```text
models/best_model.pkl
```

The application loads the trained model using `joblib`.

## Example

```python
model = joblib.load("models/best_model.pkl")
```

The extracted features are converted into a pandas DataFrame and passed into the trained model.

## Example

```python
prediction = model.predict(input_df)
```

This ensures that the actual trained machine learning model is used for inference.

---

# 3. Contextual Response Generation

The application generates engineering explanations instead of returning only a numerical prediction.

## Example Output

```text
Predicted Remaining Useful Life: 85.4 cycles

Battery Status: High degradation

Explanation:
The battery has a low remaining useful life and shows significant degradation.
This may indicate reduced energy storage capability and possible need for replacement or maintenance soon.
```

This helps users understand the practical meaning of the prediction in battery engineering context.

---

# 4. Edge Case Handling

The application validates user inputs before prediction.

Handled edge cases include:

- missing battery information
- incomplete queries
- ambiguous requests
- empty inputs
- invalid feature combinations

## Example Missing Input

```text
Predict battery RUL for cycle 120 and charge current 1.5
```

## Application Response

```text
Missing required information:
Please provide charge voltage, charge temperature,
discharge voltage, discharge temperature,
battery capacity time, and SOH.
```

This prevents invalid predictions and improves reliability.

---

# 5. Streamlit Interface

The interface is implemented using Streamlit.

The application includes:

- text input area
- prediction button
- parsed feature visualization
- engineering explanation output

This creates a simple and user-friendly battery-health assistant application.

---

# How to Run the Application

## Step 1: Activate Virtual Environment

```bash
cd ~/battery-rul-llm-project
source .venv/bin/activate
```

---

## Step 2: Run Streamlit Application

```bash
python -m streamlit run src/app.py
```

---

# Example Questions for Testing

## Example 1

```text
Predict RUL for cycle 120, charge current 1.5,
charge voltage 4.1, charge temperature 30,
discharge current 1.2, discharge voltage 3.7,
discharge temperature 28, battery capacity time 250,
and SOH 85.
```

``` response
Prediction Result
Predicted Remaining Useful Life: 70.62 cycles

Battery Status: High degradation

Explanation:
The battery has a low remaining useful life. This suggests significant degradation, reduced capacity, and possible need for replacement or maintenance soon.
```

## Example 2

```text
Battery cycle 250, charge current 2.0,
charge voltage 4.0, SOH 70,
discharge voltage 3.5, discharge current 1.8,
charge temperature 35, discharge temperature 32,
capacity time 180.
```
``` response
Prediction Result
Predicted Remaining Useful Life: -78.66 cycles

Battery Status: High degradation

Explanation:
The battery has a low remaining useful life. This suggests significant degradation, reduced capacity, and possible need for replacement or maintenance soon.
```
-------

## Example 3: Missing Features

```text
Predict battery life for cycle 100
```

Expected application behavior:

```text
The application asks the user to provide missing battery information before prediction.
```

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| pandas | Data processing |
| scikit-learn | Machine learning |
| Streamlit | Web application |
| joblib | Model loading |
| Regex | Natural language feature extraction |
| python-dotenv | Environment variable management |

---

# Summary

This stage demonstrates complete integration between:
- natural language input parsing,
- machine learning inference,
- engineering response generation,
- edge case handling,
- and an interactive Streamlit web application.

The final system allows users to interact with the trained battery Remaining Useful Life prediction model using conversational language while receiving interpretable engineering explanations about battery degradation and battery health.