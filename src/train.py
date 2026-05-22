import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(model_name, y_true, y_pred):
    """
    Calculate regression evaluation metrics.
    """

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


def main():

    # ==============================
    # Load cleaned dataset
    # ==============================

    input_path = (
        "/home/susan/battery-rul-llm-project/"
        "data/processed/battery_cleaned.csv"
    )

    df = pd.read_csv(input_path)

    print("Dataset loaded successfully")
    print("Dataset shape:", df.shape)

    # ==============================
    # Select target column
    # ==============================

    target_column = "RUL"

    # ==============================
    # Create features and target
    # ==============================

    features = df.drop(columns=[target_column])

    target = df[target_column]

    # ==============================
    # Remove ID column
    # ==============================

    if "battery_id" in features.columns:

        features = features.drop(columns=["battery_id"])

        print("battery_id column removed")

    # ==============================
    # Remove text columns
    # ==============================

    text_columns = features.select_dtypes(
        include="object"
    ).columns

    if len(text_columns) > 0:

        print("Text columns removed:")
        print(list(text_columns))

        features = features.drop(columns=text_columns)

    # ==============================
    # Train-test split
    # ==============================

    X_temp, X_test, y_temp, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42
    )

    # ==============================
    # Train-validation split
    # ==============================

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=42
    )

    print("Train shape:", X_train.shape)

    print("Validation shape:", X_val.shape)

    print("Test shape:", X_test.shape)

    # ==============================
    # Define models
    # ==============================

    models = {

    # Run 1
    "Linear Regression":
    LinearRegression(),

    # Run 2
    "Random Forest Depth 5":
    RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ),

    # Run 3
    "Random Forest Depth 10":
    RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ),

    # Run 4
    "Gradient Boosting LR 0.1":
    GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),

    # Run 5
    "Gradient Boosting LR 0.05":
    GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
}

    # ==============================
    # Set MLflow experiment
    # ==============================

    mlflow.set_experiment(
        "battery_rul_prediction"
    )

    # ==============================
    # Train and evaluate models
    # ==============================

    results = []

    trained_models = {}

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        # ==========================
        # Start MLflow run
        # ==========================

        with mlflow.start_run(
            run_name=model_name
        ):

            # ======================
            # Train model
            # ======================

            model.fit(
                X_train,
                y_train
            )

            # ======================
            # Validation predictions
            # ======================

            y_val_pred = model.predict(
                X_val
            )

            # ======================
            # Calculate metrics
            # ======================

            metrics = evaluate_model(
                model_name,
                y_val,
                y_val_pred
            )

            results.append(metrics)

            trained_models[model_name] = model

            # ======================
            # MLflow parameters
            # ======================

            mlflow.log_param(
                "data_path",
                input_path
            )

            mlflow.log_param(
                "data_version",
                "battery_cleaned.csv"
            )

            mlflow.log_param(
                "target_column",
                target_column
            )

            # ======================
            # Log hyperparameters
            # ======================

            mlflow.log_params(
                model.get_params()
            )

            # ======================
            # Log metrics
            # ======================

            mlflow.log_metric(
                "MAE",
                metrics["MAE"]
            )

            mlflow.log_metric(
                "MSE",
                metrics["MSE"]
            )

            mlflow.log_metric(
                "RMSE",
                metrics["RMSE"]
            )

            mlflow.log_metric(
                "R2",
                metrics["R2"]
            )

            # ======================
            # Log model artifact
            # ======================

            mlflow.sklearn.log_model(
                model,
                artifact_path="model"
            )

            # ======================
            # Print metrics
            # ======================

            print(
                "MAE:",
                round(metrics["MAE"], 4)
            )

            print(
                "RMSE:",
                round(metrics["RMSE"], 4)
            )

            print(
                "R2:",
                round(metrics["R2"], 4)
            )

    # ==============================
    # Compare models
    # ==============================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="RMSE",
        ascending=True
    )

    print("\nModel Comparison:")

    print(results_df)

    # ==============================
    # Select best model
    # ==============================

    best_model_name = (
        results_df.iloc[0]["Model"]
    )

    best_model = (
        trained_models[best_model_name]
    )

    print("\nBest Model:")

    print(best_model_name)

    # ==============================
    # Final test prediction
    # ==============================

    y_test_pred = best_model.predict(
        X_test
    )

    # ==============================
    # Final test evaluation
    # ==============================

    test_metrics = evaluate_model(
        best_model_name + " Final Test",
        y_test,
        y_test_pred
    )

    print("\nFinal Test Results:")

    print(test_metrics)

    # ==============================
    # Create reports folder
    # ==============================

    os.makedirs(
        "reports",
        exist_ok=True
    )

    # ==============================
    # Save comparison report
    # ==============================

    results_df.to_csv(
        "reports/model_comparison.csv",
        index=False
    )

    # ==============================
    # Save final test report
    # ==============================

    test_results_df = pd.DataFrame(
        [test_metrics]
    )

    test_results_df.to_csv(
        "reports/final_test_results.csv",
        index=False
    )

    # ==============================
    # Create models folder
    # ==============================

    os.makedirs(
        "models",
        exist_ok=True
    )

    # ==============================
    # Save best model locally
    # ==============================

    joblib.dump(
        best_model,
        "models/best_model.pkl"
    )

    print(
        "\nBest model saved to "
        "models/best_model.pkl"
    )

    # ==============================
    # Feature importance
    # ==============================

    rf_model = trained_models[
    "Random Forest Depth 10"]

    feature_importance = pd.DataFrame({

        "Feature": X_train.columns,

        "Importance":
        rf_model.feature_importances_
    })

    feature_importance = (
        feature_importance.sort_values(
            by="Importance",
            ascending=False
        )
    )

    # ==============================
    # Save feature importance CSV
    # ==============================

    feature_importance.to_csv(
        "reports/feature_importance.csv",
        index=False
    )

    # ==============================
    # Save feature importance plot
    # ==============================

    plt.figure(figsize=(10, 6))

    plt.bar(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    plt.xticks(rotation=45)

    plt.xlabel("Features")

    plt.ylabel("Importance")

    plt.title(
        "Feature Importance for "
        "Battery RUL Prediction"
    )

    plt.tight_layout()

    plt.savefig(
        "reports/feature_importance.png"
    )

    plt.close()

    print(
        "Model comparison saved "
        "to reports/model_comparison.csv"
    )

    print(
        "Final test results saved "
        "to reports/final_test_results.csv"
    )

    print(
        "Feature importance saved "
        "to reports/feature_importance.csv"
    )

    print(
        "Feature importance plot saved "
        "to reports/feature_importance.png"
    )


if __name__ == "__main__":

    main()
