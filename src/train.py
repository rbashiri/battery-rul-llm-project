import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model_name, y_true, y_pred):
    """Calculate regression evaluation metrics."""
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
    # Load cleaned dataset
    input_path = "/home/susan/battery-rul-llm-project/data/processed/battery_cleaned.csv"
    df = pd.read_csv(input_path)

    # Select target column
    target_column = "RUL"

    # Create features and target
    features = df.drop(columns=[target_column])
    target = df[target_column]

    # Remove battery ID column because it is an identifier, not a numeric feature
    if "battery_id" in features.columns:
        features = features.drop(columns=["battery_id"])

    # Remove any remaining text columns
    text_columns = features.select_dtypes(include="object").columns
    if len(text_columns) > 0:
        print("Text columns removed:", list(text_columns))
        features = features.drop(columns=text_columns)

    # Split into temporary data and final test data
    X_temp, X_test, y_temp, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42
    )

    # Split temporary data into training and validation data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=42
    )

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    # Define models
    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }

    # Train and evaluate models
    results = []
    trained_models = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)

        metrics = evaluate_model(model_name, y_val, y_val_pred)
        results.append(metrics)

        trained_models[model_name] = model

    # Compare validation results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="RMSE", ascending=True)

    print("\nModel Comparison:")
    print(results_df)

    # Select best model based on lowest RMSE
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    print("\nBest Model:", best_model_name)

    # Evaluate best model on final test set
    y_test_pred = best_model.predict(X_test)

    test_metrics = evaluate_model(
        best_model_name + " Final Test",
        y_test,
        y_test_pred
    )

    print("\nFinal Test Results:")
    print(test_metrics)

    # Save reports folder
    os.makedirs("reports", exist_ok=True)

    # Save model comparison results
    results_df.to_csv("reports/model_comparison.csv", index=False)

    # Save final test results
    test_results_df = pd.DataFrame([test_metrics])
    test_results_df.to_csv("reports/final_test_results.csv", index=False)

    # Save models folder
    os.makedirs("models", exist_ok=True)

    # Save best model
    joblib.dump(best_model, "models/best_model.pkl")

    print("\nBest model saved to models/best_model.pkl")

    # Feature importance using Random Forest
    rf_model = trained_models["Random Forest Regressor"]

    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": rf_model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    # Save feature importance table
    feature_importance.to_csv(
        "reports/feature_importance.csv",
        index=False
    )

    # Save feature importance plot
    plt.figure(figsize=(10, 6))

    plt.bar(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    plt.xticks(rotation=45)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Feature Importance for Battery RUL Prediction")
    plt.tight_layout()

    plt.savefig("reports/feature_importance.png")
    plt.close()

    print("Model comparison saved to reports/model_comparison.csv")
    print("Final test results saved to reports/final_test_results.csv")
    print("Feature importance saved to reports/feature_importance.csv")
    print("Feature importance plot saved to reports/feature_importance.png")


if __name__ == "__main__":
    main()
