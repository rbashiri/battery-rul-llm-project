import mlflow


def main():
    """Compare MLflow runs and identify the best model based on lowest RMSE."""

    experiment_name = "battery_rul_prediction"

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print("Experiment not found.")
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.RMSE ASC"]
    )

    if runs.empty:
        print("No runs found.")
        return

    best_run = runs.iloc[0]

    print("\nBest MLflow Run")
    print("----------------")
    print("Run ID:", best_run["run_id"])
    print("Model:", best_run["tags.mlflow.runName"])
    print("RMSE:", best_run["metrics.RMSE"])
    print("MAE:", best_run["metrics.MAE"])
    print("R2:", best_run["metrics.R2"])


if __name__ == "__main__":
    main()
