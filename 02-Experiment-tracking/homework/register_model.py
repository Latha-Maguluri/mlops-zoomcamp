import os
import pickle
import click
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from sklearn.metrics import mean_squared_error


HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models to evaluate"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Step 1: get top N models from HPO experiment
    hpo_experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=hpo_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )

    # Step 2: load test set
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    # Step 3: evaluate top N models on test set
    best_rmse = float("inf")
    best_run = None

    for run in runs:
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5

        print(f"Run: {run_id}, RMSE: {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_run = run

    # Step 4: log to new experiment
    with mlflow.start_run():
        mlflow.log_metric("rmse", best_rmse)
        mlflow.set_tag("model_type", "random_forest")
        mlflow.set_tag("run_id", best_run.info.run_id)

    # Step 5: register the best model
    best_model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name = "random-forest-regressor-best"
    mlflow.register_model(model_uri=best_model_uri, name=model_name)
    print(f"✅ Registered model: {model_name} (RMSE: {best_rmse:.4f})")


if __name__ == '__main__':
    run_register_model()
