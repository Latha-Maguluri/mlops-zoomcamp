from pathlib import Path
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import mlflow
from prefect import flow, task

@task
def read_dataframe(year: int, month: int) -> pd.DataFrame:
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)
    df = df.sample(n=10000, random_state=42)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

@task
def prepare_features(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dv = DictVectorizer()
    
    train_dicts = df[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    y_train = df['duration'].values
    
    return X_train, y_train, dv

@task
def train_and_log_model(X_train, y_train, dv):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc-taxi-experiment")
    
    models_folder = Path("models")
    models_folder.mkdir(exist_ok=True)
    
    with mlflow.start_run():
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)

        mlflow.log_param("fit_intercept", lr.fit_intercept)
        mlflow.log_metric("mse", mse)

        # Save model
        model_path = models_folder / "linear_regression_model.pkl"
        with open(model_path, "wb") as f_out:
            pickle.dump(lr, f_out)
        mlflow.log_artifact(str(model_path))

        # Save DictVectorizer
        dv_path = models_folder / "dv.pkl"
        with open(dv_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(str(dv_path))

@flow
def main(year: int = 2023, month: int = 3):
    df = read_dataframe(year, month)
    X_train, y_train, dv = prepare_features(df)
    train_and_log_model(X_train, y_train, dv)

if __name__ == "__main__":
    main()
