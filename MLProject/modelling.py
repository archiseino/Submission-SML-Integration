import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __name__ == "__main__":

    # Load preprocessed dataset
    df = pd.read_csv('AmesHousing_preprocessing.csv')

    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = [
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 15},
        {"n_estimators": 150, "max_depth": None},
    ]

    best_score = float("inf")
    best_params = None
    best_model = None

    for params in param_grid:
        model = RandomForestRegressor(**params, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)

        # Track best
        if mae < best_score:
            best_score = mae
            best_params = params
            best_model = model  # Save best model object

    ## Make the best model run
    with mlflow.start_run():
        model = RandomForestRegressor(**best_params, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log metrics
        mlflow.log_metric("val_mean_absolute_error", mae)
        mlflow.log_metric("val_mean_squared_error", mse)
        mlflow.log_metric("val_r2_score", r2)