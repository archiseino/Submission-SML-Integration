import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

mlflow.set_tracking_uri("http://localhost:5000")  # Set your MLflow tracking URI

mlflow.set_experiment("AmesHousing_Experiment_Tuned")

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

# 🔁 Log the best result in a final separate MLflow run
with mlflow.start_run(run_name="best_model_summary"):
    mlflow.log_params(best_params)
    mlflow.log_metric("best_val_mae", best_score)

    # Optional: save model artifact
    mlflow.sklearn.log_model(best_model, "best_random_forest_model")

    print("Best MAE:", best_score)
    print("Best Params:", best_params)
