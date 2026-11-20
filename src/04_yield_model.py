import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

BASE_DIR = Path(__file__).resolve().parents[1]  # project root (PS1/)
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

data_path = BASE_DIR / "data" / "processed" / "crop_yield_processed.csv"
if not data_path.exists():
    raise FileNotFoundError("crop_yield_processed.csv was not found. Run src/02_preprocessing.py first.")

df = pd.read_csv(data_path)

df = df.dropna(subset=["Yield_kg_per_acre", "Crop_Year"])
df["Crop_Year"] = pd.to_numeric(df["Crop_Year"])
df = df.sort_values("Crop_Year").reset_index(drop=True)

df["Fertilizer_per_acre"] = df["Fertilizer_kg"] / df["Area_acres"].replace(0, np.nan)
df["Pesticide_per_acre"] = df["Pesticide_kg"] / df["Area_acres"].replace(0, np.nan)

print("Dataset shape:", df.shape)
print(df.head())

features = [
    "Crop", "Crop_Year", "Season", "State", "Area_acres",
    "Annual_Rainfall_mm", "Fertilizer_kg", "Pesticide_kg",
    "Fertilizer_per_acre", "Pesticide_per_acre"
]

X = df[features]
y = df["Yield_kg_per_acre"]

train_rows = df["Crop_Year"] <= 2016
validation_rows = df["Crop_Year"].between(2017, 2018)
test_rows = df["Crop_Year"] >= 2019

X_train, y_train = X.loc[train_rows], y.loc[train_rows]
X_val, y_val = X.loc[validation_rows], y.loc[validation_rows]
X_test, y_test = X.loc[test_rows], y.loc[test_rows]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

categorical_columns = ["Crop", "Season", "State"]
numeric_columns = [column for column in features if column not in categorical_columns]

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_columns),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical_columns)
])

models = {
    "Dummy": DummyRegressor(strategy="median"),
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=10),
    "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=2, max_features=0.9, n_jobs=-1, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, max_features=0.9, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=100, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=80, learning_rate=0.05, random_state=42)
}

model_results = []
trained_models = {}

for model_name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    validation_predictions = pipeline.predict(X_val)

    model_results.append({
        "Model": model_name,
        "MAE": mean_absolute_error(y_val, validation_predictions),
        "RMSE": mean_squared_error(y_val, validation_predictions) ** 0.5,
        "R2": r2_score(y_val, validation_predictions)
    })

    trained_models[model_name] = pipeline

comparison = pd.DataFrame(model_results).sort_values("RMSE").reset_index(drop=True)
print(comparison.round(3))

best_model_name = comparison.loc[0, "Model"]
best_model = trained_models[best_model_name]

print("Best model:", best_model_name)

parameter_grid = {
    "model__n_estimators": [100, 150, 200],
    "model__max_depth": [None, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 3, 5],
    "model__max_features": [0.7, 0.9, 1.0]
}

tuning = RandomizedSearchCV(
    Pipeline([
        ("preprocessor", preprocessor),
        ("model", ExtraTreesRegressor(n_jobs=-1, random_state=42))
    ]),
    param_distributions=parameter_grid,
    n_iter=8,
    scoring="neg_root_mean_squared_error",
    cv=TimeSeriesSplit(n_splits=3),
    random_state=42,
    n_jobs=1
)

tuning.fit(X_train, y_train)

tuned_model = tuning.best_estimator_

print("Best parameters:")
print(tuning.best_params_)

before_tuning = best_model.predict(X_val)
after_tuning = tuned_model.predict(X_val)

tuning_results = pd.DataFrame([
    {
        "Version": "Before tuning",
        "MAE": mean_absolute_error(y_val, before_tuning),
        "RMSE": mean_squared_error(y_val, before_tuning) ** 0.5,
        "R2": r2_score(y_val, before_tuning)
    },
    {
        "Version": "After tuning",
        "MAE": mean_absolute_error(y_val, after_tuning),
        "RMSE": mean_squared_error(y_val, after_tuning) ** 0.5,
        "R2": r2_score(y_val, after_tuning)
    }
])

print(tuning_results.round(3))

X_train_validation = pd.concat([X_train, X_val])
y_train_validation = pd.concat([y_train, y_val])

final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", ExtraTreesRegressor(
        n_estimators=100,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=0.7,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    ))
])

final_model.fit(X_train_validation, y_train_validation)
test_predictions = final_model.predict(X_test)

final_results = pd.DataFrame([{
    "Model": "Tuned Extra Trees",
    "MAE": mean_absolute_error(y_test, test_predictions),
    "RMSE": mean_squared_error(y_test, test_predictions) ** 0.5,
    "R2": r2_score(y_test, test_predictions)
}])

print(final_results.round(3))

MODEL_PATH = MODELS_DIR / "best_yield_prediction_model.pkl"
joblib.dump(final_model, MODEL_PATH)
comparison.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
tuning_results.to_csv(MODELS_DIR / "tuning_comparison.csv", index=False)
final_results.to_csv(MODELS_DIR / "final_test_metrics.csv", index=False)

print("Model saved to:", MODEL_PATH)
print("Result files saved in:", MODELS_DIR)
