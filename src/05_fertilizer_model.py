import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

BASE_DIR = Path(__file__).resolve().parents[1]  # project root (PS1/)

DATA_PATH = BASE_DIR / "data" / "processed" / "Fertilizer.csv"

if not DATA_PATH.is_file():
    raise FileNotFoundError("Fertilizer.csv was not found. Run src/03_split_fertilizer_pesticide_data.py first.")
df = pd.read_csv(DATA_PATH)

print("Using file:", DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())

print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())
print(df.describe())

df["Fertilizer_per_area"] = df["Fertilizer"] / df["Area"]

yearly_rate = df.groupby("Crop_Year")["Fertilizer_per_area"].agg(
    ["mean", "std", "min", "max", "count"]
)

print(yearly_rate)

HECTARE_TO_ACRE = 2.47105

df['Area_acres'] = df['Area'] * HECTARE_TO_ACRE

features = [
    "Crop",
    "Crop_Year",
    "Season",
    "State",
    "Area_acres",
    "Annual_Rainfall"
]

target = "Fertilizer"

X = df[features]
y = df[target]

categorical_features = ["Crop", "Season", "State"]
numerical_features = ["Crop_Year", "Area_acres", "Annual_Rainfall"]

print("Input columns:", features)
print("Target column:", target)

train = df[df["Crop_Year"] <= 2016].copy()
validation = df[df["Crop_Year"].isin([2017, 2018])].copy()
test = df[df["Crop_Year"] >= 2019].copy()

X_train = train[features]
y_train = train[target]

X_validation = validation[features]
y_validation = validation[target]

print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Test rows:", len(test))
print("Training years:", train["Crop_Year"].min(), "-", train["Crop_Year"].max())
print("Validation years:", sorted(validation["Crop_Year"].unique()))
print("Test years:", sorted(test["Crop_Year"].unique()))

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ("numerical", StandardScaler(), numerical_features)
])

models = {
    "Dummy": DummyRegressor(strategy="mean"),
    "Ridge": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        min_samples_leaf=5
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=120,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=120,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42)
}

results = []
trained_models = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_validation)

    mae = mean_absolute_error(y_validation, predictions)
    rmse = mean_squared_error(y_validation, predictions) ** 0.5
    r2 = r2_score(y_validation, predictions)

    results.append([name, mae, rmse, r2])
    trained_models[name] = pipeline

results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2"]
).sort_values("MAE")

print(results_df)

results_df.set_index("Model")[["MAE", "RMSE"]].plot(
    kind="bar",
    figsize=(11, 5),
    title="Model comparison on validation data"
)
plt.ylabel("Error")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

results_df.set_index("Model")["R2"].plot(
    kind="bar",
    figsize=(11, 4),
    title="R-squared comparison"
)
plt.ylabel("R-squared")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

train_validation = df[df["Crop_Year"] <= 2018].sort_values("Crop_Year").copy()

X_train_validation = train_validation[features]
y_train_validation = train_validation[target]

random_forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_jobs=1))
])

parameters = {
    "model__n_estimators": [80, 120, 160],
    "model__max_depth": [None, 15, 25],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": [1.0, "sqrt"]
}

time_split = TimeSeriesSplit(n_splits=2)

search = RandomizedSearchCV(
    random_forest_pipeline,
    parameters,
    n_iter=5,
    cv=time_split,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=1
)

search.fit(X_train_validation, y_train_validation)

print("Best parameters:")
print(search.best_params_)
print("Best cross-validation MAE:", -search.best_score_)

best_model = search.best_estimator_

X_test = test[features]
y_test = test[target]

test_predictions = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = mean_squared_error(y_test, test_predictions) ** 0.5
test_r2 = r2_score(y_test, test_predictions)

print("Final Test Results")
print("MAE:", round(test_mae, 2))
print("RMSE:", round(test_rmse, 2))
print("R2:", round(test_r2, 4))

comparison = pd.DataFrame({
    "Actual_Fertilizer": y_test.values,
    "Predicted_Fertilizer": test_predictions
})

print(comparison.head(15))

MODEL_PATH = BASE_DIR / "models" / "fertilizer_prediction_model1.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, MODEL_PATH)

print("Model saved successfully:")
print(MODEL_PATH.resolve())
