import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:,.2f}'.format)

HECTARE_TO_ACRE = 2.47105

BASE_DIR = Path(__file__).resolve().parents[1]  # project root (PS1/)

DATA_PATH = BASE_DIR / "data" / "processed" / "Pesticide.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Pesticide.csv was not found. Expected near: {DATA_PATH.resolve()}")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())

print('Missing values:')
print(df.isnull().sum())

print('Duplicate rows:', df.duplicated().sum())
print('Years:', df['Crop_Year'].min(), 'to', df['Crop_Year'].max())
print('Number of crops:', df['Crop'].nunique())
print('Number of states:', df['State'].nunique())

print(df[['Area', 'Annual_Rainfall', 'Pesticide']].describe())

df['Pesticide_per_hectare_check'] = df['Pesticide'] / df['Area']
yearly_rate = df.groupby('Crop_Year')['Pesticide_per_hectare_check'].agg(
    ['mean', 'std', 'min', 'max']
)
print(yearly_rate)

plt.figure(figsize=(10, 4))
plt.plot(yearly_rate.index, yearly_rate['mean'], marker='o')
plt.xlabel('Year')
plt.ylabel('Average pesticide quantity per hectare')
plt.title('Pesticide quantity per hectare by year')
plt.grid(True)
plt.show()

df['Area_acres'] = df['Area'] * HECTARE_TO_ACRE
df['Annual_Rainfall_mm'] = df['Annual_Rainfall']

features = [
    'Crop',
    'Crop_Year',
    'Season',
    'State',
    'Area_acres',
    'Annual_Rainfall_mm'
]
target = 'Pesticide'

df = df.sort_values('Crop_Year').reset_index(drop=True)

train = df[df['Crop_Year'] <= 2016].copy()
validation = df[df['Crop_Year'].isin([2017, 2018])].copy()
test = df[df['Crop_Year'] >= 2019].copy()

X_train = train[features]
y_train = train[target]

X_validation = validation[features]
y_validation = validation[target]

X_test = test[features]
y_test = test[target]

print('Training rows:', len(train))
print('Validation rows:', len(validation))
print('Test rows:', len(test))
print('Test years:', sorted(test['Crop_Year'].unique()))

categorical_features = ['Crop', 'Season', 'State']
numerical_features = ['Crop_Year', 'Area_acres', 'Annual_Rainfall_mm']

preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('numerical', 'passthrough', numerical_features)
    ]
)

models = {
    'Extra Trees': ExtraTreesRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Hist Gradient Boosting': HistGradientBoostingRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'Dummy': DummyRegressor(strategy='mean')
}

results = []
trained_models = {}

for model_name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_validation)

    mae = mean_absolute_error(y_validation, predictions)
    rmse = np.sqrt(mean_squared_error(y_validation, predictions))
    r2 = r2_score(y_validation, predictions)

    results.append({
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    })

    trained_models[model_name] = pipeline

results_df = pd.DataFrame(results).sort_values('MAE').reset_index(drop=True)
print(results_df)

best_model_name = results_df.loc[0, 'Model']
best_model = trained_models[best_model_name]

print('Best model based on validation MAE:', best_model_name)
print('Validation MAE:', results_df.loc[0, 'MAE'])
print('Validation RMSE:', results_df.loc[0, 'RMSE'])
print('Validation R2:', results_df.loc[0, 'R2'])

parameter_options = {
    'Extra Trees': {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 15, 25, 35],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': [0.7, 1.0]
    },
    'Random Forest': {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 15, 25, 35],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': [0.7, 1.0]
    },
    'Gradient Boosting': {
        'model__n_estimators': [100, 150, 200],
        'model__learning_rate': [0.03, 0.05, 0.1],
        'model__max_depth': [2, 3, 4],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'Hist Gradient Boosting': {
        'model__max_iter': [100, 150, 200],
        'model__learning_rate': [0.03, 0.05, 0.1],
        'model__max_leaf_nodes': [15, 31, 45],
        'model__l2_regularization': [0.0, 0.1, 1.0]
    },
    'Decision Tree': {
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'Linear Regression': {
        'model__fit_intercept': [True, False]
    },
    'Ridge': {
        'model__alpha': [0.1, 1.0, 10.0, 50.0, 100.0]
    },
    'AdaBoost': {
        'model__n_estimators': [50, 100, 150, 200],
        'model__learning_rate': [0.03, 0.05, 0.1, 0.5, 1.0]
    }
}

if best_model_name == 'Dummy':
    tuned_model = best_model
    print('Dummy model does not require hyperparameter tuning.')
else:
    time_split = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        estimator=best_model,
        param_distributions=parameter_options[best_model_name],
        n_iter=8,
        scoring='neg_mean_absolute_error',
        cv=time_split,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_

    print('Best parameters:')
    print(search.best_params_)
    print('Best cross-validation MAE:', -search.best_score_)

original_validation_predictions = best_model.predict(X_validation)
tuned_validation_predictions = tuned_model.predict(X_validation)

comparison = pd.DataFrame([
    {
        'Model': 'Original ' + best_model_name,
        'MAE': mean_absolute_error(y_validation, original_validation_predictions),
        'RMSE': np.sqrt(mean_squared_error(y_validation, original_validation_predictions)),
        'R2': r2_score(y_validation, original_validation_predictions)
    },
    {
        'Model': 'Tuned ' + best_model_name,
        'MAE': mean_absolute_error(y_validation, tuned_validation_predictions),
        'RMSE': np.sqrt(mean_squared_error(y_validation, tuned_validation_predictions)),
        'R2': r2_score(y_validation, tuned_validation_predictions)
    }
])

print(comparison)

if comparison.loc[1, 'MAE'] < comparison.loc[0, 'MAE']:
    selected_model = tuned_model
    selected_model_name = 'Tuned ' + best_model_name
else:
    selected_model = best_model
    selected_model_name = 'Original ' + best_model_name

print('Selected model:', selected_model_name)

test_predictions = selected_model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_r2 = r2_score(y_test, test_predictions)

test_results = pd.DataFrame([{
    'Selected Model': selected_model_name,
    'Test MAE': test_mae,
    'Test RMSE': test_rmse,
    'Test R2': test_r2
}])

print(test_results)

training_validation = pd.concat([train, validation], ignore_index=True)

X_final = training_validation[features]
y_final = training_validation[target]

final_model = selected_model
final_model.fit(X_final, y_final)

model_path = BASE_DIR / 'models' / 'pesticide_prediction_model.pkl'
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(final_model, model_path)

print('Model saved successfully:')
print(model_path.resolve())
print('Model:', selected_model_name)
print('Test MAE:', round(test_mae, 2))
print('Test RMSE:', round(test_rmse, 2))
print('Test R2:', round(test_r2, 4))
