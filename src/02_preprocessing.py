from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', context='notebook')
pd.set_option('display.max_columns', None)

BASE_DIR = Path(__file__).resolve().parents[1]  # project root (PS1/)
raw_path = BASE_DIR / 'data' / 'raw' / 'crop_yield.csv'
if not raw_path.exists():
    raise FileNotFoundError(f'Could not find raw dataset at {raw_path}')

df = pd.read_csv(raw_path)
original_record_count = len(df)

print(f'Loaded raw data from: {raw_path}')
print(f'Dataset shape: {df.shape}')
print('Columns:', df.columns.tolist())
print(df.dtypes.rename('Data type').to_frame())
print(df.head())

coconut_mask = df['Crop'].astype(str).str.strip().str.casefold().eq('coconut')
coconut_records_removed = int(coconut_mask.sum())

print(f'Rows before Coconut exclusion: {len(df):,}')
print(f'Coconut records to remove: {coconut_records_removed:,}')
print('Reason: Coconut records were excluded because their production measurement and yield-unit consistency require separate verification.')

df = df[
    df['Crop'].astype(str).str.strip().str.casefold() != 'coconut'
].copy()

remaining_coconut_records = int(
    df['Crop'].astype(str).str.strip().str.casefold().eq('coconut').sum()
)
print(f'Rows after Coconut exclusion: {len(df):,}')
print(f'Coconut records remaining: {remaining_coconut_records:,}')
assert remaining_coconut_records == 0, 'Coconut exclusion verification failed.'

HECTARE_TO_ACRE = 2.47105
METRIC_TONNE_TO_KILOGRAM = 1000

df['Area_acres'] = df['Area'] * HECTARE_TO_ACRE
df['Production_tonnes'] = df['Production']
df['Annual_Rainfall_mm'] = df['Annual_Rainfall']
df['Fertilizer_kg'] = df['Fertilizer']
df['Pesticide_kg'] = df['Pesticide']
df['Yield_kg_per_acre'] = df['Yield'] * METRIC_TONNE_TO_KILOGRAM / HECTARE_TO_ACRE

area_conversion_valid = np.allclose(
    df['Area_acres'], df['Area'] * HECTARE_TO_ACRE, equal_nan=True
)
yield_conversion_valid = np.allclose(
    df['Yield_kg_per_acre'],
    df['Yield'] * METRIC_TONNE_TO_KILOGRAM / HECTARE_TO_ACRE,
    equal_nan=True,
)
assert area_conversion_valid, 'Area-to-acre conversion validation failed.'
assert yield_conversion_valid, 'Yield-to-kg-per-acre conversion validation failed.'
print('Area_acres conversion validation: PASSED')
print('Yield_kg_per_acre conversion validation: PASSED (working unit assumption)')

converted_columns = [
    'Area', 'Area_acres', 'Production', 'Production_tonnes',
    'Annual_Rainfall', 'Annual_Rainfall_mm', 'Fertilizer', 'Fertilizer_kg',
    'Pesticide', 'Pesticide_kg', 'Yield', 'Yield_kg_per_acre'
]
print(df[converted_columns].head())

missing_counts = df.isna().sum()
missing_percentages = (missing_counts / len(df) * 100).round(2)
missing_summary = pd.DataFrame({
    'Missing values': missing_counts,
    'Percentage': missing_percentages,
}).sort_values('Missing values', ascending=False)
print(missing_summary)

target_missing_count = int(df['Yield_kg_per_acre'].isna().sum())
print(f'Missing target values (Yield_kg_per_acre): {target_missing_count:,}')

if missing_counts.sum() == 0:
    print('No missing values found after Coconut exclusion and conversion; no imputation is performed.')
else:
    print('Missing predictors require a documented, training-only strategy in a future sklearn pipeline.')
    print('Missing target values are not imputed.')

if target_missing_count > 0:
    print(f'Excluding {target_missing_count:,} records with missing target values from the processed modelling dataset.')
    df = df.dropna(subset=['Yield_kg_per_acre']).copy()
else:
    print('No records are excluded for missing target values.')

duplicate_records_removed = int(df.duplicated().sum())
print(f'Exact duplicate rows found: {duplicate_records_removed:,}')

df = df.drop_duplicates().copy()
print(f'Rows after exact-duplicate removal: {len(df):,}')

numeric_columns_to_check = [
    'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield',
    'Area_acres', 'Production_tonnes', 'Annual_Rainfall_mm',
    'Fertilizer_kg', 'Pesticide_kg', 'Yield_kg_per_acre'
]

zero_summary = pd.DataFrame({
    'Zero values': (df[numeric_columns_to_check] == 0).sum(),
    'Negative values': (df[numeric_columns_to_check] < 0).sum(),
})
print(zero_summary)

zero_yield_records = df.loc[df['Yield'] == 0]
print(f'Zero Area records: {(df["Area"] == 0).sum():,}')
print(f'Zero Production records: {(df["Production"] == 0).sum():,}')
print(f'Zero Yield records: {len(zero_yield_records):,}')
print(f'Zero Yield records that also have zero Production: {(zero_yield_records["Production"] == 0).sum():,}')
print('Decision: retain zero-yield records pending agricultural/domain validation; no automatic deletion occurs.')

categorical_columns = ['Crop', 'Season', 'State']
blank_category_summary = pd.Series({
    column: df[column].isna().sum() + df[column].astype(str).str.strip().eq('').sum()
    for column in categorical_columns
}, name='Blank or missing categorical values')
print(blank_category_summary.to_frame())
for column in categorical_columns:
    print(f'\n{column} unique labels: {df[column].nunique()}')
    print(df[column].value_counts(dropna=False).head(15).to_frame('Records'))

outlier_columns = [
    'Area_acres', 'Production_tonnes', 'Annual_Rainfall_mm',
    'Fertilizer_kg', 'Pesticide_kg', 'Yield_kg_per_acre'
]

iqr_results = []
for column in outlier_columns:
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    flagged_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    iqr_results.append({
        'Column': column,
        'Q1': q1,
        'Q3': q3,
        'IQR lower bound': lower_bound,
        'IQR upper bound': upper_bound,
        'IQR-screened records': int(flagged_count),
    })

iqr_summary = pd.DataFrame(iqr_results).set_index('Column')
print(iqr_summary)
print('Decision: IQR-screened values are retained. Review them with crop and unit context before any future treatment.')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for axis, column in zip(axes.flat, outlier_columns):
    sns.boxplot(y=df[column], ax=axis, color='#76B7B2')
    axis.set_title(f'{column} distribution')
    axis.set_ylabel(column)
plt.suptitle('Boxplots for Converted Numerical Fields', y=1.02)
plt.tight_layout()
plt.show()

final_columns = [
    'Crop',
    'Crop_Year',
    'Season',
    'State',
    'Area_acres',
    'Production_tonnes',
    'Annual_Rainfall_mm',
    'Fertilizer_kg',
    'Pesticide_kg',
    'Yield_kg_per_acre',
]
redundant_original_columns = [
    'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield'
]
processed_numeric_columns = [
    'Crop_Year', 'Area_acres', 'Production_tonnes', 'Annual_Rainfall_mm',
    'Fertilizer_kg', 'Pesticide_kg', 'Yield_kg_per_acre'
]

processed_df = df[final_columns].copy()
print('Redundant source columns excluded from processed output:', redundant_original_columns)
print('Retained processed columns:', processed_df.columns.tolist())
print(processed_df.head())

numeric_values = processed_df[processed_numeric_columns]
infinite_value_mask = numeric_values.notna() & ~np.isfinite(numeric_values)
infinite_value_counts = infinite_value_mask.sum()
infinite_value_total = int(infinite_value_counts.sum())
invalid_numeric_record_mask = infinite_value_mask.any(axis=1)

print(infinite_value_counts.to_frame('Infinite values'))
print(f'Total infinite numerical values: {infinite_value_total:,}')

if infinite_value_total > 0:
    invalid_numeric_records = processed_df.loc[invalid_numeric_record_mask].copy()
    print(f'Excluding {len(invalid_numeric_records):,} records containing invalid infinite numerical values.')
    print(invalid_numeric_records)
    processed_df = processed_df.loc[~invalid_numeric_record_mask].copy()
else:
    print('No infinite numerical values found; no records are excluded.')

assert not (processed_df[processed_numeric_columns].notna() & ~np.isfinite(processed_df[processed_numeric_columns])).any().any(), 'Infinite numerical values remain in processed data.'

target_column = 'Yield_kg_per_acre'
feature_columns = [
    'Crop',
    'Crop_Year',
    'Season',
    'State',
    'Area_acres',
    'Annual_Rainfall_mm',
    'Fertilizer_kg',
    'Pesticide_kg',
]

excluded_from_model_inputs = [
    'Production_tonnes',
]

print('Target:', target_column)
print('Baseline feature columns:', feature_columns)
print('Excluded model columns:', excluded_from_model_inputs)

X = processed_df[feature_columns].copy()
y = processed_df[target_column].copy()

categorical_features = ['Crop', 'Season', 'State']
numerical_features = [
    'Crop_Year', 'Area_acres', 'Annual_Rainfall_mm',
    'Fertilizer_kg', 'Pesticide_kg'
]

print('X shape:', X.shape)
print('y shape:', y.shape)
print('Categorical features:', categorical_features)
print('Numerical features:', numerical_features)
assert 'Production_tonnes' not in X.columns
assert y.notna().all(), 'Target contains missing values after target validation.'

year_distribution = processed_df['Crop_Year'].value_counts().sort_index()
print(year_distribution.to_frame('Records'))

chronological_cutoff_year = 2018
train_mask = processed_df['Crop_Year'] <= chronological_cutoff_year
test_mask = processed_df['Crop_Year'] > chronological_cutoff_year

X_train_time = X.loc[train_mask].copy()
X_test_time = X.loc[test_mask].copy()
y_train_time = y.loc[train_mask].copy()
y_test_time = y.loc[test_mask].copy()

print(f'Training years: {processed_df.loc[train_mask, "Crop_Year"].min()}–{processed_df.loc[train_mask, "Crop_Year"].max()} ({len(X_train_time):,} records)')
print(f'Testing years: {processed_df.loc[test_mask, "Crop_Year"].min()}–{processed_df.loc[test_mask, "Crop_Year"].max()} ({len(X_test_time):,} records)')
assert processed_df.loc[train_mask, 'Crop_Year'].max() < processed_df.loc[test_mask, 'Crop_Year'].min()

processed_directory = BASE_DIR / 'data' / 'processed'
processed_path = processed_directory / 'crop_yield_processed.csv'
processed_directory.mkdir(parents=True, exist_ok=True)

required_schema = [
    'Crop', 'Crop_Year', 'Season', 'State', 'Area_acres',
    'Production_tonnes', 'Annual_Rainfall_mm', 'Fertilizer_kg',
    'Pesticide_kg', 'Yield_kg_per_acre'
]
redundant_columns_present = set(redundant_original_columns).intersection(processed_df.columns)
processed_duplicate_count = int(processed_df.duplicated().sum())
processed_coconut_count = int(
    processed_df['Crop'].astype(str).str.strip().str.casefold().eq('coconut').sum()
)

assert processed_df.shape[1] == 10, 'Processed output must contain exactly 10 columns.'
assert processed_df.columns.tolist() == required_schema, 'Processed columns do not match the required schema/order.'
assert not processed_df.columns.duplicated().any(), 'Duplicate column names found.'
assert processed_duplicate_count == 0, 'Exact duplicate rows remain in processed output.'
assert processed_coconut_count == 0, 'Coconut records remain in processed output.'
assert all(pd.api.types.is_numeric_dtype(processed_df[column]) for column in processed_numeric_columns), 'A required numerical column is not numeric.'
assert not (processed_df[processed_numeric_columns].notna() & ~np.isfinite(processed_df[processed_numeric_columns])).any().any(), 'Infinite numerical values remain.'
assert processed_df['Yield_kg_per_acre'].notna().all(), 'Target contains missing values.'
assert not redundant_columns_present, f'Redundant columns remain: {redundant_columns_present}'
assert processed_path != raw_path.resolve(), 'Processed output path must not be the raw dataset path.'

print('Final validation before export: PASSED')
print(f'Final dataset shape: {processed_df.shape}')
print('Final columns:', processed_df.columns.tolist())
print(processed_df.isna().sum().to_frame('Missing values'))
print(processed_df['Yield_kg_per_acre'].describe().to_frame('Yield_kg_per_acre'))

processed_df.to_csv(processed_path, index=False)
print(f'Processed dataset saved to: {processed_path}')

saved_df = pd.read_csv(processed_path)
saved_redundant_columns = set(redundant_original_columns).intersection(saved_df.columns)
saved_coconut_count = int(saved_df['Crop'].astype(str).str.strip().str.casefold().eq('coconut').sum())

assert saved_df.shape[1] == 10, 'Saved CSV does not contain exactly 10 columns.'
assert saved_df.columns.tolist() == required_schema, 'Saved CSV schema/order does not match requirements.'
assert not saved_redundant_columns, f'Saved CSV contains redundant columns: {saved_redundant_columns}'
assert saved_coconut_count == 0, 'Saved CSV contains Coconut records.'
assert saved_df['Yield_kg_per_acre'].notna().all(), 'Saved CSV target contains missing values.'

print('Post-save read-back validation: PASSED')
print(f'Saved CSV shape: {saved_df.shape}')

final_missing_values = int(processed_df.isna().sum().sum())
final_record_count = len(processed_df)

summary = pd.Series({
    'Original record count': original_record_count,
    'Coconut records removed': coconut_records_removed,
    'Duplicate records removed': duplicate_records_removed,
    'Missing values remaining': final_missing_values,
    'Final record count': final_record_count,
    'Final column count': processed_df.shape[1],
    'Target name': target_column,
    'Training year range': f'{processed_df.loc[train_mask, "Crop_Year"].min()}–{processed_df.loc[train_mask, "Crop_Year"].max()}',
    'Testing year range': f'{processed_df.loc[test_mask, "Crop_Year"].min()}–{processed_df.loc[test_mask, "Crop_Year"].max()}',
})
print(summary.to_frame('Value'))

units = pd.Series({
    'Area_acres': 'acres',
    'Production_tonnes': 'metric tonnes',
    'Annual_Rainfall_mm': 'millimeters',
    'Fertilizer_kg': 'kilograms',
    'Pesticide_kg': 'kilograms (working assumption)',
    'Yield_kg_per_acre': 'kilograms per acre (working assumption: original Yield is t/ha)',
})
print(units.to_frame('Final working unit'))
print('Final feature names:', feature_columns)
