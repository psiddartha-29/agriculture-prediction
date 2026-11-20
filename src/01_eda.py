from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style='whitegrid', context='notebook')
pd.set_option('display.max_columns', None)

BASE_DIR = Path(__file__).resolve().parents[1]  # project root (PS1/)
data_path = BASE_DIR / 'data' / 'raw' / 'crop_yield.csv'
if not data_path.exists():
    raise FileNotFoundError(f'Could not find raw dataset at {data_path}')
df = pd.read_csv(data_path)
print(f'Loaded: {data_path}')

print(f'Rows: {df.shape[0]:,}')
print(f'Columns: {df.shape[1]}')

print(df.head())
print(df.tail())

print(pd.DataFrame({'Column': df.columns, 'Data type': df.dtypes.astype(str).values}))
df.info()

missing_values = df.isna().sum().sort_values(ascending=False)
print(missing_values.to_frame('Missing values'))

print(f'Exact duplicate rows: {df.duplicated().sum():,}')

print(df.describe(include='all').T)

crop_counts = df['Crop'].value_counts()
print(f'Unique crop types: {df["Crop"].nunique()}')
print(crop_counts.to_frame('Records'))

state_counts = df['State'].value_counts()
print(f'Unique states: {df["State"].nunique()}')
print(state_counts.to_frame('Records'))

season_counts = df['Season'].value_counts()
print(f'Unique season labels: {df["Season"].nunique()}')
print(season_counts.to_frame('Records'))
print('Labels are displayed as stored; no whitespace is normalized.')

year_counts = df['Crop_Year'].value_counts().sort_index()
print(year_counts.to_frame('Records'))
print(f'Year range: {df["Crop_Year"].min()} to {df["Crop_Year"].max()}')

crop_yield_summary = df.groupby('Crop')['Yield'].agg(['count', 'mean', 'median', 'min', 'max']).sort_values('median', ascending=False)
print(crop_yield_summary)

state_yield_summary = df.groupby('State')['Yield'].agg(['count', 'mean', 'median', 'min', 'max']).sort_values('median', ascending=False)
print(state_yield_summary)

print(df['Annual_Rainfall'].describe().to_frame('Annual_Rainfall'))

print(df[['Fertilizer', 'Pesticide']].describe().T)

numeric_columns = df.select_dtypes(include='number').columns
zero_counts = (df[numeric_columns] == 0).sum()
negative_counts = (df[numeric_columns] < 0).sum()
outlier_counts = {}
for column in numeric_columns:
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_counts[column] = ((df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)).sum()
screening = pd.DataFrame({'Zero values': zero_counts, 'Negative values': negative_counts, 'IQR-screened values': pd.Series(outlier_counts)})
print(screening)
print('Confirm definitions in dataset documentation before treating flags as invalid data.')

correlation = df[numeric_columns].corr()
print(correlation)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
sns.countplot(data=df, x='Crop_Year', ax=axes[0, 0], color='#4C78A8')
axes[0, 0].set_title('Records by Crop Year'); axes[0, 0].tick_params(axis='x', rotation=45)
sns.histplot(data=df, x='Annual_Rainfall', bins=30, kde=True, ax=axes[0, 1], color='#59A14F')
axes[0, 1].set_title('Annual Rainfall Distribution')
sns.histplot(data=df, x='Fertilizer', bins=30, ax=axes[1, 0], color='#F28E2B')
axes[1, 0].set_title('Fertilizer Distribution')
sns.histplot(data=df, x='Pesticide', bins=30, ax=axes[1, 1], color='#E15759')
axes[1, 1].set_title('Pesticide Distribution')
plt.tight_layout(); plt.show()

top_crops = crop_yield_summary.head(20).index
plt.figure(figsize=(12, 8))
sns.boxplot(data=df[df['Crop'].isin(top_crops)], y='Crop', x='Yield', order=top_crops, color='#76B7B2')
plt.title('Yield Distribution for 20 Crops with Highest Median Yield')
plt.xlabel('Yield (unit not assumed)'); plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Numerical Variable Correlation Matrix'); plt.tight_layout(); plt.show()

telangana_mask = df['State'].astype(str).str.strip().str.casefold().eq('telangana')
telangana_df = df.loc[telangana_mask].copy()
print(f'Telangana records: {len(telangana_df):,}')
if not telangana_df.empty:
    print(telangana_df.describe(include='all').T)
    print(telangana_df.groupby('Crop')['Yield'].agg(['count', 'mean', 'median', 'min', 'max']).sort_values('median', ascending=False))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(data=telangana_df, x='Crop_Year', ax=axes[0], color='#4C78A8')
    axes[0].set_title('Telangana Records by Year'); axes[0].tick_params(axis='x', rotation=45)
    sns.histplot(data=telangana_df, x='Yield', bins=25, kde=True, ax=axes[1], color='#59A14F')
    axes[1].set_title('Telangana Yield Distribution'); axes[1].set_xlabel('Yield (unit not assumed)')
    plt.tight_layout(); plt.show()

df['Calculated_Yield'] = (
    df['Production'] / df['Area']
)

df['Yield_Difference'] = (
    df['Yield'] - df['Calculated_Yield']
)

print(
    df[['Crop', 'Area', 'Production',
        'Yield', 'Calculated_Yield',
        'Yield_Difference']].head(15)
)

print(
    df['Yield'].corr(df['Calculated_Yield'])
)

df['Calculated_Yield'] = df['Production'] / df['Area']

df['Yield_Difference'] = (
    df['Yield'] - df['Calculated_Yield']
)

print(
    df[['Crop', 'Area', 'Production',
        'Yield', 'Calculated_Yield',
        'Yield_Difference']].head(15)
)

print(
    "Correlation:",
    df['Yield'].corr(df['Calculated_Yield'])
)

print(
    "Mean Absolute Difference:",
    df['Yield_Difference'].abs().mean()
)
