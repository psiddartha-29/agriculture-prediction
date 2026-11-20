from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]  # project root (PS1/)
df = pd.read_csv(BASE_DIR / "data" / "raw" / "crop_yield.csv")
print(df.head())

x=df.drop(['Yield','Production','Pesticide'], axis=1)
print(x.head(5))

output_path = BASE_DIR / "data" / "processed" / "Fertilizer.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
x.to_csv(output_path, index=False)

y=df.drop(['Yield','Production','Fertilizer'], axis=1)
print(y.head(5))

pesticide_path = output_path.with_name("Pesticide.csv")
pesticide_path.parent.mkdir(parents=True, exist_ok=True)
y.to_csv(pesticide_path, index=False)
