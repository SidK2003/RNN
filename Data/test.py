import pandas as pd

df = pd.read_csv("Data/historical_data/reliance_daily.csv")

print(df.columns)
print(df.head())