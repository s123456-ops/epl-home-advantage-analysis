import os
import pandas as pd

print("script started")
print("working directory:", os.getcwd())

df = pd.read_csv("data/raw/premier-league-matches.csv")

print("dataset loaded successfully")
print(df.head())