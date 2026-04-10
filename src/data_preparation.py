import os
import pandas as pd

print("script started")
print("working directory:", os.getcwd())

df = pd.read_csv("data/raw/premier-league-matches.csv")

print("dataset loaded successfully!")
print(df.head(10))
print("Shape:", df.shape)
print(df.columns)
print(df.isna().sum())

df = df.rename(columns={
    "Home": "HomeTeam",
    "Away": "AwayTeam",
    "HomeGoals": "FTHG",
    "AwayGoals": "FTAG"
})

print(df.columns)

def get_points(row, team):
    if row["FTR"] == "H":
        return 3 if team == row["HomeTeam"] else 0
    elif row["FTR"] == "A":
        return 3 if team == row["AwayTeam"] else 0
    else:
        return 1