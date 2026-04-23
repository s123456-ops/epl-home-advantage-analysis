import os
import pandas as pd

# =========================================================
# 1) Loading the raw dataset
# =========================================================
df = pd.read_csv("data/raw/premier-league-matches.csv")

# Sort chronologically — needed for rolling strength calculation
df = df.sort_values(["Season_End_Year", "Wk", "Date"]).reset_index(drop=True)

print("\nDataset loaded successfully")
print(df.head(10))
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nMissing values:")
print(df.isna().sum())


# =========================================================
# 2) Renaming columns so they are easier to work with ! 
# =========================================================
df = df.rename(columns={
    "Home": "HomeTeam",
    "Away": "AwayTeam",
    "HomeGoals": "HomeGoals",   # kept explicit for clarity
    "AwayGoals": "AwayGoals"
})

# Robustness: handle both possible input column names
# Some datasets use "FTHG" (Full Time Home Goals) and "FTAG" (Full Time Away Goals) instead of "HomeGoals" and "AwayGoals"
if "HomeGoals" not in df.columns and "FTHG" in df.columns:
    df = df.rename(columns={"FTHG": "HomeGoals", "FTAG": "AwayGoals"})
elif "HomeGoals" not in df.columns:
    # Fallback: original raw column names
    df = df.rename(columns={
        "Home": "HomeTeam",
        "Away": "AwayTeam",
    })
    # HomeGoals / AwayGoals already named correctly in raw

print("\nRenamed columns:")
print(df.columns)


# =========================================================
# 3) Assigning points based on match result for each team
# Win = 3 points, draw = 1 point, loss = 0 points
# =========================================================
def get_points(row, team):
    if row["FTR"] == "H":
        return 3 if team == row["HomeTeam"] else 0
    elif row["FTR"] == "A":
        return 3 if team == row["AwayTeam"] else 0
    else:
        return 1


# =========================================================
# 4) Building a team-level table from home matches
# =========================================================
home_df = df[["Season_End_Year", "Wk", "HomeTeam", "AwayTeam", "FTR"]].copy()
home_df["Team"] = home_df["HomeTeam"]
home_df["Points"] = home_df.apply(
    lambda row: get_points(row, row["HomeTeam"]),
    axis=1
)


# =========================================================
# 5) Building the same thing for away matches !
# =========================================================
away_df = df[["Season_End_Year", "Wk", "HomeTeam", "AwayTeam", "FTR"]].copy()
away_df["Team"] = away_df["AwayTeam"]
away_df["Points"] = away_df.apply(
    lambda row: get_points(row, row["AwayTeam"]),
    axis=1
)


# =========================================================
# 6) Putting both tables together
# =========================================================
teams_df = pd.concat([
    home_df[["Season_End_Year", "Wk", "Team", "Points"]],
    away_df[["Season_End_Year", "Wk", "Team", "Points"]]
]).sort_values(["Team", "Season_End_Year", "Wk"]).reset_index(drop=True)

print("\nCombined team-level dataset:")
print(teams_df.head())


# =========================================================
# 7) Computing team strength by season
# Rolling average points over last 5 matches (before current match)
# =========================================================
WINDOW = 5

teams_df["Strength"] = (
    teams_df
    .groupby("Team")["Points"]
    .transform(lambda x: x.shift(1).rolling(window=WINDOW, min_periods=1).mean())
)

# Filling NaN (first match per team) with season average points to avoid losing data
season_avg = teams_df.groupby("Season_End_Year")["Points"].transform("mean")
teams_df["Strength"] = teams_df["Strength"].fillna(season_avg)
# =========================================================
# 8) Merging home team strength back into the main dataset
# =========================================================
df = df.merge(
    teams_df[["Season_End_Year", "Wk", "Team", "Strength"]],
    left_on=["Season_End_Year", "Wk", "HomeTeam"],
    right_on=["Season_End_Year", "Wk", "Team"],
    how="left"
).rename(columns={"Strength": "HomeStrength"}).drop(columns=["Team"])


# =========================================================
# 9) Merging away team strength into the main dataset
# =========================================================
df = df.merge(
    teams_df[["Season_End_Year", "Wk", "Team", "Strength"]],
    left_on=["Season_End_Year", "Wk", "AwayTeam"],
    right_on=["Season_End_Year", "Wk", "Team"],
    how="left"
).rename(columns={"Strength": "AwayStrength"}).drop(columns=["Team"])


# =========================================================
# 10) Create the strength difference variable
# Positive = home team stronger on paper
# Negative = away team stronger
# This variable will be key for our analysis of home advantage and Covid impact ! 
# =========================================================
df["StrengthDiff"] = df["HomeStrength"] - df["AwayStrength"]

print("\nPreview of strength variables:")
print(df[["HomeTeam", "AwayTeam", "HomeStrength", "AwayStrength", "StrengthDiff"]].head())

# Just checking for any missing values in strength variables
print("\nMissing values in strength variables:")
print(df[["HomeStrength", "AwayStrength", "StrengthDiff"]].isnull().sum())

# =========================================================
# 11) Creating target and control variables
# =========================================================
df["HomeWin"]    = (df["FTR"] == "H").astype(int)
df["Covid"]      = df["Season_End_Year"].isin([2020, 2021]).astype(int)

# --- NEW: goal variables ---
df["TotalGoals"] = df["HomeGoals"] + df["AwayGoals"]   # total goals per match
df["GoalDiff"]   = df["HomeGoals"] - df["AwayGoals"]   # positive = home advantage in goals

print("\nSummary statistics for strength variables:")
print(df[["HomeStrength", "AwayStrength", "StrengthDiff"]].describe())

print("\nSummary statistics for goal variables:")
print(df[["HomeGoals", "AwayGoals", "TotalGoals", "GoalDiff"]].describe())

print("\nHome win rate (overall):",    df["HomeWin"].mean().round(3))
print("Home win rate (Covid):",       df[df["Covid"] == 1]["HomeWin"].mean().round(3))
print("Home win rate (non-Covid):",   df[df["Covid"] == 0]["HomeWin"].mean().round(3))

print("\nAvg goals at home:",  df["HomeGoals"].mean().round(3))
print("Avg goals away:",       df["AwayGoals"].mean().round(3))
print("Avg total goals/game:", df["TotalGoals"].mean().round(3))


# =========================================================
# 12) Final column order (clean and readable)
# =========================================================
df = df[[
    "Season_End_Year", "Wk", "Date",
    "HomeTeam", "AwayTeam", "FTR",
    "HomeGoals", "AwayGoals", "TotalGoals", "GoalDiff",
    "HomeStrength", "AwayStrength", "StrengthDiff",
    "HomeWin", "Covid"
]]


# =========================================================
# 13) The dataset is processed and ready to be saved for analysis !
# =========================================================
output_path = "data/processed/epl_strength.csv"

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"\nDataset saved successfully at: {output_path}")
print(f"Final shape: {df.shape}")
print(f"\nFinal columns:\n{list(df.columns)}")