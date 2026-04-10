import os
import pandas as pd

# =========================================================
# 1) Load the raw dataset
# =========================================================
df = pd.read_csv("data/raw/premier-league-matches.csv")

print("\nDataset loaded successfully")
print(df.head(10))
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nMissing values:")
print(df.isna().sum())


# =========================================================
# 2) Clean column names
# Just making them more standard and easier to work with
# =========================================================
df = df.rename(columns={
    "Home": "HomeTeam",
    "Away": "AwayTeam",
    "HomeGoals": "FTHG",
    "AwayGoals": "FTAG"
})

print("\nRenamed columns:")
print(df.columns)


# =========================================================
# 3) Function to assign points
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
# 4) Build a team-level table from home matches
# Each row represents the points earned by the home team
# =========================================================
home_df = df[["Season_End_Year", "HomeTeam", "AwayTeam", "FTR"]].copy()
home_df["Team"] = home_df["HomeTeam"]
home_df["Points"] = home_df.apply(
    lambda row: get_points(row, row["HomeTeam"]),
    axis=1
)


# =========================================================
# 5) Build the same thing for away matches
# This time each row represents the away team's points
# =========================================================
away_df = df[["Season_End_Year", "HomeTeam", "AwayTeam", "FTR"]].copy()
away_df["Team"] = away_df["AwayTeam"]
away_df["Points"] = away_df.apply(
    lambda row: get_points(row, row["AwayTeam"]),
    axis=1
)


# =========================================================
# 6) Put both tables together
# Now we have one dataset with all teams and all points won
# =========================================================
teams_df = pd.concat([
    home_df[["Season_End_Year", "Team", "Points"]],
    away_df[["Season_End_Year", "Team", "Points"]]
])

print("\nCombined team-level dataset:")
print(teams_df.head())


# =========================================================
# 7) Compute team strength by season
# Here, strength = average points per match
# =========================================================
team_strength = teams_df.groupby(["Season_End_Year", "Team"]).agg(
    TotalPoints=("Points", "sum"),
    Matches=("Points", "count")
).reset_index()

team_strength["Strength"] = team_strength["TotalPoints"] / team_strength["Matches"]

print("\nTeam strength by season:")
print(team_strength.head())


# =========================================================
# 8) Merge home team strength back into the main dataset
# =========================================================
df = df.merge(
    team_strength[["Season_End_Year", "Team", "Strength"]],
    left_on=["Season_End_Year", "HomeTeam"],
    right_on=["Season_End_Year", "Team"],
    how="left"
).rename(columns={"Strength": "HomeStrength"}).drop(columns=["Team"])


# =========================================================
# 9) Merge away team strength into the main dataset
# =========================================================
df = df.merge(
    team_strength[["Season_End_Year", "Team", "Strength"]],
    left_on=["Season_End_Year", "AwayTeam"],
    right_on=["Season_End_Year", "Team"],
    how="left"
).rename(columns={"Strength": "AwayStrength"}).drop(columns=["Team"])


# =========================================================
# 10) Create the strength difference variable
# Positive value = home team is stronger on paper
# Negative value = away team is stronger
# =========================================================
df["StrengthDiff"] = df["HomeStrength"] - df["AwayStrength"]

print("\nPreview of strength variables:")
print(df[["HomeTeam", "AwayTeam", "HomeStrength", "AwayStrength", "StrengthDiff"]].head())


# =========================================================
# 11) Final check on the new variables
# Making sure everything looks normal!
# =========================================================
print("\nSummary statistics for strength variables:")
print(df[["HomeStrength", "AwayStrength", "StrengthDiff"]].describe())


# =========================================================
# 12) Save final dataset (ready for analysis/modeling)
# =========================================================

output_path = "data/processed/epl_strength.csv"

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"\nDataset saved successfully at: {output_path}")