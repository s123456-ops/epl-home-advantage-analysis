import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import os

# =========================================================
# 0) Setup
# =========================================================
df = pd.read_csv("data/processed/epl_strength.csv")

os.makedirs("outputs/figures", exist_ok=True)

# Style global
sns.set_theme(style="whitegrid", palette="muted")
COLORS = {"Home": "#1a78cf", "Draw": "#f0a500", "Away": "#e0393e"}

print("Dataset loaded — shape:", df.shape)
print(df.head(3))


# =========================================================
# 1) Résultats globaux : Home / Draw / Away
# =========================================================
result_counts = df["FTR"].value_counts().reindex(["H", "D", "A"])
result_pct    = result_counts / len(df) * 100

print("\n--- Résultats globaux ---")
print(result_pct.rename({"H": "Victoire domicile", "D": "Match nul", "A": "Victoire extérieur"}).round(1))

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(
    ["Victoire\ndomicile", "Match nul", "Victoire\nextérieur"],
    result_pct.values,
    color=[COLORS["Home"], COLORS["Draw"], COLORS["Away"]],
    edgecolor="white", width=0.5
)
for bar, val in zip(bars, result_pct.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.set_ylabel("% des matchs")
ax.set_title("Répartition des résultats en Premier League (1993–2023)", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, 60)
plt.tight_layout()
plt.savefig("outputs/figures/01_resultats_globaux.png", dpi=150)
plt.close()
print("→ Figure 1 sauvegardée")


# =========================================================
# 2) Évolution du taux de victoire domicile par saison
# =========================================================
season_stats = df.groupby("Season_End_Year").agg(
    HomeWinRate=("HomeWin", "mean"),
    NbMatchs=("HomeWin", "count")
).reset_index()

print("\n--- Taux de victoire domicile par saison ---")
print(season_stats.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(season_stats["Season_End_Year"], season_stats["HomeWinRate"] * 100,
        color=COLORS["Home"], linewidth=2.5, marker="o", markersize=5, label="Taux victoire domicile")

# Mise en évidence des saisons Covid
for yr in [2020, 2021]:
    ax.axvspan(yr - 0.5, yr + 0.5, color="lightyellow", alpha=0.8, zorder=0)
ax.axvspan(2019.5, 2021.5, color="lightyellow", alpha=0.5, zorder=0, label="Saisons Covid (sans public)")

# Ligne de référence = moyenne globale
global_mean = df["HomeWin"].mean() * 100
ax.axhline(global_mean, color="grey", linestyle="--", linewidth=1.2, label=f"Moyenne globale ({global_mean:.1f}%)")

ax.set_xlabel("Saison (année de fin)")
ax.set_ylabel("% victoires domicile")
ax.set_title("Évolution du taux de victoire à domicile par saison", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(25, 65)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/02_evolution_homewin.png", dpi=150)
plt.close()
print("→ Figure 2 sauvegardée")


# =========================================================
# 3) Effet Covid : comparaison avant / pendant / après
# =========================================================
df["Periode"] = pd.cut(
    df["Season_End_Year"],
    bins=[1992, 2019, 2021, 2023],
    labels=["Avant Covid\n(1993–2019)", "Covid\n(2020–2021)", "Après Covid\n(2022–2023)"]
)

covid_stats = df.groupby("Periode", observed=True).agg(
    HomeWinRate=("HomeWin", "mean"),
    AvgHomeGoals=("HomeGoals", "mean"),
    AvgAwayGoals=("AwayGoals", "mean"),
    NbMatchs=("HomeWin", "count")
).reset_index()

print("\n--- Effet Covid ---")
print(covid_stats.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Sous-graphe 1 : taux de victoire domicile
colors_periode = ["#1a78cf", "#e0393e", "#2ca02c"]
axes[0].bar(covid_stats["Periode"], covid_stats["HomeWinRate"] * 100,
            color=colors_periode, edgecolor="white", width=0.5)
for i, (_, row) in enumerate(covid_stats.iterrows()):
    axes[0].text(i, row["HomeWinRate"] * 100 + 0.5,
                 f"{row['HomeWinRate']*100:.1f}%", ha="center", fontweight="bold", fontsize=11)
axes[0].set_ylabel("% victoires domicile")
axes[0].set_title("Taux de victoire à domicile", fontweight="bold")
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[0].set_ylim(0, 60)

# Sous-graphe 2 : buts moyens home vs away
x = np.arange(len(covid_stats))
width = 0.35
axes[1].bar(x - width/2, covid_stats["AvgHomeGoals"], width, label="Buts domicile", color=COLORS["Home"], edgecolor="white")
axes[1].bar(x + width/2, covid_stats["AvgAwayGoals"], width, label="Buts extérieur", color=COLORS["Away"], edgecolor="white")
axes[1].set_xticks(x)
axes[1].set_xticklabels(covid_stats["Periode"])
axes[1].set_ylabel("Buts moyens par match")
axes[1].set_title("Buts domicile vs extérieur", fontweight="bold")
axes[1].legend()
axes[1].set_ylim(0, 2.5)

fig.suptitle("Impact des saisons Covid sur l'avantage domicile", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/03_effet_covid.png", dpi=150)
plt.close()
print("→ Figure 3 sauvegardée")


# =========================================================
# 4) Buts : distribution domicile vs extérieur
# =========================================================
print("\n--- Statistiques sur les buts ---")
print(df[["HomeGoals", "AwayGoals", "TotalGoals", "GoalDiff"]].describe().round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogramme buts par match (home vs away)
max_goals = int(df[["HomeGoals", "AwayGoals"]].max().max())
bins = range(0, max_goals + 2)
axes[0].hist(df["HomeGoals"], bins=bins, alpha=0.7, color=COLORS["Home"],
             label=f"Domicile (moy. {df['HomeGoals'].mean():.2f})", density=True, align="left")
axes[0].hist(df["AwayGoals"], bins=bins, alpha=0.7, color=COLORS["Away"],
             label=f"Extérieur (moy. {df['AwayGoals'].mean():.2f})", density=True, align="left")
axes[0].set_xlabel("Buts marqués")
axes[0].set_ylabel("Fréquence relative")
axes[0].set_title("Distribution des buts par match", fontweight="bold")
axes[0].legend()

# Distribution de GoalDiff
axes[1].hist(df["GoalDiff"], bins=range(-9, 10), color="#7f7f7f", edgecolor="white", alpha=0.85, density=True, align="left")
axes[1].axvline(0, color="black", linestyle="--", linewidth=1.2)
axes[1].axvline(df["GoalDiff"].mean(), color=COLORS["Home"], linestyle="-", linewidth=2,
                label=f"Moyenne ({df['GoalDiff'].mean():.2f})")
axes[1].set_xlabel("Différence de buts (domicile − extérieur)")
axes[1].set_ylabel("Fréquence relative")
axes[1].set_title("Distribution de la différence de buts", fontweight="bold")
axes[1].legend()

fig.suptitle("Analyse des buts en Premier League (1993–2023)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/04_buts_distribution.png", dpi=150)
plt.close()
print("→ Figure 4 sauvegardée")


# =========================================================
# 5) StrengthDiff et HomeWin : l'avantage domicile tient-il
#    même quand l'équipe visiteuse est meilleure ?
# =========================================================

# Déciles de StrengthDiff
df["StrengthDiff_bin"] = pd.qcut(df["StrengthDiff"], q=5, labels=[
    "Très défavorable", "Défavorable", "Équilibré", "Favorable", "Très favorable"
])

strength_stats = df.groupby("StrengthDiff_bin", observed=True)["HomeWin"].mean().reset_index()
strength_stats.columns = ["Contexte (force relative)", "HomeWinRate"]

print("\n--- HomeWin par quintile de StrengthDiff ---")
print(strength_stats.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = ["#e0393e", "#f0a500", "#aec7e8", "#1f77b4", "#0a4c8f"]
bars = ax.bar(strength_stats["Contexte (force relative)"],
              strength_stats["HomeWinRate"] * 100,
              color=bar_colors, edgecolor="white", width=0.55)
for bar, val in zip(bars, strength_stats["HomeWinRate"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val*100:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
ax.axhline(df["HomeWin"].mean() * 100, color="grey", linestyle="--", linewidth=1.2,
           label=f"Moyenne globale ({df['HomeWin'].mean()*100:.1f}%)")
ax.set_ylabel("% victoires domicile")
ax.set_title("Taux de victoire domicile selon le niveau relatif des équipes", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, 75)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/05_homewin_par_strength.png", dpi=150)
plt.close()
print("→ Figure 5 sauvegardée")


# =========================================================
# 6) Top équipes avec le meilleur avantage domicile
# Critère : différence entre taux de victoire HOME et AWAY
# (uniquement équipes ayant joué au moins 3 saisons en PL)
# =========================================================

# Nombre de saisons disputées par équipe
seasons_per_team = df.groupby("HomeTeam")["Season_End_Year"].nunique()
teams_3plus = seasons_per_team[seasons_per_team >= 3].index

# Taux victoire à domicile (en tant qu'équipe domicile)
home_win_rate = (
    df[df["HomeTeam"].isin(teams_3plus)]
    .groupby("HomeTeam")["HomeWin"].mean()
    .rename("HomeWinRate")
)

# Taux victoire à l'extérieur (quand cette équipe joue away)
df["AwayWin"] = (df["FTR"] == "A").astype(int)
away_win_rate = (
    df[df["AwayTeam"].isin(teams_3plus)]
    .groupby("AwayTeam")["AwayWin"].mean()
    .rename("AwayWinRate")
)

# Avantage domicile = différence entre les deux taux
team_advantage = pd.concat([home_win_rate, away_win_rate], axis=1).dropna()
team_advantage["HomeAdvantage"] = team_advantage["HomeWinRate"] - team_advantage["AwayWinRate"]
team_advantage = team_advantage.sort_values("HomeAdvantage", ascending=False)

top_n    = 15
top_teams = team_advantage.head(top_n).reset_index()
top_teams.columns = ["Team", "HomeWinRate", "AwayWinRate", "HomeAdvantage"]

print("\n--- Top 15 équipes : avantage domicile ---")
print(top_teams[["Team", "HomeWinRate", "AwayWinRate", "HomeAdvantage"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(top_n)
ax.barh(y, top_teams["HomeWinRate"] * 100, height=0.4,
        color=COLORS["Home"], label="Victoires à domicile", align="center")
ax.barh(y - 0.4, top_teams["AwayWinRate"] * 100, height=0.4,
        color=COLORS["Away"], label="Victoires à l'extérieur", align="center")
ax.set_yticks(y - 0.2)
ax.set_yticklabels(top_teams["Team"], fontsize=9)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_xlabel("Taux de victoire")
ax.set_title(f"Top {top_n} équipes : écart victoires domicile vs extérieur\n(équipes ayant joué ≥ 3 saisons en PL)",
             fontsize=12, fontweight="bold")
ax.legend(loc="lower right")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/figures/06_top_teams_home_advantage.png", dpi=150)
plt.close()
print("→ Figure 6 sauvegardée")


# =========================================================
# 7) Big Six : comparaison domicile vs extérieur
# Pour chaque club : taux victoire home/away + buts moyens
# =========================================================

BIG_SIX = ["Arsenal", "Chelsea", "Liverpool",
           "Manchester City", "Manchester Utd", "Tottenham"]

# Résultats quand le Big Six joue à domicile
big6_home = (
    df[df["HomeTeam"].isin(BIG_SIX)]
    .groupby("HomeTeam")
    .agg(
        HomeWinRate=("HomeWin", "mean"),
        AvgHomeGoals=("HomeGoals", "mean"),
        AvgConcededHome=("AwayGoals", "mean")
    )
    .reset_index()
    .rename(columns={"HomeTeam": "Team"})
)

# Résultats quand le Big Six joue à l'extérieur
big6_away = (
    df[df["AwayTeam"].isin(BIG_SIX)]
    .groupby("AwayTeam")
    .agg(
        AwayWinRate=("AwayWin", "mean"),
        AvgAwayGoals=("AwayGoals", "mean"),
        AvgConcededAway=("HomeGoals", "mean")
    )
    .reset_index()
    .rename(columns={"AwayTeam": "Team"})
)

big6 = big6_home.merge(big6_away, on="Team")
big6 = big6.sort_values("HomeWinRate", ascending=False).reset_index(drop=True)

print("\n--- Big Six : domicile vs extérieur ---")
print(big6[["Team", "HomeWinRate", "AwayWinRate", "AvgHomeGoals", "AvgAwayGoals"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(BIG_SIX))
width = 0.35

# Sous-graphe 1 : taux de victoire
axes[0].bar(x - width/2, big6["HomeWinRate"] * 100, width,
            color=COLORS["Home"], label="À domicile", edgecolor="white")
axes[0].bar(x + width/2, big6["AwayWinRate"] * 100, width,
            color=COLORS["Away"], label="À l'extérieur", edgecolor="white")
for i, row in big6.iterrows():
    axes[0].text(i - width/2, row["HomeWinRate"] * 100 + 0.5,
                 f"{row['HomeWinRate']*100:.0f}%", ha="center", fontsize=8, fontweight="bold")
    axes[0].text(i + width/2, row["AwayWinRate"] * 100 + 0.5,
                 f"{row['AwayWinRate']*100:.0f}%", ha="center", fontsize=8, fontweight="bold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(big6["Team"], rotation=20, ha="right", fontsize=9)
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[0].set_ylabel("Taux de victoire")
axes[0].set_title("Taux de victoire", fontweight="bold")
axes[0].legend()
axes[0].set_ylim(0, 85)

# Sous-graphe 2 : buts marqués moyens
axes[1].bar(x - width/2, big6["AvgHomeGoals"], width,
            color=COLORS["Home"], label="Buts marqués domicile", edgecolor="white")
axes[1].bar(x + width/2, big6["AvgAwayGoals"], width,
            color=COLORS["Away"], label="Buts marqués extérieur", edgecolor="white")
for i, row in big6.iterrows():
    axes[1].text(i - width/2, row["AvgHomeGoals"] + 0.02,
                 f"{row['AvgHomeGoals']:.2f}", ha="center", fontsize=8, fontweight="bold")
    axes[1].text(i + width/2, row["AvgAwayGoals"] + 0.02,
                 f"{row['AvgAwayGoals']:.2f}", ha="center", fontsize=8, fontweight="bold")
axes[1].set_xticks(x)
axes[1].set_xticklabels(big6["Team"], rotation=20, ha="right", fontsize=9)
axes[1].set_ylabel("Buts moyens par match")
axes[1].set_title("Buts marqués en moyenne", fontweight="bold")
axes[1].legend()
axes[1].set_ylim(0, 3.0)

fig.suptitle("Big Six — Domicile vs Extérieur (1993–2023)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/07_big_six_home_vs_away.png", dpi=150)
plt.close()
print("→ Figure 7 sauvegardée")


# =========================================================
# 8) Résumé chiffré final
# =========================================================
print("\n" + "="*55)
print("RÉSUMÉ DESCRIPTIF — AVANTAGE DOMICILE EN PL 1993–2023")
print("="*55)
print(f"Nombre de matchs analysés      : {len(df):,}")
print(f"Saisons couvertes               : {df['Season_End_Year'].min()}–{df['Season_End_Year'].max()}")
print(f"Taux victoire domicile (global) : {df['HomeWin'].mean()*100:.1f}%")
print(f"Taux nul (global)               : {(df['FTR']=='D').mean()*100:.1f}%")
print(f"Taux victoire extérieur (global): {(df['FTR']=='A').mean()*100:.1f}%")
print(f"Buts moyens domicile/match      : {df['HomeGoals'].mean():.2f}")
print(f"Buts moyens extérieur/match     : {df['AwayGoals'].mean():.2f}")
print(f"Différence de buts moyenne      : {df['GoalDiff'].mean():.2f}")
print(f"Taux victoire domicile (Covid)  : {df[df['Covid']==1]['HomeWin'].mean()*100:.1f}%")
print(f"Taux victoire domicile (normal) : {df[df['Covid']==0]['HomeWin'].mean()*100:.1f}%")
print("="*55)
print("\nTous les graphiques ont été sauvegardés dans outputs/figures/")