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
# 6) Résumé chiffré final
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