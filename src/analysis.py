import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

# =========================================================
# 0) SETUP — Identité visuelle Premier League
# =========================================================
df = pd.read_csv("data/processed/epl_strength.csv")
os.makedirs("outputs/figures", exist_ok=True)

# Palette officielle PL
PL = {
    "bg":     "#37003C",   # fond violet foncé signature
    "green":  "#00FF87",   # accent vert menthe (domicile)
    "pink":   "#E90052",   # accent rose/rouge (extérieur)
    "purple": "#963CFF",   # violet accent
    "white":  "#FFFFFF",
    "grey":   "#7A7A8C",   # texte secondaire
    "grid":   "#FFFFFF15", # grille discrète
    "spine":  "#FFFFFF22", # bordures axes
}

def pl_theme(fig, axes, title, subtitle=None):
    """Applique le thème visuel Premier League à une figure."""
    fig.patch.set_facecolor(PL["bg"])
    for ax in axes:
        ax.set_facecolor(PL["bg"])
        ax.tick_params(colors=PL["white"], labelsize=9)
        ax.xaxis.label.set_color(PL["white"])
        ax.yaxis.label.set_color(PL["white"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PL["spine"])
        ax.grid(True, color=PL["grid"], linewidth=0.5, linestyle="--")
        ax.set_axisbelow(True)
    fig.text(0.5, 0.97, title, ha="center", va="top",
             color=PL["white"], fontsize=13, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.91, subtitle, ha="center", va="top",
                 color=PL["grey"], fontsize=9)
    bar_ax = fig.add_axes([0, 0, 1, 0.018])
    bar_ax.set_facecolor(PL["pink"])
    bar_ax.set_axis_off()

def pl_legend(ax):
    """Légende au style PL."""
    return ax.legend(facecolor="#FFFFFF15", edgecolor="#FFFFFF30",
                     labelcolor=PL["white"], fontsize=9)

def save_fig(fig, name_normal, name_transparent):
    """Sauvegarde en version fond PL (rapport) et fond transparent (slides)."""
    fig.patch.set_facecolor(PL["bg"])
    plt.savefig(f"outputs/figures/{name_normal}",
                dpi=150, bbox_inches="tight", facecolor=PL["bg"])
    fig.patch.set_alpha(0)
    plt.savefig(f"outputs/figures/{name_transparent}",
                dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"✓ {name_normal} + {name_transparent} sauvegardées")

# Variable AwayWin utile pour certains graphiques
df["AwayWin"] = (df["FTR"] == "A").astype(int)

print("✓ Setup OK — dataset chargé :", df.shape)
print("✓ Colonnes :", list(df.columns))


# =========================================================
# 1) DISTRIBUTION DE L'ÉCHANTILLON
#    Répartition des 3 résultats possibles
#    → permet de voir si la variable cible est équilibrée
#    1    = fond PL (rapport)
#    1bis = fond transparent (slides)
# =========================================================

result_counts = df["FTR"].value_counts().reindex(["H", "D", "A"])
result_pct    = result_counts / len(df) * 100

labels     = ["Victoire\nà domicile", "Match\nnul", "Victoire\nà l'extérieur"]
bar_colors = [PL["green"], PL["grey"], PL["pink"]]

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.set_facecolor(PL["bg"])
fig.subplots_adjust(top=0.89, bottom=0.13, left=0.11, right=0.95)

bars = ax.bar(labels, result_pct.values,
              color=bar_colors, width=0.5,
              edgecolor=PL["bg"], linewidth=1.5)

for bar, val, count in zip(bars, result_pct.values, result_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%\n({count:,} matchs)",
            ha="center", va="bottom",
            color=PL["white"], fontweight="bold", fontsize=11)

ax.axhline(100/3, color=PL["white"], linestyle="--",
           linewidth=1.2, alpha=0.4)
ax.set_ylabel("% des matchs", color=PL["white"], fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, 62)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(colors=PL["white"], labelsize=10)

fig.text(0.5, 0.97, "Distribution de l'échantillon (1993–2023)",
         ha="center", va="top", color=PL["white"], fontsize=13, fontweight="bold")

ax.legend(
    handles=[plt.Line2D([0], [0], color=PL["white"], linestyle="--",
                        linewidth=1.5, label="Équilibre parfait (33.3%)")],
    facecolor=PL["bg"], edgecolor=PL["white"],
    labelcolor=PL["white"], fontsize=10, loc="upper right"
)

bar_ax = fig.add_axes([0, 0, 1, 0.018])
bar_ax.set_facecolor(PL["pink"])
bar_ax.set_axis_off()

save_fig(fig,
         "FIG_01_distribution_echantillon.png",
         "FIG_01bis_distribution_echantillon_transparent.png")


# =========================================================
# 2) ÉVOLUTION DU TAUX DE VICTOIRE À DOMICILE PAR SAISON
#    Vue temporelle : l'avantage domicile a-t-il toujours existé ?
#    Met en évidence l'effet Covid (2020-2021)
#    2    = fond PL (rapport)
#    2bis = fond transparent (slides)
# =========================================================

season_stats = df.groupby("Season_End_Year").agg(
    HomeWinRate=("HomeWin", "mean")
).reset_index()

global_mean = df["HomeWin"].mean() * 100

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_facecolor(PL["bg"])
fig.subplots_adjust(top=0.88, bottom=0.12, left=0.09, right=0.97)

# Zone Covid surlignée
ax.axvspan(2019.5, 2021.5, color=PL["pink"], alpha=0.12, zorder=0)

# Remplissage sous/au-dessus de la moyenne
ax.fill_between(season_stats["Season_End_Year"],
                season_stats["HomeWinRate"] * 100, global_mean,
                where=season_stats["HomeWinRate"] * 100 >= global_mean,
                alpha=0.15, color=PL["green"], interpolate=True)
ax.fill_between(season_stats["Season_End_Year"],
                season_stats["HomeWinRate"] * 100, global_mean,
                where=season_stats["HomeWinRate"] * 100 < global_mean,
                alpha=0.15, color=PL["pink"], interpolate=True)

# Ligne moyenne globale
ax.axhline(global_mean, color=PL["white"], linestyle="--",
           linewidth=1.2, alpha=0.4,
           label=f"Moyenne sur 30 ans ({global_mean:.1f}%)")

# Courbe principale
ax.plot(season_stats["Season_End_Year"],
        season_stats["HomeWinRate"] * 100,
        color=PL["green"], linewidth=2.5,
        marker="o", markersize=5,
        markerfacecolor=PL["green"],
        markeredgecolor=PL["bg"], markeredgewidth=1.5,
        label="Taux de victoire à domicile", zorder=5)

# Annotation Covid
ax.annotate("Saisons Covid\n(sans public)",
            xy=(2020.5, season_stats[season_stats["Season_End_Year"] == 2020]
                ["HomeWinRate"].values[0] * 100),
            xytext=(2018.2, 58),
            color=PL["pink"], fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PL["pink"],
                            lw=1.2, connectionstyle="arc3,rad=-0.2"))

ax.set_xlabel("Saison (année de fin)", color=PL["white"], fontsize=10)
ax.set_ylabel("% de victoires à domicile", color=PL["white"], fontsize=10)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(28, 68)
ax.tick_params(colors=PL["white"], labelsize=9)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(PL["white"])
ax.spines["left"].set_color(PL["white"])

fig.text(0.5, 0.97,
         "Évolution du taux de victoire à domicile par saison (1993–2023)",
         ha="center", va="top",
         color=PL["white"], fontsize=13, fontweight="bold")

ax.legend(facecolor=PL["bg"], edgecolor=PL["white"],
          labelcolor=PL["white"], fontsize=9, loc="upper right")

bar_ax = fig.add_axes([0, 0, 1, 0.018])
bar_ax.set_facecolor(PL["pink"])
bar_ax.set_axis_off()

save_fig(fig,
         "FIG_02_evolution_homewin.png",
         "FIG_02bis_evolution_homewin_transparent.png")

# =========================================================
# 3) EFFET COVID — AVANT / PENDANT / APRÈS
#    Compare le taux de victoire domicile et les buts moyens
#    selon les 3 périodes : normale, Covid, post-Covid
#    3    = fond PL (rapport)
#    3bis = fond transparent (slides)
# =========================================================
 
df["Periode"] = pd.cut(
    df["Season_End_Year"],
    bins=[1992, 2019, 2021, 2023],
    # 3 périodes : avant Covid, Covid (sans public), après Covid
    labels=["Avant Covid\n(1993–2019)", "Covid\n(2020–2021)", "Après Covid\n(2022–2023)"]
)
 
# Agrégation par période : taux de victoire + buts moyens
covid_stats = df.groupby("Periode", observed=True).agg(
    HomeWinRate=("HomeWin", "mean"),
    AvgHomeGoals=("HomeGoals", "mean"),
    AvgAwayGoals=("AwayGoals", "mean"),
).reset_index()
 
# 2 sous-graphes côte à côte : taux victoire (gauche) + buts (droite)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(top=0.84, bottom=0.13, left=0.07, right=0.97, wspace=0.3)
 
for ax in axes:
    ax.set_facecolor(PL["bg"])
 
# ── Sous-graphe gauche : taux de victoire domicile ────────────────────────
colors_periode = [PL["green"], PL["pink"], PL["purple"]]
bars = axes[0].bar(covid_stats["Periode"], covid_stats["HomeWinRate"] * 100,
                   color=colors_periode, width=0.5,
                   edgecolor=PL["bg"], linewidth=1.5)
 
# Labels de pourcentage au-dessus de chaque barre
for i, (_, row) in enumerate(covid_stats.iterrows()):
    axes[0].text(i, row["HomeWinRate"] * 100 + 0.5,
                 f"{row['HomeWinRate']*100:.1f}%",
                 ha="center", color=PL["white"], fontweight="bold", fontsize=12)
 
# Ligne de référence = moyenne globale sur toute la période
axes[0].axhline(df["HomeWin"].mean() * 100, color=PL["white"], linestyle="--",
                linewidth=1.2, alpha=0.4,
                label=f"Moyenne globale ({df['HomeWin'].mean()*100:.1f}%)")
axes[0].set_ylabel("% de victoires à domicile", color=PL["white"], fontsize=10)
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[0].set_ylim(0, 62)
axes[0].grid(False)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[0].spines["bottom"].set_color(PL["white"])
axes[0].spines["left"].set_color(PL["white"])
axes[0].tick_params(colors=PL["white"], labelsize=9)
axes[0].set_title("Taux de victoire à domicile", color=PL["white"],
                  fontweight="bold", fontsize=11, pad=10)
axes[0].legend(facecolor=PL["bg"], edgecolor=PL["white"],
               labelcolor=PL["white"], fontsize=9)
 
# ── Sous-graphe droit : buts moyens home vs away ─────────────────────────
x = np.arange(len(covid_stats))
w = 0.35  # largeur de chaque barre dans le groupe
 
# Barres décalées : home à gauche, away à droite
b1 = axes[1].bar(x - w/2, covid_stats["AvgHomeGoals"], w,
                 color=PL["green"], label="Buts domicile",
                 edgecolor=PL["bg"], linewidth=1.5)
b2 = axes[1].bar(x + w/2, covid_stats["AvgAwayGoals"], w,
                 color=PL["pink"], label="Buts extérieur",
                 edgecolor=PL["bg"], linewidth=1.5)
 
# Labels de valeur au-dessus de chaque barre
for bar in list(b1) + list(b2):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f"{bar.get_height():.2f}",
                 ha="center", color=PL["white"], fontweight="bold", fontsize=10)
 
axes[1].set_xticks(x)
axes[1].set_xticklabels(covid_stats["Periode"], fontsize=9)
axes[1].set_ylabel("Buts moyens par match", color=PL["white"], fontsize=10)
axes[1].set_ylim(0, 2.8)
axes[1].grid(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
axes[1].spines["bottom"].set_color(PL["white"])
axes[1].spines["left"].set_color(PL["white"])
axes[1].tick_params(colors=PL["white"], labelsize=9)
axes[1].set_title("Buts moyens domicile vs extérieur", color=PL["white"],
                  fontweight="bold", fontsize=11, pad=10)
axes[1].legend(facecolor=PL["bg"], edgecolor=PL["white"],
               labelcolor=PL["white"], fontsize=9)
 
# Titre global de la figure
fig.text(0.5, 0.97, "Impact des saisons Covid sur l'avantage domicile",
         ha="center", va="top", color=PL["white"], fontsize=13, fontweight="bold")
 
# Bande signature PL en bas
bar_ax = fig.add_axes([0, 0, 1, 0.018])
bar_ax.set_facecolor(PL["pink"])
bar_ax.set_axis_off()
 
save_fig(fig,
         "FIG_03_effet_covid.png",
         "FIG_03bis_effet_covid_transparent.png")
 
 
# =========================================================
# 4) TAUX DE VICTOIRE DOMICILE PAR QUINTILE DE STRENGTHDIFF
#    Question clé : l'avantage domicile persiste-t-il même
#    quand l'équipe visiteuse est objectivement plus forte ?
#    4    = fond PL (rapport)
#    4bis = fond transparent (slides)
# =========================================================
 
# Découpage de StrengthDiff en 5 quintiles égaux
# Q1 = domicile très inférieur sur le papier, Q5 = domicile très supérieur
df["StrengthDiff_bin"] = pd.qcut(df["StrengthDiff"], q=5, labels=[
    "Q1 — Très\ndéfavorable", "Q2 —\nDéfavorable",
    "Q3 —\nÉquilibré",        "Q4 —\nFavorable",
    "Q5 — Très\nfavorable"
])
 
# Taux de victoire domicile moyen + effectif par quintile
strength_stats = (df.groupby("StrengthDiff_bin", observed=True)["HomeWin"]
                    .agg(["mean", "count"])
                    .reset_index())
strength_stats.columns = ["Quintile", "HomeWinRate", "Count"]
 
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_facecolor(PL["bg"])
fig.subplots_adjust(top=0.84, bottom=0.16, left=0.10, right=0.95)
 
# Dégradé de couleur : rose = très défavorable → gris = équilibré → vert = très favorable
bar_colors = [PL["pink"], "#C0406A", PL["grey"], "#40C87A", PL["green"]]
 
bars = ax.bar(strength_stats["Quintile"], strength_stats["HomeWinRate"] * 100,
              color=bar_colors, width=0.6,
              edgecolor=PL["bg"], linewidth=1.5)
 
# Labels de pourcentage au-dessus de chaque barre
for bar, val in zip(bars, strength_stats["HomeWinRate"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val*100:.1f}%",
            ha="center", va="bottom",
            color=PL["white"], fontweight="bold", fontsize=11)
 
# Ligne de référence = moyenne globale (toutes conditions confondues)
ax.axhline(df["HomeWin"].mean() * 100, color=PL["white"], linestyle="--",
           linewidth=1.2, alpha=0.4,
           label=f"Moyenne globale ({df['HomeWin'].mean()*100:.1f}%)")
 
ax.set_ylabel("% de victoires à domicile", color=PL["white"], fontsize=10)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, 82)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(PL["white"])
ax.spines["left"].set_color(PL["white"])
ax.tick_params(colors=PL["white"], labelsize=9)
 
ax.legend(facecolor=PL["bg"], edgecolor=PL["white"],
          labelcolor=PL["white"], fontsize=9, loc="upper left")
 
fig.text(0.5, 0.97,
         "Taux de victoire à domicile selon le niveau relatif des équipes",
         ha="center", va="top",
         color=PL["white"], fontsize=13, fontweight="bold")
fig.text(0.5, 0.90,
         "Quintiles de StrengthDiff — même en Q1, l'équipe à domicile gagne dans près d'1 match sur 4",
         ha="center", va="top", color=PL["grey"], fontsize=9)
 
bar_ax = fig.add_axes([0, 0, 1, 0.018])
bar_ax.set_facecolor(PL["pink"])
bar_ax.set_axis_off()
 
save_fig(fig,
         "FIG_04_homewin_par_quintile.png",
         "FIG_04bis_homewin_par_quintile_transparent.png")



# =========================================================
# 5) HEATMAP DES CORRÉLATIONS
#    Vue d'ensemble des relations linéaires entre toutes
#    les variables numériques du dataset
#    5    = fond PL (rapport)
#    5bis = fond transparent (slides)
# =========================================================
 
# Variables numériques pertinentes pour l'analyse
numeric_cols = [
    "HomeGoals", "AwayGoals", "TotalGoals", "GoalDiff",
    "HomeStrength", "AwayStrength", "StrengthDiff",
    "HomeWin", "Covid"
]
 
# Matrice de corrélation de Pearson
corr = df[numeric_cols].corr()
 
# Masque triangle supérieur pour n'afficher que la moitié basse
# (la matrice est symétrique, inutile de répéter l'information)
mask = np.triu(np.ones_like(corr, dtype=bool))
 
# Palette personnalisée rose → violet (bg) → vert, centrée sur 0
pl_cmap = LinearSegmentedColormap.from_list(
    "pl_corr", [PL["pink"], PL["bg"], PL["green"]], N=256
)
 
fig, ax = plt.subplots(figsize=(9, 7.5))
fig.subplots_adjust(top=0.90, bottom=0.18, left=0.15, right=0.97)
ax.set_facecolor(PL["bg"])
 
sns.heatmap(
    corr,
    mask=mask,
    annot=True,          # affiche la valeur dans chaque cellule
    fmt=".2f",           # arrondi à 2 décimales
    linewidths=0.5,
    cmap=pl_cmap,
    vmin=-1, vmax=1,     # bornes fixes pour cohérence de l'échelle
    linecolor=PL["bg"],
    ax=ax,
    annot_kws={"size": 9, "color": PL["white"]},
    cbar_kws={"shrink": 0.75}
)
 
# Colorbar styling
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelcolor=PL["white"], labelsize=8)
cbar.ax.set_facecolor(PL["bg"])
 
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",
                   color=PL["white"], fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                   color=PL["white"], fontsize=9)
 
fig.text(0.5, 0.97, "Matrice de corrélation des variables numériques",
         ha="center", va="top", color=PL["white"], fontsize=13, fontweight="bold")
fig.text(0.5, 0.02, "Vert : corrélation positive · Rose : corrélation négative",
         ha="center", color=PL["grey"], fontsize=8)
 
# Bande signature PL en bas
bar_ax = fig.add_axes([0, 0, 1, 0.018])
bar_ax.set_facecolor(PL["pink"])
bar_ax.set_axis_off()
 
save_fig(fig,
         "FIG_05_heatmap_correlations.png",
         "FIG_05bis_heatmap_correlations_transparent.png")
 

 
# =========================================================
# 6) ÉVOLUTION DES BUTS DOMICILE VS EXTÉRIEUR PAR SAISON
#    Permet de voir si l'écart de buts entre domicile et
#    extérieur se réduit dans le temps, notamment pendant Covid
#    6    = fond PL (rapport)
#    6bis = fond transparent (slides)
# =========================================================
 
# Moyenne des buts domicile et extérieur par saison
season_goals = df.groupby("Season_End_Year").agg(
    AvgHomeGoals=("HomeGoals", "mean"),
    AvgAwayGoals=("AwayGoals", "mean"),
).reset_index()
 
fig, ax = plt.subplots(figsize=(13, 6))
ax.set_facecolor(PL["bg"])
# right=0.72 : on réserve la droite pour la légende en dehors du graphique
fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.72)
 
# Zone Covid surlignée en rose discret
ax.axvspan(2019.5, 2021.5, color=PL["pink"], alpha=0.10, zorder=0)
 
# Courbe buts domicile
ax.plot(season_goals["Season_End_Year"], season_goals["AvgHomeGoals"],
        color=PL["green"], linewidth=2.5,
        marker="o", markersize=5,
        markerfacecolor=PL["green"],
        markeredgecolor=PL["bg"], markeredgewidth=1.5,
        label=f"Buts domicile (moy. {df['HomeGoals'].mean():.2f})", zorder=5)
 
# Courbe buts extérieur
ax.plot(season_goals["Season_End_Year"], season_goals["AvgAwayGoals"],
        color=PL["pink"], linewidth=2.5,
        marker="o", markersize=5,
        markerfacecolor=PL["pink"],
        markeredgecolor=PL["bg"], markeredgewidth=1.5,
        label=f"Buts extérieur (moy. {df['AwayGoals'].mean():.2f})", zorder=5)
 
# Remplissage entre les deux courbes
# Vert quand les buts domicile > extérieur (avantage domicile)
# Rose quand les buts extérieur >= domicile (avantage extérieur)
ax.fill_between(season_goals["Season_End_Year"],
                season_goals["AvgHomeGoals"],
                season_goals["AvgAwayGoals"],
                where=season_goals["AvgHomeGoals"] >= season_goals["AvgAwayGoals"],
                alpha=0.12, color=PL["green"], interpolate=True,
                label="Écart en faveur du domicile")
ax.fill_between(season_goals["Season_End_Year"],
                season_goals["AvgHomeGoals"],
                season_goals["AvgAwayGoals"],
                where=season_goals["AvgHomeGoals"] < season_goals["AvgAwayGoals"],
                alpha=0.12, color=PL["pink"], interpolate=True,
                label="Écart en faveur de l'extérieur")
 
# Annotation Covid
ax.text(2020.5, season_goals["AvgHomeGoals"].max() * 0.97,
        "COVID", ha="center", fontsize=8,
        color=PL["pink"], fontweight="bold")
 
ax.set_xlabel("Saison (année de fin)", color=PL["white"], fontsize=10)
ax.set_ylabel("Buts moyens par match", color=PL["white"], fontsize=10)
ax.tick_params(colors=PL["white"], labelsize=9)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(PL["white"])
ax.spines["left"].set_color(PL["white"])
 
ax.legend(facecolor=PL["bg"], edgecolor=PL["white"],
          labelcolor=PL["white"], fontsize=9,
          loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
 
fig.text(0.5, 0.97,
         "Évolution des buts domicile vs extérieur par saison (1993–2023)",
         ha="center", va="top",
         color=PL["white"], fontsize=13, fontweight="bold")

# Bande signature PL en bas
bar_ax = fig.add_axes([0, 0, 1, 0.018])
bar_ax.set_facecolor(PL["pink"])
bar_ax.set_axis_off()
 
save_fig(fig,
         "FIG_06_evolution_buts.png",
         "FIG_06bis_evolution_buts_transparent.png")
 
print("\n✓ Tous les graphiques ont été générés dans outputs/figures/")
print("  FIG_01 + FIG_01bis — Distribution de l'échantillon")
print("  FIG_02 + FIG_02bis — Évolution taux victoire domicile")
print("  FIG_03 + FIG_03bis — Effet Covid")
print("  FIG_04 + FIG_04bis — HomeWin par quintile de StrengthDiff")
print("  FIG_05 + FIG_05bis — Heatmap des corrélations")
print("  FIG_06 + FIG_06bis — Évolution des buts domicile vs extérieur")


# =========================================================
# 7) BIG SIX — TAUX DE VICTOIRE DOMICILE VS EXTÉRIEUR
#    Compare pour chaque grand club le taux de victoire
#    à domicile et à l'extérieur sur toute la période
#    7    = fond PL (rapport)
#    7bis = fond transparent (slides)
# =========================================================
 
BIG_SIX = ["Arsenal", "Chelsea", "Liverpool",
           "Manchester City", "Manchester Utd", "Tottenham"]
 
# Taux de victoire quand le Big Six joue à domicile
big6_home = (
    df[df["HomeTeam"].isin(BIG_SIX)]
    .groupby("HomeTeam")
    .agg(HomeWinRate=("HomeWin", "mean"))
    .reset_index()
    .rename(columns={"HomeTeam": "Team"})
)
 
# Taux de victoire quand le Big Six joue à l'extérieur
big6_away = (
    df[df["AwayTeam"].isin(BIG_SIX)]
    .groupby("AwayTeam")
    .agg(AwayWinRate=("AwayWin", "mean"))
    .reset_index()
    .rename(columns={"AwayTeam": "Team"})
)
 
# Fusion et tri par taux de victoire domicile décroissant
big6 = big6_home.merge(big6_away, on="Team")
big6 = big6.sort_values("HomeWinRate", ascending=False).reset_index(drop=True)
 
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_facecolor(PL["bg"])
fig.subplots_adjust(top=0.84, bottom=0.14, left=0.12, right=0.95)
 
x = np.arange(len(BIG_SIX))
w = 0.35  # largeur de chaque barre dans le groupe
 
# Barres domicile (vert) et extérieur (rose) côte à côte
b1 = ax.bar(x - w/2, big6["HomeWinRate"] * 100, w,
            color=PL["green"], label="À domicile",
            edgecolor=PL["bg"], linewidth=1.5)
b2 = ax.bar(x + w/2, big6["AwayWinRate"] * 100, w,
            color=PL["pink"], label="À l'extérieur",
            edgecolor=PL["bg"], linewidth=1.5)
 
# Labels de pourcentage au-dessus de chaque barre
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.0f}%",
            ha="center", va="bottom",
            color=PL["white"], fontweight="bold", fontsize=10)
 
ax.set_xticks(x)
ax.set_xticklabels(big6["Team"], fontsize=10, color=PL["white"])
ax.set_ylabel("Taux de victoire (%)", color=PL["white"], fontsize=10)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, 90)
ax.tick_params(colors=PL["white"], labelsize=9)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(PL["white"])
ax.spines["left"].set_color(PL["white"])
 
ax.legend(facecolor=PL["bg"], edgecolor=PL["white"],
          labelcolor=PL["white"], fontsize=10, loc="upper right")
 
fig.text(0.5, 0.97,
         "Big Six — Taux de victoire à domicile vs à l'extérieur (1993–2023)",
         ha="center", va="top",
         color=PL["white"], fontsize=13, fontweight="bold")
fig.text(0.5, 0.90,
         "Pour chaque grand club, le taux de victoire est systématiquement plus élevé à domicile",
         ha="center", va="top", color=PL["grey"], fontsize=9)
 
# Bande signature PL en bas
bar_ax = fig.add_axes([0, 0, 1, 0.018])
bar_ax.set_facecolor(PL["pink"])
bar_ax.set_axis_off()
 
save_fig(fig,
         "FIG_07_big6_taux_victoire.png",
         "FIG_07bis_big6_taux_victoire_transparent.png")