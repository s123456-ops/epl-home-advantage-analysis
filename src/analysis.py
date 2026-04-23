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
    # Bande signature PL en bas
    bar_ax = fig.add_axes([0, 0, 1, 0.018])
    bar_ax.set_facecolor(PL["pink"])
    bar_ax.set_axis_off()

def pl_legend(ax):
    """Légende au style PL."""
    return ax.legend(facecolor="#FFFFFF15", edgecolor="#FFFFFF30",
                     labelcolor=PL["white"], fontsize=9)

# Variable AwayWin utile pour certains graphiques
df["AwayWin"] = (df["FTR"] == "A").astype(int)

print("✓ Setup OK — dataset chargé :", df.shape)
print("✓ Colonnes :", list(df.columns))

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
    # Bande signature PL en bas
    bar_ax = fig.add_axes([0, 0, 1, 0.018])
    bar_ax.set_facecolor(PL["pink"])
    bar_ax.set_axis_off()

def pl_legend(ax):
    """Légende au style PL."""
    return ax.legend(facecolor="#FFFFFF15", edgecolor="#FFFFFF30",
                     labelcolor=PL["white"], fontsize=9)

# Variable AwayWin utile pour certains graphiques
df["AwayWin"] = (df["FTR"] == "A").astype(int)

print("✓ Setup OK — dataset chargé :", df.shape)
print("✓ Colonnes :", list(df.columns))


# =========================================================
# 1) DISTRIBUTION DE L'ÉCHANTILLON
#    Répartition des 3 résultats possibles
#    → permet de voir si la variable cible est équilibrée
# =========================================================

# Calcul des effectifs et pourcentages
result_counts = df["FTR"].value_counts().reindex(["H", "D", "A"])
result_pct    = result_counts / len(df) * 100

labels     = ["Victoire\nà domicile", "Match\nnul", "Victoire\nà l'extérieur"]
bar_colors = [PL["green"], PL["grey"], PL["pink"]]

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor(PL["bg"])
ax.set_facecolor(PL["bg"])
fig.subplots_adjust(top=0.89, bottom=0.13, left=0.11, right=0.95)

bars = ax.bar(labels, result_pct.values,
              color=bar_colors, width=0.5,
              edgecolor=PL["bg"], linewidth=1.5)

# Valeurs + effectifs au-dessus des barres
for bar, val, count in zip(bars, result_pct.values, result_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%\n({count:,} matchs)",
            ha="center", va="bottom",
            color=PL["white"], fontweight="bold", fontsize=11)

# Ligne d'équilibre parfait (33.3%)
ax.axhline(100/3, color=PL["white"], linestyle="--",
           linewidth=1.2, alpha=0.4,
           label="Équilibre parfait (33.3%)")

ax.set_ylabel("% des matchs", color=PL["white"], fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, 62)

# Pas de grille, pas de bordures
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(colors=PL["white"], labelsize=10)
pl_legend(ax)

# Titre principal
fig.text(0.5, 0.97, "Distribution de l'échantillon (1993–2023)",
         ha="center", va="top", color=PL["white"], fontsize=13, fontweight="bold")

# Légende manuelle bien visible
legend_patch = mpatches.Patch(facecolor="none", edgecolor="none", label="")
ax.legend(
    handles=[plt.Line2D([0], [0], color=PL["white"], linestyle="--",
                        linewidth=1.5, label="Équilibre parfait (33.3%)")],
    facecolor=PL["bg"],
    edgecolor=PL["white"],
    labelcolor=PL["white"],
    fontsize=10,
    loc="upper right"
)

# Bande signature PL en bas
bar_ax = fig.add_axes([0, 0, 1, 0.018])
bar_ax.set_facecolor(PL["pink"])
bar_ax.set_axis_off()

plt.savefig("outputs/figures/FIG_01_distribution_echantillon.png",
            dpi=150, bbox_inches="tight", facecolor=PL["bg"])
plt.close()
print("✓ FIG_01 sauvegardée — Distribution de l'échantillon")


# =========================================================
# 2) DISTRIBUTION DE L'ÉCHANTILLON — VERSION FOND TRANSPARENT
#    Même graphique, fond transparent pour intégration slides
# =========================================================

result_counts = df["FTR"].value_counts().reindex(["H", "D", "A"])
result_pct    = result_counts / len(df) * 100

labels     = ["Victoire\nà domicile", "Match\nnul", "Victoire\nà l'extérieur"]
bar_colors = [PL["green"], PL["grey"], PL["pink"]]

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_alpha(0)          # fond figure transparent
ax.set_facecolor("none")        # fond axes transparent
fig.subplots_adjust(top=0.89, bottom=0.13, left=0.11, right=0.95)

bars = ax.bar(labels, result_pct.values,
              color=bar_colors, width=0.5,
              edgecolor="none", linewidth=0)

# Valeurs + effectifs au-dessus des barres
for bar, val, count in zip(bars, result_pct.values, result_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%\n({count:,} matchs)",
            ha="center", va="bottom",
            color=PL["white"], fontweight="bold", fontsize=11)

# Ligne d'équilibre parfait (33.3%)
ax.axhline(100/3, color=PL["white"], linestyle="--",
           linewidth=1.2, alpha=0.6,
           label="Équilibre parfait (33.3%)")

ax.set_ylabel("% des matchs", color=PL["white"], fontsize=11)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(0, 62)
ax.tick_params(colors=PL["white"], labelsize=10)

# Seulement axe X et axe Y visibles, les autres masqués
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(PL["white"])
ax.spines["left"].set_color(PL["white"])

# Titre
fig.text(0.5, 0.97, "Distribution de l'échantillon (1993–2023)",
         ha="center", va="top", color=PL["white"], fontsize=13, fontweight="bold")

# Légende
ax.legend(
    handles=[plt.Line2D([0], [0], color=PL["white"], linestyle="--",
                        linewidth=1.5, label="Équilibre parfait (33.3%)")],
    facecolor=PL["bg"],
    edgecolor=PL["white"],
    labelcolor=PL["white"],
    fontsize=10,
    loc="upper right"
)

plt.savefig("outputs/figures/FIG_02_distribution_transparent.png",
            dpi=150, bbox_inches="tight", transparent=True)
plt.close()
print("✓ FIG_02 sauvegardée — Distribution fond transparent")