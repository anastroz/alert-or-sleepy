# -*- coding: utf-8 -*-
"""
This script was used to generate the 3D histograms for the experimental study.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("2026-04-15_Sample_of_SUBJ_COORDS_before_blocks_and_after_last - 2026-04-15_Sample_of_SUBJ_before_blocks_and_after_last.tsv", sep="\t")

# ============================================================
# PARSE (x, y)
# ============================================================

def parse_tuple(val):

    if pd.isna(val):
        return None

    val = str(val).strip()

    if val in ["NA", "(NA, NA)"]:
        return None

    try:
        return tuple(ast.literal_eval(val))
    except:
        return None

# ============================================================
# JOINT DISTRIBUTION
# ============================================================

def joint_distribution(series):

    pairs = []

    for val in series.dropna():

        parsed = parse_tuple(val)

        if parsed is not None:
            pairs.append(parsed)

    counts = Counter(pairs)

    total = sum(counts.values())

    if total == 0:
        return {}

    return {k: v / total for k, v in counts.items()}

# ============================================================
# SETTINGS
# ============================================================

sessions = ["Day", "Night"]

measurement_points = [1, 2, 3, 4, 5]

# ============================================================
# GLOBAL MAX FOR Z AXIS
# ============================================================

all_distributions = []

for session in sessions:

    df_sess = df[df["Session"] == session]

    for i in measurement_points:

        kss_col = f"KSS_coords_{i}"
        sas_col = f"SAS_coords_{i}"

        kss_dist = joint_distribution(df_sess[kss_col])
        sas_dist = joint_distribution(df_sess[sas_col])

        all_distributions.append(kss_dist)
        all_distributions.append(sas_dist)

global_max = max(
    max(d.values()) if d else 0
    for d in all_distributions
)

# ============================================================
# PLOT
# ============================================================

fig = plt.figure(figsize=(14, 28))

plot_index = 1

for row_idx, measurement in enumerate(measurement_points):

    for col_idx, session in enumerate(sessions):

        ax = fig.add_subplot(
            len(measurement_points),
            len(sessions),
            plot_index,
            projection='3d'
        )

        plot_index += 1

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        df_sess = df[df["Session"] == session]

        kss_col = f"KSS_coords_{measurement}"
        sas_col = f"SAS_coords_{measurement}"

        kss = joint_distribution(df_sess[kss_col])
        sas = joint_distribution(df_sess[sas_col])

        # ----------------------------------------------------
        # BAR SETTINGS
        # ----------------------------------------------------

        dx = dy = 0.1

        # ----------------------------------------------------
        # KSS
        # ----------------------------------------------------

        for (x, y), z in kss.items():

            ax.bar3d(
                x - 0.05,
                y - 0.1,
                0,
                dx,
                dy,
                z,
                color='#427898',
                lightsource=LightSource(azdeg=140,altdeg=30),
                alpha=1,
                shade=True
            )

        # ----------------------------------------------------
        # SAS
        # ----------------------------------------------------

        for (x, y), z in sas.items():

            ax.bar3d(
                x,
                y + 0.1,
                0,
                dx,
                dy,
                z,
                color='#b5d2dd',
                lightsource=LightSource(azdeg=140,altdeg=30),
                alpha=1,
                shade=True
            )

        # ----------------------------------------------------
        # TITLES
        # ----------------------------------------------------
        if measurement == 1:
          subplot_title = "0'"
        elif measurement == 2:
          subplot_title = "30'"
        elif measurement == 3:
          subplot_title = "60'"
        elif measurement == 4:
          subplot_title = "90'"
        elif measurement == 5:
          subplot_title = "115'"

        ax.set_title(
            f"{session}: {subplot_title}",
            fontsize=16,
            pad=0
        )

        # ----------------------------------------------------
        # AXES
        # ----------------------------------------------------
        ax.set_xlabel("Sleepiness", fontsize=15, labelpad=5)
        ax.set_ylabel("Alertness", fontsize=15, labelpad=5)
        ax.text2D(
                 -0.12,          # pozycja pozioma
                  0.5,            # pozycja pionowa
                  "P(x,y)",
                  transform=ax.transAxes,
                  rotation=90,
                  fontsize=15,
                  va='center'
        )
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_zlim(0, global_max)

        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_yticks([0, 1, 2, 3, 4])

        ax.tick_params(labelsize=9)

        # ----------------------------------------------------
        # VIEW ANGLE
        # ----------------------------------------------------

        ax.view_init(elev=25, azim=25)

# ============================================================
# LEGEND
# ============================================================

handles = [
    plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor='#427898',
        edgecolor='black'
    ),
    plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor='#b5d2dd',
        edgecolor='black'
    )
]

fig.legend(
    handles,
    ["KSS", "SAS"],
    loc="upper right",
    fontsize=12
)

# ============================================================
# LAYOUT
# ============================================================

plt.subplots_adjust(
    left=0.3,
    right=1,
    top=1,
    bottom=0.03,
    hspace=0.02,
    wspace=0.15
)
plt.savefig("Contingency_plots_Study_2.png")
plt.show()

