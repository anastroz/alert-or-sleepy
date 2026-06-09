# -*- coding: utf-8 -*-
"""

A set of scripts used to generate the plots for the survey study. 

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss

"""Preparing plots for the descriptives from JASP.

"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# DATA
labels = ["SASa", "SASs"]

# Manually created positions on the X axis
x_positions = [0.45, 0.55]

means_1 = [0.712, 2.309] # sleepiness
means_2 = [2.721, 0.138] # alertness
means_3 = [1.715, 1.318] # mixed

se_1 = [0.030, 0.050] # sleepiness
se_2 = [0.040, 0.017] # alertness
se_3 = [0.035, 0.042] # mixed

ci_1 = [1.96 * s for s in se_1] # sleepiness
ci_2 = [1.96 * s for s in se_2] # alertness
ci_3 = [1.96 * s for s in se_3] # mixed

# SUBPLOTS
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        "Sleepiness-oriented <br> stories",
        "Alertness-oriented <br> stories",
        "Mixed stories"
    ],
    horizontal_spacing=0.05
)

# Function adding one panel
def add_panel(fig, col, means, ci):

    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=means,
            mode='markers',

            marker=dict(
                size=7,
                color='#427898'
            ),

            error_y=dict(
                type='data',
                array=ci,
                visible=True,
                color='#427898',
                thickness=1.5,
                width=7
            ),

            hoverinfo='skip',
            showlegend=False
        ),

        row=1,
        col=col
    )

#  Adding three panels
add_panel(fig, 1, means_1, ci_1)
add_panel(fig, 2, means_2, ci_2)
add_panel(fig, 3, means_3, ci_3)

#  Axes style
for i in range(1, 4):

    fig.update_xaxes(
        title_text="SAS dimensions" if i == 1 else "",

        # Ticks
        tickvals=x_positions,
        ticktext=labels,

        # Axes range
        range=[0.35, 0.65],

        linecolor='black',
        linewidth=1,
        ticks='outside',
        showgrid=False,
        zeroline=False,

        row=1,
        col=i
    )

    fig.update_yaxes(
        title_text="Mean SAS value (0-4)" if i == 1 else "",

        range=[0, 4],
        tickmode='linear',
        tick0=0,
        dtick=1,

        showline=True,
        linecolor='black',
        linewidth=1,
        ticks='outside',
        showgrid=False,
        zeroline=False,

        row=1,
        col=i
    )

#  Layout
fig.update_layout(
    width=600,
    height=300,
    template="simple_white",
)

#  Text styling
fig.update_annotations(font_size=11)

fig.layout.annotations[0].update(y=1.08)
fig.layout.annotations[1].update(y=1.08)
fig.layout.annotations[2].update(y=1.21)

fig.update_xaxes(title_font_size=11)
fig.update_yaxes(title_font_size=11)

fig.show()


#########################################################

"""Preparation of the 3D histograms for the first study"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import ast

df = pd.read_csv('COORDS_Dane_po_filtracji_Krzysztofa_2026-04-28.tsv', sep="\t")

from matplotlib.colors import LightSource


# -----------------------
# Parse (x, y)
# -----------------------
def parse_tuple(val):
    if pd.isna(val):
        raise ValueError('NaN')
    return tuple(ast.literal_eval(val))

# -----------------------
# Empirical joint distribution
# -----------------------
def joint_distribution(df, cols):
    pairs = []

    for c in cols:
        pairs.extend(df[c].dropna().apply(parse_tuple))

    counts = Counter(pairs)
    total = sum(counts.values())

    return {k: v / total for k, v in counts.items()}

# -----------------------
# Groups
# -----------------------
groups = {
    "Sleepiness-oriented stories": range(1, 6),
    "Alertness-oriented stories": range(6, 11),
    "Mixed stories": range(11, 16)
}

# -----------------------
# Plot
# -----------------------
fig = plt.figure(figsize=(18, 6))

# --- compute global Z max ---
all_dists = []

for r in groups.values():
    kss_cols = [f"KSS {j:02d}" for j in r]
    sas_cols = [f"SAS {j:02d}" for j in r]

    all_dists.append(joint_distribution(df, kss_cols))
    all_dists.append(joint_distribution(df, sas_cols))

global_max = max(
    max(d.values()) if d else 0 for d in all_dists
)

# --- plotting ---
for i, (title, r) in enumerate(groups.items(), 1):

    ax = fig.add_subplot(1, 3, i, projection='3d')

    kss_cols = [f"KSS {j:02d}" for j in r]
    sas_cols = [f"SAS {j:02d}" for j in r]

    kss = joint_distribution(df, kss_cols)
    sas = joint_distribution(df, sas_cols)

    dx = dy = 0.1

    # -----------------------
    # ONLY NON-ZERO POINTS
    # -----------------------

    # KSS
    for (x, y), z in kss.items():
        ax.bar3d(x - 0.05, y - 0.1, 0,
                 dx, dy, z,
                 color='#427898', lightsource=LightSource(azdeg=140,altdeg=30), alpha=1)

    # SAS
    for (x, y), z in sas.items():
        ax.bar3d(x, y+0.1, 0,
                 dx, dy, z,
                 color='#b5d2dd', lightsource=LightSource(azdeg=140,altdeg=30), alpha=1)

    # --- labels ---
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Sleepiness", fontsize=15, labelpad=5)
    ax.set_ylabel("Alertness", fontsize=15, labelpad=5)
    ax.text2D(
                 -0.12,          # horizontal
                  0.5,            # vertical
                  "P(x,y)",
                  transform=ax.transAxes,
                  rotation=90,
                  fontsize=15,
                  va='center'
        )

    # --- FIXED SCALES ---
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_zlim(0, global_max)

    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3, 4])

    handles = [
    plt.Rectangle((0,0),1,1,facecolor='#427898', alpha=1, edgecolor='black'),
    plt.Rectangle((0,0),1,1,facecolor='#b5d2dd', alpha=1, edgecolor='black')
     ]

    fig.legend(handles, ["KSS", "SAS"], loc="center right", ncol=1)
    ax.tick_params(labelsize=12)
    ax.view_init(25, 25)

    fig.subplots_adjust(
      left=0,
      right=0.95,
      top=1,
      bottom=0.08,
      wspace=0.2
    )
    fig.savefig('Contingency_plots_study_1.png')
    fig.show()

# -----------------------
# Parse (x, y)
# -----------------------
def parse_tuple(val):
    if pd.isna(val):
        return None
    return tuple(ast.literal_eval(val))

# -----------------------
# Grouping of the SAS
# -----------------------
def classify_sas(pairs):
    counts = {
        "zero": 0,        # (0,0)
        "x_axis": 0,      # (x,0)
        "y_axis": 0,      # (0,y)
        "off_axis": 0     # (x,y)
    }

    for (x, y) in pairs:
        if x == 0 and y == 0:
            counts["zero"] += 1
        elif x == 0:
            counts["y_axis"] += 1
        elif y == 0:
            counts["x_axis"] += 1
        else:
            counts["off_axis"] += 1

    total = sum(counts.values())

    return {k: v / total for k, v in counts.items()}

# -----------------------
# Grouping of the KSS
# -----------------------
def classify_kss(pairs):
    counts = {
        "zero": 0,
        "x_axis": 0,
        "y_axis": 0
    }

    for (x, y) in pairs:
        if x == 0 and y == 0:
            counts["zero"] += 1
        elif x == 0:
            counts["y_axis"] += 1
        elif y == 0:
            counts["x_axis"] += 1

    total = sum(counts.values())

    return {k: v / total for k, v in counts.items()}

# -----------------------
# Obtaining pairs from the columns
# -----------------------
def get_pairs(df, cols):
    pairs = []
    for c in cols:
        pairs.extend(df[c].dropna().apply(parse_tuple))
    return pairs

# -----------------------
# Groups
# -----------------------
groups = {
    "Sleepiness-related stories": range(1, 6),
    "Alertness-related stories": range(6, 11),
    "Mixed stories": range(11, 16)
}

# -----------------------
# Computing the data
# -----------------------
results = {}

for name, r in groups.items():

    kss_cols = [f"KSS {i:02d}" for i in r]
    sas_cols = [f"SAS {i:02d}" for i in r]

    kss_pairs = get_pairs(df, kss_cols)
    sas_pairs = get_pairs(df, sas_cols)

    results[name] = {
        "KSS": classify_kss(kss_pairs),
        "SAS": classify_sas(sas_pairs)
    }

# -----------------------
# DataFrame as a result
# -----------------------
rows = []

for group, data in results.items():
    for scale, vals in data.items():
        row = {"group": group, "scale": scale}
        row.update(vals)
        rows.append(row)

result_df = pd.DataFrame(rows)

print(result_df)

"""
Histograms for the KSS

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("Dane_po_filtracji_Krzysztofa_2026-04-28 - Dane_po_filtracji_Krzysztofa_2026-04-28_cleaned.csv", sep=",")

df.columns

kss_alert = df['KSS MEAN alert stories']
kss_sleepy = df['KSS MEAN sleepy stories']
kss_mixed = df['KSS MEAN mixed stories']

data = [kss_alert, kss_sleepy, kss_mixed]
titles = ['Alertness-oriented stories', 'Sleepiness-oriented stories', 'Mixed stories']

# Histogram settings
bins = np.arange(1, 9.25, 0.25)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for i, ax in enumerate(axes):
    ax.hist(data[i], bins=bins, color='#b5d2dd', edgecolor='black')
    ax.set_title(titles[i])
    ax.set_xlabel('Mean KSS')

    if i == 0:
        ax.set_ylabel('Count')  

    ax.set_xticks(range(1, 10))
    ax.set_xlim(1, 9)
    ax.axvline(np.mean(data[i]), color='black', linestyle='dashed', linewidth=1)

plt.tight_layout()
plt.savefig("Mean_KSS_per_story_type.png")
plt.show()



'''
Equivalence plots for the survey study.

'''


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -----------------------------
# Data
# -----------------------------

titles_left = [
    "Sleepiness-oriented stories",
    "Alertness-oriented stories",
    "Mixed stories"
]

# Manually added values for each point
# left column = alertness
# right column = sleepiness

y_left = [0.712, 2.721, 1.715]
y_right = [2.309, 0.138, 1.318]

# -----------------------------
# Parameters
# -----------------------------

rect_width = 0.6   # rectangle width
rect_height = 2    # rectangle height

x_center = 0.25     # center of the plot
point_x = 0.25      # position of the point

# -----------------------------
# Figure
# -----------------------------

fig, axes = plt.subplots(3, 2, figsize=(8, 8))

for row in range(3):

    # ===== Left column =====
    ax = axes[row, 0]

    
    rect = Rectangle(
        (x_center - rect_width/2, -1),  
        rect_width,
        rect_height,
        color='#b5d2dd', alpha=0.7
    )
    ax.add_patch(rect)

    
    ax.plot(point_x, y_left[row], 'o', markersize=5, color="#427898")

    
    ax.set_xlim(0, 1)
    ax.set_ylim(-3, 3)

    ax.set_xticks([])
    ax.set_yticks(range(-3, 4))

    ax.set_ylabel("SAS alertness (SASa)")
    ax.set_title(titles_left[row], loc='left')

    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # ===== Right column =====
    ax = axes[row, 1]

    rect = Rectangle(
        (x_center - rect_width/2, -1),
        rect_width,
        rect_height,
        color='#b5d2dd', alpha=0.7
    )
    ax.add_patch(rect)

    ax.plot(point_x, y_right[row], 'o', markersize=5, color="#427898")

    ax.set_xlim(0, 1)
    ax.set_ylim(-3, 3)

    ax.set_xticks([])
    ax.set_yticks(range(-3, 4))

    ax.set_ylabel("SAS sleepiness (SASs)")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


plt.tight_layout()
plt.subplots_adjust(wspace=-0.20)
plt.savefig("Equivalence_testing_plots_study_1.png")
plt.show()


'''
Pearson's r plots for the survey study.
'''


import numpy as np
import matplotlib.pyplot as plt

# =========================
# Data
# =========================

labels = ["Sleepiness", "Alertness", "Mixed"]

means = [0.615, 0.874, 0.208]

lower = [0.578, 0.859, 0.163]
upper = [0.647, 0.887, 0.252]

lower_err = np.array(means) - np.array(lower)
upper_err = np.array(upper) - np.array(means)

yerr = np.array([lower_err, upper_err])

colors = [
    "#b5d2dd",
    "#b5d2dd",
    "#b5d2dd"
]

# =========================
# Figure
# =========================

fig, ax = plt.subplots(figsize=(5.5, 4))

x = np.arange(len(labels))

ax.bar(
    x,
    means,
    width=0.5,
    color=colors,
    edgecolor="black",
    linewidth=1.2,
    yerr=yerr,
    capsize=5
)

# =========================
# Descriptions
# =========================

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_ylabel("Pearson's r")
ax.set_xlabel("Condition")

# =========================
# Layout
# =========================

ax.set_ylim(0, 1)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

ax.tick_params(
    axis="both",
    which="major",
    width=1.2,
    length=4
)

plt.tight_layout()

plt.savefig("figure.pdf", bbox_inches="tight")
plt.savefig("figure.png", dpi=600, bbox_inches="tight")

plt.show()
