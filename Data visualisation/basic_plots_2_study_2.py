# -*- coding: utf-8 -*-
"""
This script was used to generate the plots of task performance and ECG variables in the experimental study. The variables to plot were changed manually. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from matplotlib.lines import Line2D

# =========================
# Reading the data
# =========================

file_path = "MCT_SAMPLE.xlsx"
df = pd.read_excel(file_path)

# =========================
# Mean + 95% CI
# =========================

def mean_ci(x):
    x = x.dropna()
    mean = np.mean(x)
    se = sem(x)
    ci = 1.96 * se
    return mean, ci

# =========================
# Plot style
# =========================

plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
})

# =========================
# Figure
# =========================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5),
    sharey=True
)

session_styles = {
    "DAY": {
        "markerfacecolor": "white",
    },
    "NIGHT": {
        "markerfacecolor": "black",
    }
}

session_offsets = {
    "DAY": -0.05,
    "NIGHT": 0.05
}

# =========================
# Plot
# =========================

for ax, block in zip(axes, [1, 2]):

    block_df = df[df["Block_order_per_type"] == block]

    for session in ["DAY", "NIGHT"]:

        session_df = block_df[block_df["Session"] == session]

        means = []
        cis = []
        trials = []

        offset = session_offsets[session]

        for trial in range(1, 6):

            values = session_df[
                session_df["Trial"] == trial
            ]["Total_omissions"]

            mean, ci = mean_ci(values)

            means.append(mean)
            cis.append(ci)
            trials.append(trial + offset)

        ax.errorbar(
            trials,
            means,
            yerr=cis,
            fmt='o-',
            color='black',
            linewidth=0.7,
            markersize=7,
            capsize=4,
            markerfacecolor=session_styles[session]["markerfacecolor"],
            markeredgecolor='black',
            label=None   
        )

    # =========================
    # Axis + style
    # =========================

    ax.set_title(f"Block {block}")
    ax.set_xlabel("Trial")
    ax.set_xticks([1, 2, 3, 4, 5])

    ax.tick_params(direction='out', length=8, width=1)

    for spine in ax.spines.values():
        spine.set_color('#B0B0B0')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Y label
axes[0].set_ylabel("Total number of omissions")

# =========================
# Custom legend
# =========================

legend_elements = [
    Line2D(
        [0], [0],
        marker='o',
        color='black',
        markerfacecolor='white',
        markeredgecolor='black',
        linestyle='None',
        markersize=8,
        label='Day'
    ),
    Line2D(
        [0], [0],
        marker='o',
        color='black',
        markerfacecolor='black',
        markeredgecolor='black',
        linestyle='None',
        markersize=8,
        label='Night'
    )
]

# =========================
# Positioning of the legend
# =========================

axes[1].legend(
    handles=legend_elements,
    frameon=False,
    loc='upper left',
    bbox_to_anchor=(1, 0.5)
)

axes[0].set_ylim(-1, 4)

fig.subplots_adjust(
    wspace=0.1,
    right=0.9
)

plt.savefig("Total omissions -- MCT.png")

plt.show()

