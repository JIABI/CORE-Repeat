#!/usr/bin/env python3
"""Current Figure 4: saved amplitude controls and frozen-scatter ablation.

Run from the repository: python paper/reproduce_figures.py 4
Inputs use OPAL2_DATA_ROOT; output paths use the portable release settings.
No model fitting or new predictive sampling is performed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image, ImageOps

from release_paths import ROOT, DATA, OUT, QA, RESEARCH
SRC = DATA

QA = QA / "fig7_amplitude_dependence"
OUT.mkdir(exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)

DATASETS = ["EU", "JUMP", "LINCS", "RxRx3"]
ARMS = ["AMP_ASC", "AMP_DESC", "AMP_HISTGB", "DISPERSION_DESC", "CORE", "HISTGB"]
ALL_ARMS = ARMS + ["RANDOM_EXPECTATION"]
COLOURS = dict(zip(ALL_ARMS, ["#7A7A7A", "#424242", "#8157A1", "#087F74", "#0072B2", "#D55E00", "#8A8A8A"]))
MARKERS = dict(zip(ALL_ARMS, ["v", "^", "s", "D", "o", "P", "d"]))
LABELS = {
    "AMP_ASC": "Low amplitude", "AMP_DESC": "High amplitude",
    "AMP_HISTGB": "Amplitude-only HistGB", "DISPERSION_DESC": "Predicted log dispersion",
    "CORE": "CORE", "HISTGB": "Direct HistGB", "RANDOM_EXPECTATION": "Random expectation",
}
OBSERVABLES = ["single_Z1", "single_Z2", "single_V", "difference_Z1_Z2", "difference_Z1_V",
               "difference_Z2_V", "average_Z1_Z2", "average_Z1_V", "average_Z2_V", "average_Z1_Z2_V"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7.5, "axes.labelsize": 7.5,
    "axes.titlesize": 8.3, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "legend.fontsize": 7.3, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "axes.linewidth": .65, "xtick.major.width": .6,
    "ytick.major.width": .6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

units = pd.read_csv(SRC / "m3_deployment_units.csv")
development = pd.read_csv(SRC / "m3_development_summary.csv")
confirmation = pd.read_csv(SRC / "m3_confirmation_summary.csv", keep_default_na=False)
observables = pd.read_csv(SRC / "m4dependence_observables.csv")
points = units[units.arm.isin(ARMS)].copy()
assert len(points) == 360
assert points.groupby("dataset").cell.nunique().to_dict() == {"EU": 5, "JUMP": 5, "LINCS": 10, "RxRx3": 40}
assert (points.selected > 0).all()
points["excess_per_selected"] = (points.total_Gamma - points.random_total_Gamma) / points.selected
points["plot_rule"] = points.arm.map({a: i + 1 for i, a in enumerate(ARMS)})
centres = []
for dataset in DATASETS:
    cells = sorted(points.loc[points.dataset.eq(dataset), "cell"].unique())
    # Deterministic cell jitter does not alter any ordinate and is shared by rules.
    jitters = np.linspace(-.20, .20, len(cells))
    jitter_map = dict(zip(cells, jitters))
    mask = points.dataset.eq(dataset)
    points.loc[mask, "plot_x"] = points.loc[mask, "plot_rule"] + points.loc[mask, "cell"].map(jitter_map)
    for arm in ARMS:
        sub = points[points.dataset.eq(dataset) & points.arm.eq(arm)]
        mean = (sub.total_Gamma.sum() - sub.random_total_Gamma.sum()) / sub.selected.sum()
        saved = development[development.dataset.eq(dataset) & development.arm.eq(arm)].iloc[0]
        assert np.isclose(mean, (saved.total_Gamma - saved.random_total_Gamma) / saved.selected)
        centres.append({"dataset": dataset, "arm": arm, "units": len(sub), "selected": sub.selected.sum(),
                        "excess_per_selected": mean})
centres = pd.DataFrame(centres)
bound_data = confirmation[confirmation.endpoint.isin(["Gamma", "NULL"])].copy()
assert len(bound_data) == 14
assert bound_data.selected.eq(192).all()
assert (bound_data.total_upper >= bound_data.total_lower).all()

assert observables.groupby(["dataset", "arm", "observable"]).crps.nunique().eq(1).all()
obs95 = observables[np.isclose(observables.nominal, .95)]
wide = obs95.pivot(index=["dataset", "observable"], columns="arm", values="crps")
wide["relative_crps_change_pct"] = 100 * (wide.DIAGONAL_SCATTER - wide.CORE_ORIGINAL) / wide.CORE_ORIGINAL
heat_data = wide.reset_index()
heat = wide.relative_crps_change_pct.unstack().reindex(index=DATASETS, columns=OBSERVABLES)
assert heat.shape == (4, 10) and heat.notna().all().all()
assert heat.loc["JUMP", "single_Z1"] < 0 and heat.loc["JUMP", "average_Z1_Z2"] < 0
points.to_csv(QA / "fig7_development_units.csv", index=False)
centres.to_csv(QA / "fig7_development_centres.csv", index=False)
bound_data.to_csv(QA / "fig7_confirmation_bounds.csv", index=False)
heat_data.to_csv(QA / "fig7_observable_crps_change.csv", index=False)

MM = 1 / 25.4
figure_width_mm = 183
fig = plt.figure(figsize=(figure_width_mm * MM, 180 * MM))
panel_axes = {}
titles = []

def panel_heading(letter, title, x, y):
    titles.append(fig.text(x, y, letter, fontsize=10, fontweight="bold", va="bottom"))
    titles.append(fig.text(x + .022, y + .0005, title, fontsize=8.2, fontweight="bold", va="bottom"))

for i, dataset in enumerate(DATASETS):
    left = .076 + i * .234
    ax = fig.add_axes([left, .756, .207, .183])
    panel_axes[chr(ord("a") + i)] = ax
    sub = points[points.dataset.eq(dataset)]
    ax.axhline(0, color="#858585", linewidth=.7, linestyle=(0, (3, 2)), zorder=1)
    ax.axhspan(-.075, 0, color="#F6F6F6", zorder=0)
    for j, arm in enumerate(ARMS):
        rows = sub[sub.arm.eq(arm)]
        ax.scatter(rows.plot_x, rows.excess_per_selected, marker=MARKERS[arm], s=13 if arm not in ["CORE", "HISTGB"] else 17,
                   color=COLOURS[arm], alpha=.74, linewidths=.35, edgecolors="white", zorder=3)
        y = centres.loc[centres.dataset.eq(dataset) & centres.arm.eq(arm), "excess_per_selected"].iloc[0]
        ax.plot([j + 1 - .26, j + 1 + .26], [y, y], color=COLOURS[arm], linewidth=2.2, solid_capstyle="butt", zorder=5)
    ax.set(xlim=(.5, 6.5), ylim=(-.075, .19), xticks=range(1, 7), yticks=[-.05, 0, .05, .10, .15])
    if i:
        ax.tick_params(axis="y", labelleft=False)
    else:
        ax.set_yticklabels(["−0.05", "0", "0.05", "0.10", "0.15"])
    ax.tick_params(axis="x", pad=3)
    panel_heading(chr(ord("a") + i), f"{dataset} · {sub.cell.nunique()} units", left - .008, .951)
fig.text(.015, .846, "Excess Γ per selected object", rotation=90, rotation_mode="anchor", fontsize=7.5, ha="center", va="center")

# Figure-wide numbered rule key; categorical comparisons are not connected.
key_text = []
for i, arm in enumerate(ARMS):
    col, row = i % 3, i // 3
    x, y = .085 + col * .318, .697 - row * .028
    fig.add_artist(Line2D([x], [y], marker=MARKERS[arm], markersize=4.5, color=COLOURS[arm],
                          linestyle="none", transform=fig.transFigure))
    key_text.append(fig.text(x + .014, y, f"{i + 1}  {LABELS[arm]}", fontsize=7.4, va="center",
                             color=COLOURS[arm], fontweight="bold" if arm in ["CORE", "HISTGB"] else "normal"))

# Fixed-cohort bounds: never use an unknown outcome's midpoint as an estimate.
for letter, endpoint, left, xlabel, limits, ticks in [
    ("e", "Gamma", .208, "Total realized Γ", (-5, 25), [-5, 0, 10, 20]),
    ("f", "NULL", .714, "Unsuccessful follow-ups (NULL)", (-2, 83), [0, 20, 40, 60, 80]),
]:
    ax = fig.add_axes([left, .405, .265, .200])
    panel_axes[letter] = ax
    sub = bound_data[bound_data.endpoint.eq(endpoint)].set_index("arm")
    for i, arm in enumerate(ALL_ARMS):
        lo, hi = sub.loc[arm, ["total_lower", "total_upper"]].astype(float)
        if np.isclose(lo, hi):
            ax.scatter([lo], [i], s=28, marker=MARKERS[arm], color=COLOURS[arm], zorder=3)
        else:
            ax.hlines(i, lo, hi, color=COLOURS[arm], linewidth=2.1, zorder=3)
            ax.vlines([lo, hi], i - .13, i + .13, color=COLOURS[arm], linewidth=1.2, zorder=3)
        if arm == "RANDOM_EXPECTATION":
            ax.axhspan(i - .35, i + .35, color="#F1F1F1", zorder=0)
    ax.set(xlim=limits, ylim=(6.6, -.6), xticks=ticks, yticks=range(7), yticklabels=[LABELS[a] for a in ALL_ARMS])
    ax.set_xlabel(xlabel, labelpad=5)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color="#B6B6B6", linewidth=.7, zorder=0)
    for tick, arm in zip(ax.get_yticklabels(), ALL_ARMS):
        tick.set_color(COLOURS[arm])
        tick.set_fontweight("bold" if arm in ["CORE", "HISTGB"] else "normal")
    panel_heading(letter, "Confirmation · 192 selections", left - .167, .620)
bound_note = fig.text(.5, .333, "Capped ranges: missing-outcome bounds, not confidence intervals", fontsize=7.3, ha="center")

# Forty unchanged contrasts, now encoded by signed position/length rather than colour.
panel_heading("g", "Error correlations improve combined-measurement forecasts", .05, .306)
observable_labels = ["Z1", "Z2", "V", "Z1 − Z2", "Z1 − V", "Z2 − V",
                     "mean(Z1, Z2)", "mean(Z1, V)", "mean(Z2, V)", "mean(Z1, Z2, V)"]
for j, dataset in enumerate(DATASETS):
    ax = fig.add_axes([.190 + j * .197, .081, .174, .191])
    panel_axes["g" + str(j)] = ax
    values = heat.loc[dataset].to_numpy()
    ax.axvline(0, color="#8A8A8A", lw=.7, zorder=1)
    for split in [2.5, 5.5, 8.5]:
        ax.axhline(split, color="#E5E5E5", lw=.5, zorder=0)
    for row, value in enumerate(values):
        colour = COLOURS["CORE"] if value >= 0 else COLOURS["HISTGB"]
        ax.barh(row, value, height=.43, color=colour, linewidth=0, zorder=2)
        ax.plot(value, row, marker="o", ms=2.2, color=colour, lw=0, zorder=3)
        ax.text(16.8, row, f"{value:.2f}".replace("-", "−"), ha="right", va="center", fontsize=7.3,
                color=colour if value < 0 else "#242424")
    ax.set(xlim=(-.7, 17.0), ylim=(9.6, -.6), xticks=[0, 6, 12], yticks=range(10),
           yticklabels=observable_labels if j == 0 else [])
    ax.tick_params(axis="y", length=0, pad=4, labelsize=7.2)
    ax.tick_params(axis="x", labelsize=7.2, pad=3)
    ax.spines["left"].set_visible(False)
    ax.set_title(dataset, fontsize=8.2, fontweight="bold", pad=5)
fig.text(.58, .023, "CRPS increase after removing scatter correlations (%)", fontsize=7.3, ha="center")

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
font_min = min(t.get_fontsize() for t in texts)
assert font_min >= 7.2, font_min
canvas = fig.bbox
clipped = []
for t in texts:
    box = t.get_window_extent(renderer)
    if box.x0 < canvas.x0 - 1 or box.y0 < canvas.y0 - 1 or box.x1 > canvas.x1 + 1 or box.y1 > canvas.y1 + 1:
        clipped.append(t.get_text())
assert not clipped, clipped
label_note_gap_pt = min(panel_axes[p].xaxis.label.get_window_extent(renderer).y0 for p in ["e", "f"]) - bound_note.get_window_extent(renderer).y1
note_heading_gap_pt = bound_note.get_window_extent(renderer).y0 - max(t.get_window_extent(renderer).y1 for t in titles[-2:])
label_note_gap_pt *= 72 / fig.dpi
note_heading_gap_pt *= 72 / fig.dpi
assert label_note_gap_pt > 0 and note_heading_gap_pt > 0

stem = OUT / "fig7_amplitude_dependence"
fig.savefig(OUT / "fig7_amplitude_dependence.pdf")
fig.savefig(OUT / "fig7_amplitude_dependence.svg")
fig.savefig(OUT / "fig7_amplitude_dependence.png", dpi=600)
preview = QA / "fig7_preview.png"
fig.savefig(preview, dpi=300)
img = Image.open(preview)
ImageOps.grayscale(img).save(QA / "fig7_grayscale.png")
for name, bounds in {
    "abcd": (.0, .645, 1, .99), "ef": (.0, .328, 1, .647), "g": (.0, .0, 1, .326),
}.items():
    x0, y0, x1, y1 = bounds
    img.crop((int(x0 * img.width), int((1-y1)*img.height), int(x1*img.width), int((1-y0)*img.height))).save(QA / f"fig7_panel_{name}.png")

qa = {
    "width_mm": 183, "height_mm": 180, "min_declared_font_pt": font_min,
    "development_points": len(points), "deployment_units": int(points.groupby("dataset").cell.nunique().sum()),
    "development_summary_marks": len(centres), "confirmation_bound_rows": len(bound_data),
    "confirmation_budget": 192, "observable_contrasts": int(heat.size),
    "negative_contrasts": heat_data.loc[heat_data.relative_crps_change_pct.lt(0)].to_dict("records"),
    "clipped_text": clipped, "no_fitting_or_sampling": True,
    "bound_axis_label_to_note_gap_pt": label_note_gap_pt,
    "bound_note_to_contrast_heading_gap_pt": note_heading_gap_pt,
    "interval_type": "fixed-cohort missing-outcome identification bounds, not confidence intervals",
    "source_inputs": ["source_data/" + p for p in ["m3_deployment_units.csv", "m3_development_summary.csv", "m3_confirmation_summary.csv", "m4dependence_observables.csv"]],
}
(QA / "layout_checks.json").write_text(json.dumps(qa, indent=2) + "\n")
print(json.dumps(qa, indent=2))
plt.close(fig)
