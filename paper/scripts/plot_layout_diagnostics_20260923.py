"""Exploratory layout diagnostics from audited, saved cohort measurements.

All jointly observed objects are shown; no model is refitted and no frozen
selection is changed. Correlations are descriptive summaries, without a test.
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


from release_paths import ROOT, DATA, OUT, QA, RESEARCH
DATA = DATA / "layout_diagnostics_20260923"

STEM = "figS_layout_diagnostics"
INK = "#182838"
TEAL = "#007D83"
ORANGE = "#C96813"
PLUM = "#814B93"
COLORS = {"B1004": ORANGE, "B1007": PLUM}
ROLE_PAIRS = ["cos_X_Z1", "cos_X_Z2", "cos_Z1_Z2", "cos_X_V", "cos_Z1_V", "cos_Z2_V"]
ROLE_LABELS = ["X–Z1", "X–Z2", "Z1–Z2", "X–V", "Z1–V", "Z2–V"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.1,
    "axes.labelsize": 7.3,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.65,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
})


def truth(series):
    """Read booleans reproducibly, including CSV string representations."""
    return series.astype(str).str.lower().isin(["true", "1", "1.0"])


def scatter_panel(fig, rect, frame, label, title, layout_colors):
    ax = fig.add_axes(rect)
    ax.text(-0.16, 1.09, label, transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax.set_title(title, loc="left", pad=16, fontweight="bold")
    for group, color in layout_colors.items():
        part = frame if group == "Other layouts" else frame[frame.layout.eq(group)]
        ax.scatter(part.gamma, part.delta, s=8, color=color, alpha=0.53,
                   linewidths=0, rasterized=False, zorder=2)
    chosen = frame[truth(frame.CORE_selected)]
    ax.scatter(chosen.gamma, chosen.delta, s=20, facecolors="none", edgecolors=INK,
               linewidths=0.65, zorder=4)
    ax.axhline(0, color="#768591", linewidth=0.75, zorder=1)
    ax.axvline(0, color="#768591", linewidth=0.75, zorder=1)
    ax.set(xlim=(-0.275, 0.375), ylim=(-0.29, 0.655),
           xticks=[-0.2, 0, 0.2], yticks=[-0.2, 0, 0.2, 0.4, 0.6],
           xlabel="Within-site gain, Γ", ylabel="Two-site mean increment")
    rho = float(spearmanr(frame.gamma, frame.delta).statistic)
    # Statistics stay above the data limits rather than hiding points in a box.
    ax.text(0, 1.015, f"n = {len(frame):,}   Spearman ρ = {rho:+.2f}".replace("-", "−"),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
    return ax, rho, len(chosen)


def main():
    layout = pd.read_csv(DATA / "layout_summary.csv").sort_values("layout")
    objects = pd.read_csv(DATA / "object_diagnostics.csv")
    assert len(layout) == 7 and layout.layout.nunique() == 7
    assert len(objects) == 1539 and objects.object_id.nunique() == 1539
    required = {"layout", "gamma", "delta", "CORE_selected"}
    assert required.issubset(objects.columns), required - set(objects.columns)
    # Missing endpoints are the sole exclusion rule for this paired scatter.
    observed = np.isfinite(objects.gamma) & np.isfinite(objects.delta)
    paired = objects.loc[observed].copy()
    assert len(paired) == 1514
    highlighted = paired.layout.isin(COLORS)
    assert highlighted.sum() == 414 and (~highlighted).sum() == 1100
    values = layout[ROLE_PAIRS].to_numpy()
    assert np.isfinite(values).all()
    assert np.all((values >= -1) & (values <= 1))
    # 183 × 99 mm, expressed in inches for the plotting/export API.
    fig = plt.figure(figsize=(7.2047244, 3.8976378))

    # Panel a: all layout/role-pair means, with a consistent 0–0.8 color scale.
    ax = fig.add_axes([0.078, 0.295, 0.285, 0.548])
    ax.text(-0.18, 1.16, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax.set_title("Agreement between measurement roles", loc="left", pad=33,
                 fontweight="bold", fontsize=7.8)
    cmap = LinearSegmentedColormap.from_list("role_agreement", ["#FAF9F4", "#76B5B0", "#005D65"])
    im = ax.imshow(values, vmin=0, vmax=0.8, cmap=cmap, aspect="auto", interpolation="nearest")
    for (row, col), value in np.ndenumerate(values):
        ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6.8,
                color="white" if value >= 0.54 else INK)
    ax.set(xticks=np.arange(6), xticklabels=ROLE_LABELS,
           yticks=np.arange(7), yticklabels=layout.layout)
    ax.tick_params(axis="both", length=0, pad=4)
    for tick, name in zip(ax.get_yticklabels(), layout.layout):
        if name in COLORS:
            tick.set_color(COLORS[name])
            tick.set_fontweight("bold")
    for row, name in enumerate(layout.layout):
        if name in COLORS:
            ax.add_patch(Rectangle((-0.5, row - 0.5), 6, 1, fill=False,
                                   edgecolor=COLORS[name], linewidth=1.1))
    ax.axvline(2.5, color="white", lw=2)
    ax.spines[:].set_visible(False)
    ax.text(1, -0.77, "First three wells", ha="center", va="bottom", fontsize=6.9)
    ax.text(4, -0.77, "Verifier pairs", ha="center", va="bottom", fontsize=6.9)
    cbax = fig.add_axes([0.11, 0.178, 0.22, 0.025])
    cb = fig.colorbar(im, cax=cbax, orientation="horizontal", ticks=[0, 0.4, 0.8])
    cb.set_label("Mean pairwise cosine similarity", fontsize=7, labelpad=3)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_visible(False)

    bx, rho_special, nselected_special = scatter_panel(
        fig, [0.465, 0.295, 0.218, 0.548], paired.loc[highlighted], "b",
        "B1004 and B1007", COLORS)
    cx, rho_other, nselected_other = scatter_panel(
        fig, [0.768, 0.295, 0.218, 0.548], paired.loc[~highlighted], "c",
        "Other five layouts", {"Other layouts": TEAL})
    # Preserve the same y scale while avoiding a duplicate long label.
    cx.set_ylabel("")
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor=ORANGE,
                       markeredgecolor="none", markersize=4, label="B1004"),
               Line2D([], [], marker="o", color="none", markerfacecolor=PLUM,
                       markeredgecolor="none", markersize=4, label="B1007"),
               Line2D([], [], marker="o", color="none", markerfacecolor=TEAL,
                       markeredgecolor="none", markersize=4, label="Other layouts"),
               Line2D([], [], marker="o", color="none", markerfacecolor="none",
                       markeredgecolor=INK, markersize=4.5, label="CORE-selected")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.682, 0.09),
               ncol=2, columnspacing=1.5, handletextpad=0.5, labelspacing=0.6)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_bounds = []
    for text in fig.findobj(match=matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.width > 0 and box.height > 0:
            text_bounds.append((text.get_text(), box))
    canvas = fig.bbox
    outside = [t for t, b in text_bounds if b.x0 < -0.5 or b.y0 < -0.5
               or b.x1 > canvas.width + 0.5 or b.y1 > canvas.height + 0.5]
    assert not outside, f"Text outside canvas: {outside}"
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "figS_layout_diagnostics.pdf")
    fig.savefig(OUT / "figS_layout_diagnostics.svg")
    fig.savefig(OUT / "figS_layout_diagnostics.png", dpi=600)
    fig.savefig(OUT / "figS_layout_diagnostics.tiff", dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / f"{STEM}_preview.png", dpi=300)
    # Python-only grayscale preview for redundant-encoding inspection.
    from PIL import Image
    Image.open(OUT / f"{STEM}_preview.png").convert("L").save(OUT / f"{STEM}_grayscale.png")
    qa = {"dimensions_mm": [183, 99], "source_rows": len(objects),
          "jointly_observed_rows": len(paired), "missing_paired_rows": int((~observed).sum()),
          "highlighted_rows": int(highlighted.sum()), "other_rows": int((~highlighted).sum()),
          "rho_highlighted": rho_special, "rho_other": rho_other,
          "core_selected_jointly_observed": nselected_special + nselected_other,
          "core_selected_highlighted": nselected_special,
          "layout_counts": paired.groupby("layout").size().to_dict(),
          "heatmap_cells": int(values.size), "text_outside_canvas": outside,
          "inference": "Descriptive, no confidence intervals or hypothesis tests",
          "missing_rule": "Scatter requires finite gain and two-site mean increment",
          "sampling": "All jointly observed objects, no subsampling"}
    print(json.dumps(qa, indent=2))
    plt.close(fig)


if __name__ == "__main__":
    main()
