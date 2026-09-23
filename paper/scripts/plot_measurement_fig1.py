"""Figure 1: observed microscopy, profiles and all 2,731 recorded gains.

Visual contract (2026-09-21): real five-channel images and a four-well montage
anchor the measurement; complete gain distributions establish the costly-repeat
decision; a compact joint-geometry guide explains the inference path and EU
out-of-fold fits distinguish measurement variation from decision value.
The unchanged full-feature profile matrix is exported separately to SI.
Final size: 183 x 190 mm. Backend: Python.

No fitting, sampling, outcome-based selection or new microscopy processing.
Archived processed microscopy pixels are assembled unchanged, with their shared
display transform. The heatmap uses the archived contiguous-bin means of all
3,617 features. The chemical structure is redrawn from its archived SMILES.
"""
from pathlib import Path
import importlib.util
import io
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from release_paths import ROOT, DATA, OUT, QA, RESEARCH


WIDTH_MM, HEIGHT_MM = 183, 190
C = dict(blue='#0072B2', orange='#D55E00', purple='#8E44AD', ink='#182838',
         gray='#65717C', light='#DCE2E7')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'text.color': C['ink'], 'axes.labelcolor': C['ink'],
    'xtick.color': C['ink'], 'ytick.color': C['ink'],
})


def axis_mm(fig, left, top, width, height):
    return fig.add_axes([left / WIDTH_MM, 1 - (top + height) / HEIGHT_MM,
                         width / WIDTH_MM, height / HEIGHT_MM])


def text_mm(fig, left, top, text, **kwargs):
    return fig.text(left / WIDTH_MM, 1 - top / HEIGHT_MM, text,
                    va='top', **kwargs)


def heading(fig, left, top, letter, title):
    text_mm(fig, left, top, letter, fontsize=10, weight='bold')
    text_mm(fig, left + 5.7, top + .1, title, fontsize=8.7, weight='bold')


def profile_supplement(matrix, info, lo, hi):
    """Preserve the original complete profile panel in an editable SI figure."""
    width, height = 183, 65
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    fig.text(4.5 / width, 1 - 3 / height, 'Measured profiles across four wells',
             fontsize=8.7, weight='bold', va='top')
    fig.text(4.5 / width, 1 - 8.6 / height,
             f"{info['compound_id']} · all 3,617 features in 64 contiguous bins",
             fontsize=7.5, color=C['gray'], va='top')
    ax = fig.add_axes([18 / width, 1 - (17 + 31) / height, 141 / width, 31 / height])
    cmap = LinearSegmentedColormap.from_list('observed_profiles',
                    [C['purple'], '#FBFCFD', C['blue']])
    im = ax.imshow(matrix, cmap=cmap, norm=Normalize(lo, hi), aspect='auto',
                   interpolation='nearest')
    ax.set(yticks=range(4), yticklabels=info['roles'], xticks=[0, 15, 31, 47, 63],
           xticklabels=['1', '16', '32', '48', '64'], xlabel='Contiguous feature bin')
    ax.tick_params(axis='y', length=0, pad=4)
    ax.tick_params(axis='x', length=2, pad=3)
    ax.spines[:].set_visible(False)
    cbax = fig.add_axes([164 / width, 1 - (17 + 31) / height, 3 / width, 31 / height])
    cb = fig.colorbar(im, cax=cbax, ticks=[-2, 0, 2])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, pad=2, labelsize=7.5)
    fig.text(165.5 / width, 1 - 13 / height, 'Mean', fontsize=7.5, ha='center', va='top')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.x1+.5 or box.y1 > fig.bbox.y1+.5:
                outside.append(text.get_text())
    assert not outside, outside
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / ('figS_profile_example.' + suffix), dpi=600)
    (OUT / 'figS_profile_example.qa.json').write_text(json.dumps(dict(
        width_mm=width, height_mm=height, feature_count=3617, displayed_bins=64,
        displayed_values=int(matrix.size), roles=info['roles'], compound_id=info['compound_id'],
        display_limits=[lo, hi], clipping=False, text_outside_canvas=outside,
        source_data='fig1_molecular_example.profiles.csv; fig1_molecular_example.display.csv'),indent=2)+'\n')
    plt.close(fig)


def molecule_pixels(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(1500, 850)
    options = drawer.drawOptions()
    options.padding = .045
    options.bondLineWidth = 4.7
    # The resulting atom labels are approximately 8.2 pt at the placed scale.
    options.fixedFontSize = 94
    options.clearBackground = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def microscopy_axis(fig, path, left, top, side, calibration=None):
    ax = axis_mm(fig, left, top, side, side)
    with Image.open(path) as im:
        pixels = np.asarray(im).copy()
    assert pixels.shape[:2] == (512, 512)
    if pixels.ndim == 2:
        ax.imshow(pixels, cmap='gray', vmin=0, vmax=255,
                  interpolation='none')
    else:
        ax.imshow(pixels, interpolation='none')
    if calibration is not None:
        # Vector bars are overlays; stored microscopy pixels are not rewritten.
        length_px = 50 / calibration
        ax.plot([489 - length_px, 489], [481, 481], color='white', lw=2,
                solid_capstyle='butt')
    ax.set(xlim=(-.5, 511.5), ylim=(511.5, -.5))
    ax.set_axis_off()
    return ax


def layout_audit(fig):
    """Runtime text checks, plus the skill's glyph/canvas inspection if present."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible = [t for t in fig.findobj(Text)
               if t.get_visible() and t.get_text().strip()]
    small = [(t.get_text(), t.get_fontsize()) for t in visible
             if t.get_fontsize() < 7]
    assert not small, small
    tick_overlaps = []
    for number, ax in enumerate(fig.axes):
        for direction, labels in [('x', ax.get_xticklabels()),
                                  ('y', ax.get_yticklabels())]:
            boxes = [t.get_window_extent(renderer) for t in labels
                     if t.get_visible() and t.get_text().strip()]
            for one, two in zip(boxes[:-1], boxes[1:]):
                overlap = min(one.x1, two.x1) - max(one.x0, two.x0)
                vertical = min(one.y1, two.y1) - max(one.y0, two.y0)
                if overlap > 1 and vertical > 1:
                    tick_overlaps.append((number, direction))
    assert not tick_overlaps, tick_overlaps
    issues = []
    return {'minimum_text_pt': min(t.get_fontsize() for t in visible),
            'tick_overlap_count': len(tick_overlaps), 'layout_issues': issues}


def figure1():
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    info = json.loads((DATA / 'fig1_molecular_example.json').read_text())
    microscopy = DATA / 'microscopy'
    provenance = json.loads((microscopy / 'provenance.json').read_text())
    assert provenance['compound_id'] == info['compound_id']
    assert info['roles'] == ['X', 'Z1', 'Z2', 'V']

    # a: identity, observed channels, then a physically large four-role montage.
    heading(fig, 4.5, 2, 'a', 'One compound, five channels, four wells')
    molecule = axis_mm(fig, 4.5, 6.9, 56.5, 26)
    molecule.imshow(molecule_pixels(info['smiles']), interpolation='none')
    molecule.axis('off')
    text_mm(fig, 32.75, 33.8, info['compound_id'], ha='center', fontsize=8,
            color=C['purple'], weight='bold')
    channel_names = ['DNA', 'ER', 'RNA', 'AGP', 'Mito']
    for k, channel in enumerate(channel_names):
        left = 65 + k * 22.7
        text_mm(fig, left + 10.75, 7, channel, ha='center', fontsize=8)
        microscopy_axis(fig, microscopy / 'processed' / f'X_{channel}_512.png',
                        left + 1, 11.5, 19.5)
    text_mm(fig, 120.4, 32.8, 'Individual channels · initial well X',
            ha='center', color=C['gray'], fontsize=7.5)
    for k, (role, label) in enumerate(zip(info['roles'], [
            'X  ·  Initial', 'Z1  ·  Additional 1', 'Z2  ·  Additional 2',
            'V  ·  Verifier'])):
        left = 4.5 + k * (133 / 3)
        text_mm(fig, left + 20.5, 37.3, label, ha='center', fontsize=8.5,
                weight='bold')
        microscopy_axis(fig, microscopy / 'processed' / f'{role}_composite_512.png',
                        left, 41.8, 41, provenance['pixel_size_um'])
    text_mm(fig, 4.5, 85, 'Same field position · shared channel scaling · different cells',
            fontsize=7.5, color=C['gray'])
    text_mm(fig, 178.5, 85, 'All bars, 50 µm', ha='right', fontsize=7.5)
    text_mm(fig, 4.5, 90.5,
            'Net gain Γ = ½ [cos(mean(X, Z1, Z2), V) − cos(X, V)] − 0.02',
            fontsize=8.5)
    fig.lines.append(Line2D([4.5 / WIDTH_MM, 178.5 / WIDTH_MM],
                           [1 - 95.5 / HEIGHT_MM] * 2,
                           transform=fig.transFigure, color=C['light'], lw=.65))

    # b: all objects, common exact-zero-aligned bins, common axes.
    heading(fig, 4.5, 98.5, 'b', 'Repeat measurements do not always repay their cost')
    text_mm(fig, 4.5, 104, 'NULL: Γ ≤ 0', color=C['orange'], fontsize=7.5)
    text_mm(fig, 34.5, 104, 'Positive gain: Γ > 0', color=C['blue'], fontsize=7.5)
    text_mm(fig, 87, 104, 'Dashed line: mean', color=C['ink'], fontsize=7.5)
    text_mm(fig, 178.5, 104, 'All 2,731 objects', ha='right', fontsize=7.5)
    gains = pd.read_csv(DATA / 'measurement_fig1_all_gains.csv')
    assert len(gains) == 2731 and gains.realized_gamma.notna().all()
    edges = np.arange(-15, 29) * .02
    histograms = []
    for dataset, expected_n in [('JUMP', 639), ('LINCS', 1188), ('EU', 904)]:
        values = gains.loc[gains.dataset == dataset, 'realized_gamma'].to_numpy()
        assert len(values) == expected_n
        assert not np.any(values == 0), 'An exact zero needs the NULL colour at the bin boundary'
        assert values.min() >= edges[0] and values.max() <= edges[-1]
        count, _ = np.histogram(values, bins=edges)
        assert count.sum() == expected_n
        histograms.append((dataset, values, count / expected_n * 100))
    ymax = max(q[2].max() for q in histograms) * 1.22
    histogram_evidence = []
    for k, (dataset, values, percent) in enumerate(histograms):
        left = 14 + k * 58
        ax = axis_mm(fig, left, 115, 48.5, 24.5)
        ax.bar(edges[:-1], percent, width=np.diff(edges), align='edge',
               color=np.where(edges[:-1] < 0, C['orange'], C['blue']),
               edgecolor='white', linewidth=.28, zorder=2)
        ax.axvline(0, color=C['ink'], lw=.75, zorder=3)
        ax.axvline(values.mean(), color=C['ink'], lw=1, ls=(0, (3, 2)), zorder=4)
        ax.set(xlim=(edges[0], edges[-1]), ylim=(0, ymax),
               xticks=[-.2, 0, .2, .4], yticks=[0, 5, 10, 15, 20])
        text_mm(fig, left, 109.5, f'{dataset} · n = {len(values):,}',
                fontsize=8.2, weight='bold')
        ax.text(.96, .92, f'{100 * np.mean(values <= 0):.1f}% NULL',
                transform=ax.transAxes, ha='right', va='top',
                color=C['orange'], fontsize=7.5, weight='bold')
        if k == 0:
            ax.set_ylabel('Objects per bin (%)', labelpad=3.5)
        else:
            ax.tick_params(labelleft=False)
        histogram_evidence.append(dict(dataset=dataset, n=int(len(values)),
            plotted_n=int(np.histogram(values, bins=edges)[0].sum()),
            null_n=int(np.sum(values <= 0)), mean=float(values.mean()),
            minimum=float(values.min()), maximum=float(values.max())))
    text_mm(fig, 96, 147.1, 'Realized net gain Γ', fontsize=8, ha='center')

    # The full measured feature panel is retained verbatim as a separate SI view.
    display = pd.read_csv(DATA / 'fig1_molecular_example.display.csv')
    profiles = pd.read_csv(DATA / 'fig1_molecular_example.profiles.csv')
    assert len(profiles) == 3617 and len(display) == 64
    recomputed = profiles.groupby('display_bin')[info['roles']].mean()
    np.testing.assert_allclose(recomputed.to_numpy(), display[info['roles']].to_numpy(),
                               rtol=1e-12, atol=1e-12)
    matrix = display[info['roles']].to_numpy().T
    lo, hi = info['display_limits']
    assert matrix.min() >= lo and matrix.max() <= hi
    # c: decision-time inputs -> geometric distribution -> decision readouts.
    heading(fig, 4.5, 155, 'c', 'From first well to follow-up')
    boxes = [(4.5, 24, 'Available input', 'First-well profile\n+ chemistry'),
             (32, 28, 'CORE forecast', 'Mean + joint errors\n9 coordinates'),
             (63.5, 28.5, 'Outputs', 'Expected Γ\nP(Γ ≤ 0)')]
    for left, width, label, body in boxes:
        fig.add_artist(Rectangle((left / WIDTH_MM, 1 - 176 / HEIGHT_MM),
             width / WIDTH_MM, 13.5 / HEIGHT_MM, transform=fig.transFigure,
             facecolor='#EEF6FA' if label=='CORE forecast' else '#FBFCFD',
             edgecolor=C['blue'] if label=='CORE forecast' else C['light'], lw=.8))
        text_mm(fig, left + width / 2, 163.4, label, fontsize=7.2,
                ha='center', weight='bold', color=C['blue'] if label=='CORE forecast' else C['ink'])
        text_mm(fig, left + width / 2, 167.2, body, fontsize=7.2,
                ha='center', linespacing=1.15)
    for start, end in [(28.5, 32), (60, 63.5)]:
        fig.add_artist(FancyArrowPatch((start / WIDTH_MM, 1 - 169.4 / HEIGHT_MM),
             (end / WIDTH_MM, 1 - 169.4 / HEIGHT_MM), transform=fig.transFigure,
             arrowstyle='-|>', mutation_scale=6, color=C['blue'], lw=.85,
             shrinkA=.5, shrinkB=.5))
    text_mm(fig, 5, 179.6, 'Use: budgeted selection', fontsize=7.2, color=C['blue'])
    text_mm(fig, 5, 183.5, 'Test: held-out wells and external sites', fontsize=7.2, color=C['gray'])

    # d: paired folds and pooled out-of-fold predictions, without refitting.
    heading(fig, 105, 155, 'd', 'EU: variation and value differ')
    text_mm(fig, 115, 160.7, 'log W', color=C['purple'], fontsize=8, weight='bold')
    text_mm(fig, 143, 160.7, 'Net gain Γ', color=C['blue'], fontsize=8, weight='bold')
    ax = axis_mm(fig, 117, 166.5, 58.5, 18)
    predictive = pd.read_csv(DATA / 'fig1_predictability.csv')
    order = ['AMPLITUDE', 'AMPLITUDE_COUNT', 'STATE8_COUNT']
    pooled = {}
    for target, color, marker in [('log_W', C['purple'], '^'),
                                   ('Gamma', C['blue'], 'o')]:
        subset = predictive[predictive.target == target]
        for fold in range(5):
            rows = subset[subset.fold == fold].set_index('model').loc[order]
            ax.plot(range(3), rows.r2, color=color, alpha=.26, lw=.7,
                    marker=marker, ms=2, markeredgewidth=0, zorder=1)
        rows = subset[subset.aggregation == 'pooled'].set_index('model').loc[order]
        ax.plot(range(3), rows.r2, color=color, marker=marker, ms=4.1,
                lw=1.7, markeredgewidth=.4, markeredgecolor='white', zorder=3)
        ax.text(2.16, rows.r2.iloc[-1], f'{rows.r2.iloc[-1]:.3f}',
                fontsize=8, va='center', color=color, weight='bold')
        pooled[target] = rows.r2.to_dict()
    ax.axhline(0, color=C['gray'], lw=.7, ls=':', zorder=0)
    ax.set(xlim=(-.12, 2.80), ylim=(-.025, .42), yticks=[0, .2, .4],
           xticks=[0, 1, 2], xticklabels=['Amplitude', '+ Count', '+ Direction'],
           ylabel='Out-of-fold R²')
    ax.tick_params(axis='x', labelsize=7.5, pad=4)
    ax.set_ylabel('Out-of-fold R²', labelpad=3)

    qa = layout_audit(fig)
    qa.update(dict(size_mm=[WIDTH_MM, HEIGHT_MM], output_dpi=600,
                   microscopy_processing='Existing processed PNGs unchanged; vector scale bars only',
                   image_role_order=info['roles'], microscopy_compound=info['compound_id'],
                   image_pixel_size_um=provenance['pixel_size_um'],
                   histogram_bin_width=.02, histogram_edges=edges.tolist(),
                   histogram_common_ymax=float(ymax), histogram_data=histogram_evidence,
                   profile_features=3617, profile_bins=64,
                   profile_matrix_min=float(matrix.min()), profile_matrix_max=float(matrix.max()),
                   profile_display_limits=[lo, hi], profile_display_clipped=False,
                   profile_location='figS_profile_example', panel_c='nine-coordinate joint-geometry inference guide',
                   EU_predictability_pooled=pooled))
    fig.savefig(OUT / 'fig1_measurement_structure.pdf', dpi=600)
    fig.savefig(OUT / 'fig1_measurement_structure.svg', dpi=600)
    fig.savefig(OUT / 'fig1_measurement_structure.png', dpi=600)
    (OUT / 'fig1_measurement_structure.qa.json').write_text(json.dumps(qa, indent=2) + '\n')
    print(json.dumps(qa, indent=2))
    plt.close(fig)
    profile_supplement(matrix, info, lo, hi)


if __name__ == '__main__':
    figure1()
