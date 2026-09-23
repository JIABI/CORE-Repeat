"""Public JUMP raw-image exemplars for the fixed first DEV compound.

Selection is fixed by compound ID and site index, never by acquisition outcome.
All raw channels and public illumination fields are preserved. Display applies
the same pooled within-channel transform to all four roles; no local edits.
"""
from __future__ import annotations

import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image

from release_paths import ROOT, DATA, OUT, QA, RESEARCH
OUT = DATA / "microscopy"
BUCKET = "https://cellpainting-gallery.s3.amazonaws.com/"
META = "https://raw.githubusercontent.com/jump-cellpainting/datasets/main/metadata/"
ROLES = [
    ("X", "AEOJUM202", "JUMPCPE-20210812-Run20_20210815_062625"),
    ("Z1", "AEOJUM402", "JUMPCPE-20210820-Run22_20210821_180957"),
    ("Z2", "AEOJUM502", "JUMPCPE-20210820-Run23_20210823_145853"),
    ("V", "AEOJUM902", "JUMPCPE-20211014-Run36_20211014_223431"),
]
CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]
RGB = {"DNA": [0.0, 0.0, 1.0], "ER": [1.0, 1.0, 0.0],
       "RNA": [0.8, 0.0, 0.8], "AGP": [0.0, 1.0, 0.0],
       "Mito": [0.0, 1.0, 1.0]}
# A single theoretical maximum, not a fitted per-image display multiplier.
COMPOSITE_DIVISOR = float(np.max(np.sum(list(RGB.values()), axis=0)))


def get_bytes(url):
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return response.content


def fetch(task):
    url, path = task
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(get_bytes(url))
    return {"url": url, "local_file": str(path.relative_to(DATA)),
            "bytes": path.stat().st_size}


def http(s3):
    assert s3.startswith("s3://cellpainting-gallery/")
    return BUCKET + s3.removeprefix("s3://cellpainting-gallery/")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    example = json.loads((DATA / "fig1_molecular_example.json").read_text())
    assert example["compound_id"] == "JCP2022_000263"
    assert example["well_ids"] == [f"source_5::{p}::M04" for _, p, _ in ROLES]
    config_url = META + "microscope_config.csv"
    config_bytes = get_bytes(config_url)
    (OUT / "microscope_config.csv").write_bytes(config_bytes)
    config = next(r for r in csv.DictReader(io.StringIO(config_bytes.decode()))
                  if r["Metadata_Source"] == "source_5")
    pixel_um = float(config["Metadata_Pixel_Size_Microns"])
    assert pixel_um == 0.64974

    tasks, selections = [], []
    for role, plate, batch in ROLES:
        url = (BUCKET + f"cpg0016-jump/source_5/workspace/load_data_csv/{batch}/"
               f"{plate}/load_data_with_illum.csv")
        text = get_bytes(url).decode()
        candidates = [r for r in csv.DictReader(io.StringIO(text))
                      if r["Metadata_Well"] == "M04"]
        row = min(candidates, key=lambda r: int(r["Metadata_Site"]))
        assert int(row["Metadata_Site"]) == 1
        assert row["Metadata_Plate"] == plate
        assert row["Metadata_Source"] == "source_5"
        selections.append({"role": role, "manifest_url": url, **row})
        for channel in CHANNELS:
            tasks += [(http(row[f"URL_Orig{channel}"]), OUT / "raw" / f"{role}_{channel}.tif"),
                      (http(row[f"URL_Illum{channel}"]), OUT / "illum" / f"{role}_{channel}.npy")]
        print(f"Located {role}: {plate} M04 site 1, five channels", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        downloads = list(pool.map(fetch, tasks))
    (OUT / "selected_source_rows.json").write_text(json.dumps(selections, indent=2) + "\n")
    print(f"Downloaded {len(downloads)} files, {sum(x['bytes'] for x in downloads):,} bytes", flush=True)

    corrected, raw_qa = {}, []
    for role, _, _ in ROLES:
        corrected[role] = {}
        for channel in CHANNELS:
            with Image.open(OUT / "raw" / f"{role}_{channel}.tif") as image:
                raw = np.asarray(image).copy()
            illum = np.load(OUT / "illum" / f"{role}_{channel}.npy")
            assert raw.ndim == 2 and raw.shape == illum.shape
            assert np.isfinite(illum).all() and (illum > 0).all()
            corrected[role][channel] = raw.astype(np.float32) / illum.astype(np.float32)
            raw_qa.append({"role": role, "channel": channel, "shape": list(raw.shape),
                           "raw_dtype": str(raw.dtype), "raw_min": int(raw.min()),
                           "raw_max": int(raw.max()), "illum_min": float(illum.min()),
                           "illum_max": float(illum.max())})
    shapes = {tuple(item["shape"]) for item in raw_qa}
    assert len(shapes) == 1
    h, w = next(iter(shapes))
    crop = {"x": (w - 512) // 2, "y": (h - 512) // 2, "width": 512, "height": 512}
    limits = {ch: np.percentile(np.concatenate([corrected[role][ch].ravel()
                                              for role, _, _ in ROLES]), [0.5, 99.7]).tolist()
              for ch in CHANNELS}
    transformed, qa = {}, []
    processed = OUT / "processed"
    processed.mkdir(exist_ok=True)
    for role, _, _ in ROLES:
        transformed[role] = {}
        composite = np.zeros((512, 512, 3), dtype=np.float32)
        for ch in CHANNELS:
            arr = corrected[role][ch]
            lo, hi = limits[ch]
            assert hi > lo
            display = np.clip((arr - lo) / (hi - lo), 0, 1) ** 0.8
            cut = display[crop["y"]:crop["y"] + 512, crop["x"]:crop["x"] + 512]
            transformed[role][ch] = cut
            Image.fromarray(np.round(cut * 255).astype(np.uint8)).save(processed / f"{role}_{ch}_512.png")
            composite += cut[..., None] * np.asarray(RGB[ch])[None, None, :]
            qa.append({"role": role, "channel": ch,
                       "fullfield_fraction_below_lower": float(np.mean(arr < lo)),
                       "fullfield_fraction_above_upper": float(np.mean(arr > hi)),
                       "crop_fraction_zero": float(np.mean(cut == 0)),
                       "crop_fraction_one": float(np.mean(cut == 1))})
        composite /= COMPOSITE_DIVISOR
        qa.append({"role": role, "channel": "composite",
                   "crop_fraction_any_rgb_clipped": float(np.mean(np.any(composite > 1, axis=2)))})
        composite = np.clip(composite, 0, 1)
        transformed[role]["composite"] = composite
        Image.fromarray(np.round(composite * 255).astype(np.uint8)).save(processed / f"{role}_composite_512.png")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(4, 6, figsize=(9.4, 6.6))
    for i, (role, plate, _) in enumerate(ROLES):
        for j, channel in enumerate(["composite"] + CHANNELS):
            ax = axes[i, j]
            ax.imshow(transformed[role][channel], cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(channel.capitalize() if channel == "composite" else channel)
            if j == 0:
                ax.set_ylabel(f"{role}: {plate}\nM04, field 1", fontsize=8)
            ax.plot([512 - 28 - 50 / pixel_um, 512 - 28], [477, 477], color="white", lw=2)
            ax.text(484, 459, "50 µm", color="white", fontsize=6, ha="right")
    fig.suptitle("JCP2022_000263 | same compound, four observed wells | fixed centre crops", fontsize=10)
    fig.subplots_adjust(left=.08, right=.995, top=.90, bottom=.025, wspace=.05, hspace=.08)
    fig.savefig(OUT / "microscopy_contact_sheet.png", dpi=300)
    fig.savefig(OUT / "microscopy_contact_sheet.pdf")
    fig.savefig(OUT / "microscopy_contact_sheet.svg")
    plt.close(fig)

    provenance = {"compound_id": example["compound_id"], "dataset": "JUMP cpg0016 source_5",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Lexicographically first JUMP DEV ID from existing Fig1; minimum site index1 for every role; no endpoint, risk, ranking or image-quality selection",
        "well_identity_source": META + "well.csv.gz", "plate_batch_source": META + "plate.csv.gz",
        "well_identity_verified": [f"source_5,{plate},M04,JCP2022_000263" for _, plate, _ in ROLES],
        "microscope_source": config_url, "microscope": config,
        "pixel_size_um": pixel_um, "scale_bar_50um_pixels": 50 / pixel_um,
        "license": "CC0 1.0 Universal for Cell Painting Gallery image data; cite original JUMP resource",
        "license_source": "https://registry.opendata.aws/cellpainting-gallery/",
        "processing": {"illumination": "raw TIFF divided by corresponding public IllumChannel.npy",
            "limits_population": "pooled full-field illumination-corrected pixels across all four displayed roles, separately per channel",
            "percentile_limits": [0.5, 99.7], "per_channel_limits": limits,
            "gamma": 0.8, "channel_rgb": RGB,
            "composite": "sum pseudo-coloured five channels divided by a single theoretical maximum RGB weight sum; the same fixed scalar applies to every RGB component and all four roles",
            "composite_divisor": COMPOSITE_DIVISOR,
            "crop": crop, "raw_shape": [h, w], "local_adjustments": "none", "stitching": "none",
            "scale_bar": "Not burned into individual crops; calibrated bars only in contact sheet; final assembler uses recorded pixel size",
            "reuse": "new images from current2.0 JUMP DEV data; no reused1.0 image pixels",
            "display_scope": "One fixed field per well illustrates the measurement; endpoint remains pre-existing multi-field well profile. Different wells do not show the same physical cells."},
        "raw_qa": raw_qa, "display_clipping_qa": qa, "downloads": downloads}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved four composites and20 channel crops; pixel scale {pixel_um}µm, crop{crop}", flush=True)


if __name__ == "__main__":
    main()
