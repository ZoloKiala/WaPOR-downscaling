#!/usr/bin/env python3
# ============================================================
# Best-Date ET Map (Pred20 vs Native20 vs Coarse 300m)
#
# Supports two sites:
#   --site lamego  (default)
#   --site baixo
#
# Input:
#   - per_file_metrics_<site>_TESTONLY.csv from per_file_eval_catboost.py
#       required columns: file, tag, date, pvn_r2, n_pix_20
#   - prediction TIFFs created by per_file_eval_catboost.py with --save-preds:
#       <CSV_DIR>/preds_<site>/<tag>_pred.tif
#       (e.g., preds_lamego, preds_baixo)
#
# Output:
#   - <site>_best_ET_map.png in the same folder as the CSV
#
# Panels:
#   (a) Pred20  ETa20m_pred             (20 m)
#   (b) Native WaPOR ETa20m (b1)        (20 m)
#   (c) Native ETa20m aggregated to 300 m,
#       then upsampled back to 20 m (coarse 300 m blocks)
# ============================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.env import Env as RasterioEnv
import matplotlib.pyplot as plt

FALLBACK_NODATA = -9999.0
NODATA_OUT = -9999.0


# ------------------------------------------------------------
# Helpers for band mapping (adapted from your eval script)
# ------------------------------------------------------------
def band_map(ds):
    desc = list(ds.descriptions or [])
    bm = {desc[i]: i for i in range(ds.count) if i < len(desc) and desc[i]}
    if bm:
        return _normalize_bm(bm)

    bm2 = {}
    for i in range(ds.count):
        tag = ds.tags(i + 1) or {}
        for key in ("BANDNAME", "band_name", "NAME", "BandName",
                    "DESCRIPTION", "description", "name"):
            if key in tag and tag[key]:
                bm2[tag[key]] = i
                break
    if bm2:
        return _normalize_bm(bm2)
    return {}


def _normalize_bm(bm):
    out = {}
    for k, v in bm.items():
        kk = str(k).strip()
        out[kk] = v
        out.setdefault(kk.lower(), v)

    aliases = {
        # ETa300m kept for generality, but not required here
        "ETa300m": ["eta300m", "ETa_300m", "ETa300", "ET300m",
                    "AETI300", "AETI_300m", "L1-AETI-D", "L1_AETI_D"],
        "b1": ["B1", "b1", "label", "LABEL", "target", "y",
               "ETa20", "ETa20m", "AETI20"],
    }

    for canonical, keys in aliases.items():
        if canonical in out:
            continue
        for kk in keys:
            if kk in out:
                out[canonical] = out[kk]
                break
            if kk.lower() in out:
                out[canonical] = out[kk.lower()]
                break
    return out


def get_label_band_index(bm: dict, label_name: str = "b1"):
    keys = [label_name, label_name.lower(), label_name.upper(), "b1", "B1"]
    for k in keys:
        if k in bm:
            return bm[k]
    return None


def get_effective_nodata(ds):
    nod = ds.nodata
    if nod is None or (not np.isfinite(nod)):
        return FALLBACK_NODATA
    return float(nod)


# ------------------------------------------------------------
# Helpers for 300 m block aggregation
# ------------------------------------------------------------
def infer_block_factor(ds, target_m=300.0):
    """Infer integer block factor from pixel size and target resolution."""
    px = float(abs(ds.transform.a))
    py = float(abs(ds.transform.e))
    if px <= 0 or py <= 0:
        return None
    fx = int(round(target_m / px))
    fy = int(round(target_m / py))
    if fx < 1 or fy < 1:
        return None
    return fx, fy


def block_mean_2d(arr, fx, fy):
    """Mean over non-overlapping blocks of size (fy, fx)."""
    H, W = arr.shape
    H2 = (H // fy) * fy
    W2 = (W // fx) * fx
    if H2 <= 0 or W2 <= 0:
        return None

    a = arr[:H2, :W2].astype(np.float32, copy=False)
    a = a.reshape(H2 // fy, fy, W2 // fx, fx)

    valid = np.isfinite(a)
    cnt = valid.sum(axis=(1, 3)).astype(np.float32)
    s = np.where(valid, a, 0.0).sum(axis=(1, 3)).astype(np.float32)

    out = np.full(cnt.shape, np.nan, np.float32)
    m = cnt > 0
    out[m] = s[m] / cnt[m]
    return out


# ------------------------------------------------------------
# Core function
# ------------------------------------------------------------
def plot_best_site_map(metrics_csv: str, site: str = "lamego",
                       label_name: str = "b1"):
    site = site.lower()
    if site not in ("lamego", "baixo"):
        raise ValueError("site must be 'lamego' or 'baixo'")

    site_upper = site.upper()
    site_title = "Lamego" if site == "lamego" else "Baixo"
    preds_dir_name = f"preds_{site}"

    metrics_csv = Path(metrics_csv)
    out_dir = metrics_csv.parent

    df = pd.read_csv(metrics_csv)

    # Basic checks
    for col in ["file", "tag", "pvn_r2", "n_pix_20"]:
        if col not in df.columns:
            raise RuntimeError(f"CSV missing required column: {col}")

    # Valid rows: have R², some valid 20m pixels
    df_valid = df[
        df["pvn_r2"].notna()
        & df["n_pix_20"].fillna(0).astype(int).gt(0)
    ]
    if df_valid.empty:
        raise RuntimeError("No valid rows with pvn_r2 and n_pix_20>0 found in CSV.")

    # Best row by max pvn_r2
    best_row = df_valid.loc[df_valid["pvn_r2"].idxmax()]
    stack_fp = Path(best_row["file"])
    tag = str(best_row["tag"])
    pred_fp = out_dir / preds_dir_name / f"{tag}_pred.tif"
    best_date = best_row.get("date", "")
    best_r2 = float(best_row["pvn_r2"])

    if not stack_fp.exists():
        raise FileNotFoundError(f"Stack GeoTIFF not found: {stack_fp}")

    if not pred_fp.exists():
        raise FileNotFoundError(
            f"Pred GeoTIFF not found: {pred_fp}\n"
            f"Make sure you ran per_file_eval_catboost.py with --save-preds "
            f"and that the directory name is '{preds_dir_name}'."
        )

    print(f"[{site_upper}] Best date by pvn_r2: {best_date} (R² = {best_r2:.3f})")
    print(f"[{site_upper}] Stack: {stack_fp}")
    print(f"[{site_upper}] Pred : {pred_fp}")

    gdal_opts = {"GDAL_CACHEMAX": 2048, "GDAL_NUM_THREADS": "ALL_CPUS"}

    with RasterioEnv(**gdal_opts):
        with rasterio.open(stack_fp) as ds_stack, rasterio.open(pred_fp) as ds_pred:
            bm = band_map(ds_stack)

            lb_idx = get_label_band_index(bm, label_name=label_name)
            if lb_idx is None:
                raise RuntimeError(
                    f"Could not find label band for '{label_name}' in stack."
                )

            nod_stack = get_effective_nodata(ds_stack)
            nod_pred = ds_pred.nodata
            if nod_pred is None or not np.isfinite(nod_pred):
                nod_pred = NODATA_OUT

            # Read arrays
            native20 = ds_stack.read(lb_idx + 1).astype(np.float32)
            pred20 = ds_pred.read(1).astype(np.float32)

            # Mask nodata
            native20[(native20 == nod_stack) | (~np.isfinite(native20))] = np.nan
            pred20[(pred20 == nod_pred) | (~np.isfinite(pred20))] = np.nan

            # --- Build true 300 m grid from native 20 m ET ---------------
            bf = infer_block_factor(ds_stack, target_m=300.0)
            if bf is None:
                raise RuntimeError(
                    "Could not infer 300 m block factor from dataset transform."
                )
            fx, fy = bf

            et300_block = block_mean_2d(native20, fx=fx, fy=fy)
            if et300_block is None:
                raise RuntimeError(
                    "block_mean_2d returned None for 300 m aggregation."
                )

            # Upsample block-mean 300 m grid back to ~20 m resolution
            et300_up = np.repeat(np.repeat(et300_block, fy, axis=0), fx, axis=1)

            # Match shape to 20 m rasters
            H, W = native20.shape
            et300_up = et300_up[:H, :W]

            # Mask any remaining invalids
            et300_up[~np.isfinite(et300_up)] = np.nan

    # --- Crop to "signal" area to avoid ugly edges ----------------------
    # Use predicted ET as signal: ET > 1 mm/dekad
    signal_mask = np.isfinite(pred20) & (pred20 > 1.0)

    if np.any(signal_mask):
        rows, cols = np.where(signal_mask)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
    else:
        valid_any = (
            np.isfinite(pred20) |
            np.isfinite(native20) |
            np.isfinite(et300_up)
        )
        rows, cols = np.where(valid_any)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1

    pred20_c   = pred20[r0:r1, c0:c1]
    native20_c = native20[r0:r1, c0:c1]
    et300_c    = et300_up[r0:r1, c0:c1]

    # --- Align shapes for all three panels (safety for rounding issues) --
    Hc = min(pred20_c.shape[0], native20_c.shape[0], et300_c.shape[0])
    Wc = min(pred20_c.shape[1], native20_c.shape[1], et300_c.shape[1])

    pred20_c   = pred20_c[:Hc, :Wc]
    native20_c = native20_c[:Hc, :Wc]
    et300_c    = et300_c[:Hc, :Wc]

    # ---- Enforce common footprint for all three panels -----------------
    common_fp = np.isfinite(pred20_c) | np.isfinite(native20_c)
    et300_c[~common_fp] = np.nan

    # Color limits based on cropped area
    vals = np.concatenate([
        pred20_c[np.isfinite(pred20_c)],
        native20_c[np.isfinite(native20_c)],
        et300_c[np.isfinite(et300_c)],
    ])
    if vals.size == 0:
        raise RuntimeError("No valid ET values found for plotting after cropping.")
    vmin, vmax = np.percentile(vals, [2, 98])

    # ------------------------------------------------------------
    # Plot side by side with panel labels
    # ------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 3, figsize=(9, 3.2), sharex=True, sharey=True, constrained_layout=True
    )

    panels = [
        (pred20_c,   "Predicted ETa20m"),
        (native20_c, "Native WaPOR ETa20m"),
        (et300_c,    "Native ETa20m aggregated to 300 m"),
    ]
    panel_labels = ["(a)", "(b)", "(c)"]

    im = None
    for ax, (data, title), plab in zip(axes, panels, panel_labels):
        im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.text(
            0.02, 0.06, plab,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="white",
            bbox=dict(
                facecolor="black",
                alpha=0.35,
                pad=1.5,
                edgecolor="none",
            ),
        )

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("ETa (mm/dekad)")

    fig.suptitle(
        f"{site_title} ET comparison – {best_date} (max R² = {best_r2:.2f})",
        fontsize=11
    )

    out_png = out_dir / f"{site}_best_ET_map.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"[{site_upper}] Saved side-by-side map to: {out_png}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics-csv",
        type=str,
        required=True,
        help="Path to per_file_metrics_<site>_TESTONLY.csv"
    )
    ap.add_argument(
        "--site",
        type=str,
        default="lamego",
        choices=["lamego", "baixo"],
        help="Site name (controls preds_<site> subdir and title)."
    )
    ap.add_argument(
        "--label-name",
        type=str,
        default="b1",
        help="Label band name used in stack (default: b1)"
    )
    args = ap.parse_args()

    plot_best_site_map(
        metrics_csv=args.metrics_csv,
        site=args.site,
        label_name=args.label_name,
    )


if __name__ == "__main__":
    main()