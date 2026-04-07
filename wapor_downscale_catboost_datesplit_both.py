#!/usr/bin/env python3
# ============================================================
# WaPOR 20m Downscaling — CatBoost (CPU/GPU) + Feature Subset
# Selection + TPOT (optional)
#
# - Train on:
#     • BAIXO:  2018-01-01 → 2022-12-16
#     • LAMEGO: 2019-01-21 → 2020-12-20
# - Per-file evaluation on:
#     • LAMEGO: all other dates (hold-out)
#     • BAIXO:  all other dates (hold-out)
# - Metrics:
#     • RMSE, MAE, R², Relative RMSE (%) for eval
# - Works fully from the command line (VS Code / terminal / Colab)
#
# GPU:
#   Use: --task-type GPU --gpu-devices 0
# CPU:
#   Default: --task-type CPU
# ============================================================

import os, re, glob, json, math, time, argparse, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

# --- Compatibility patch for NumPy >= 2.0 (for TPOT and old libs) ---
if not hasattr(np, "float"):
    np.float = float   # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int       # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool     # type: ignore[attr-defined]
# --------------------------------------------------------------------

import rasterio
from rasterio.windows import Window
from rasterio.env import Env as RasterioEnv

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from catboost import CatBoostRegressor

import joblib

# ---------------------------
# Date windows (train)
# ---------------------------
BAIXO_TRAIN_START   = dt.date(2018, 1, 1)
BAIXO_TRAIN_END     = dt.date(2022, 12, 16)

LAMEGO_TRAIN_START  = dt.date(2019, 1, 21)
LAMEGO_TRAIN_END    = dt.date(2020, 12, 20)

# ---------------------------
# Constants / bands
# ---------------------------
FALLBACK_NODATA = -9999.0
NODATA_OUT = -9999.0

FEATURE_BANDS = (
    "B8", "B11", "B4",
    "NDVI", "NDMI", "FVC",
    "ETa300m",
    "DEM", "Slope",
    "Aspect_sin", "Aspect_cos",
    "RAIN_10d", "RAIN_10d_lag",
    "SIN_DOY", "COS_DOY",
)
REQUIRED_BANDS = ("B4", "B8", "B11", "ETa300m", "DEM", "Slope")

WC_URBAN = 50
WC_WATER = 80

# ============================================================
# Utils
# ============================================================
def seed_everything(seed=7):
    np.random.seed(seed)

def parse_any_date(text):
    m = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", text) or re.search(r"(\d{4})(\d{2})(\d{2})", text)
    if not m:
        return None, None
    y, mo, d = map(int, m.groups())
    try:
        dte = dt.date(y, mo, d)
    except ValueError:
        return None, None
    doy = (dte - dt.date(y, 1, 1)).days + 1
    return dte.isoformat(), doy

def extract_date(text):
    dstr, _ = parse_any_date(text)
    if dstr is None:
        return None
    try:
        return dt.date.fromisoformat(dstr)
    except Exception:
        return None

def sincos_doy(doy: int):
    ang = 2.0 * math.pi * (doy / 365.0)
    return float(math.sin(ang)), float(math.cos(ang))

def list_all_tifs(root):
    patterns = ["**/*.tif","**/*.tiff","**/*.TIF","**/*.TIFF"]
    files = []
    for pat in patterns:
        files += glob.glob(os.path.join(root, pat), recursive=True)
    return sorted(set(files))

def count_tifs(root):
    if not root or not os.path.exists(root):
        return 0
    return len(list_all_tifs(root))

def resolve_data_dir(data_dir: str):
    """
    Auto-discover BAIXO training directory if --data-dir is empty or wrong.
    Adjusted for VS Code/local layout.
    """
    if data_dir and os.path.exists(data_dir) and count_tifs(data_dir) > 0:
        return data_dir
    fallbacks = [
        "wapor_downscale_data/BAIXO_STACK_S2_MATCH_L3_20M_FULL_1",
        "wapor_downscale_data/BAIXO_STACK_S2_MATCH_L3_20M",
        "wapor_downscale_data/BAIXO",
    ]
    for d in fallbacks:
        if os.path.exists(d) and count_tifs(d) > 0:
            print(f"[AUTO] Using discovered TRAIN (BAIXO) data dir: {d}")
            return d
    return data_dir

def resolve_eval_dir(eval_dir: str):
    """
    Auto-discover LAMEGO evaluation directory if --eval-dir is empty or wrong.
    Adjusted for VS Code/local layout.
    """
    if eval_dir and os.path.exists(eval_dir) and count_tifs(eval_dir) > 0:
        return eval_dir
    fallbacks = [
        "wapor_downscale_data/LAMEGO_STACK_S2_MATCH_L3_20M_FULL_1",
        "wapor_downscale_data/LAMEGO_STACK_S2_MATCH_L3_20M",
        "wapor_downscale_data/LAMEGO",
    ]
    for d in fallbacks:
        if os.path.exists(d) and count_tifs(d) > 0:
            print(f"[AUTO] Using discovered EVAL (LAMEGO) data dir: {d}")
            return d
    return None

# ============================================================
# Raster band mapping (robust aliases)
# ============================================================
def band_map(ds):
    desc = list(ds.descriptions or [])
    bm = {desc[i]: i for i in range(ds.count) if i < len(desc) and desc[i]}
    if bm:
        return _normalize_bm(bm)

    bm2 = {}
    for i in range(ds.count):
        tag = ds.tags(i+1) or {}
        for key in ("BANDNAME","band_name","NAME","BandName","DESCRIPTION","description","name"):
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
        "B4":  ["b4","B04","red","RED","band4"],
        "B8":  ["b8","B08","nir","NIR","band8"],
        "B11": ["b11","swir","SWIR","swir1","SWIR1","band11"],

        "ETa300m": ["eta300m","ETa_300m","ETa300","ETa300m","ET300m","AETI300","AETI_300m","L1-AETI-D","L1_AETI_D"],
        "DEM":   ["dem","elevation","ELEVATION","elev"],
        "Slope": ["slope","SLOPE","terrain_slope","SLOPE_DEG"],

        "NDVI": ["ndvi","NDVI"],
        "NDMI": ["ndmi","NDMI"],
        "FVC":  ["fvc","FVC"],

        "Aspect":     ["aspect","ASPECT","Aspect_deg","aspect_deg"],
        "Aspect_sin": ["aspect_sin","ASPECT_SIN"],
        "Aspect_cos": ["aspect_cos","ASPECT_COS"],

        "RAIN_10d":     ["rain_10d","RAIN_10D","precip_10d","P10D","RAIN10D"],
        "RAIN_10d_lag": ["rain_10d_lag","RAIN_10D_LAG","precip_10d_lag","P10D_LAG","RAIN10D_LAG"],

        "b1": ["B1","b1","label","LABEL","target","y","ETa20","ETa20m","AETI20"],

        "WorldCover": ["worldcover","WorldCover","ESA_WorldCover","ESAWC","WC","lc","LC","landcover","LandCover"],
    }

    for canonical, keys in aliases.items():
        if canonical in out:
            continue
        for kk in keys:
            if kk in out:
                out[canonical] = out[kk]; break
            if kk.lower() in out:
                out[canonical] = out[kk.lower()]; break
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

# ============================================================
# Discovery
# ============================================================
def is_candidate_stack(fp: str) -> bool:
    p = fp.replace("\\", "/").lower()
    if "/preds" in p or "/tb" in p or "/cache_tfrecord" in p or "/cache_npz" in p:
        return False
    return fp.lower().endswith((".tif",".tiff"))

def file_has_required_bands(fp: str) -> bool:
    try:
        with rasterio.open(fp) as ds:
            bm = band_map(ds)
            if not bm:
                return False
            ok_feats = all(k in bm for k in REQUIRED_BANDS)
            lb = get_label_band_index(bm, "b1")
            return ok_feats and (lb is not None)
    except Exception:
        return False

def split_files_by_date(files, train_start, train_end):
    """Split into train_files (within [start,end]) and eval_files (outside)."""
    train_files, eval_files = [], []
    for fp in files:
        base = os.path.basename(fp)
        d = extract_date(base)
        if d is None:
            # ambiguous: send to train by default
            train_files.append(fp)
            continue
        if (train_start is not None and d < train_start) or (train_end is not None and d > train_end):
            eval_files.append(fp)
        else:
            train_files.append(fp)
    return sorted(train_files), sorted(eval_files)

# ============================================================
# Feature engineering
# ============================================================
def _sanitize_band(a, nod):
    a = a.astype(np.float32, copy=False)
    a[~np.isfinite(a)] = np.nan
    a[a == nod] = np.nan
    return a

def _safe_index(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out = out.astype(np.float32)
    out[~np.isfinite(out)] = np.nan
    return out

def _fvc_from_ndvi(ndvi, ndvi_soil=0.2, ndvi_veg=0.86):
    f = (ndvi - ndvi_soil) / max(1e-6, (ndvi_veg - ndvi_soil))
    f = np.clip(f, 0.0, 1.0).astype(np.float32)
    return f

def normalize_X_chw(X):
    X = X.astype(np.float32, copy=False)
    # reflectance *10000
    X[0] /= 10000.0
    X[1] /= 10000.0
    X[2] /= 10000.0
    # ETa300m
    X[6] /= 100.0
    # DEM, slope
    X[7] /= 3000.0
    X[8] /= 90.0
    # rain
    X[11] /= 200.0
    X[12] /= 200.0
    return X

def read_features_window(ds, bm, win: Window, doy: int):
    nod = get_effective_nodata(ds)

    b8  = _sanitize_band(ds.read(bm["B8"] + 1,    window=win), nod)
    b11 = _sanitize_band(ds.read(bm["B11"] + 1,   window=win), nod)
    b4  = _sanitize_band(ds.read(bm["B4"] + 1,    window=win), nod)
    et  = _sanitize_band(ds.read(bm["ETa300m"]+1, window=win), nod)
    dem = _sanitize_band(ds.read(bm["DEM"] + 1,   window=win), nod)
    slp = _sanitize_band(ds.read(bm["Slope"]+ 1,  window=win), nod)

    if "NDVI" in bm:
        ndvi = _sanitize_band(ds.read(bm["NDVI"] + 1, window=win), nod)
    else:
        ndvi = _safe_index(b8 - b4, b8 + b4)

    if "NDMI" in bm:
        ndmi = _sanitize_band(ds.read(bm["NDMI"] + 1, window=win), nod)
    else:
        ndmi = _safe_index(b8 - b11, b8 + b11)

    if "FVC" in bm:
        fvc = _sanitize_band(ds.read(bm["FVC"] + 1, window=win), nod)
    else:
        fvc = _fvc_from_ndvi(ndvi)

    if ("Aspect_sin" in bm) and ("Aspect_cos" in bm):
        asp_sin = _sanitize_band(ds.read(bm["Aspect_sin"] + 1, window=win), nod)
        asp_cos = _sanitize_band(ds.read(bm["Aspect_cos"] + 1, window=win), nod)
    elif "Aspect" in bm:
        aspect_deg = _sanitize_band(ds.read(bm["Aspect"] + 1, window=win), nod)
        ang = np.deg2rad(aspect_deg.astype(np.float32))
        asp_sin = np.sin(ang).astype(np.float32)
        asp_cos = np.cos(ang).astype(np.float32)
        asp_sin[~np.isfinite(asp_sin)] = np.nan
        asp_cos[~np.isfinite(asp_cos)] = np.nan
    else:
        H, W = b8.shape
        asp_sin = np.zeros((H, W), np.float32)
        asp_cos = np.zeros((H, W), np.float32)

    if "RAIN_10d" in bm:
        rain = _sanitize_band(ds.read(bm["RAIN_10d"] + 1, window=win), nod)
    else:
        H, W = b8.shape
        rain = np.zeros((H, W), np.float32)

    if "RAIN_10d_lag" in bm:
        rain_lag = _sanitize_band(ds.read(bm["RAIN_10d_lag"] + 1, window=win), nod)
    else:
        H, W = b8.shape
        rain_lag = np.zeros((H, W), np.float32)

    sd, cd = sincos_doy(int(doy) if doy else 1)
    H, W = b8.shape
    sin_doy = np.full((H, W), sd, np.float32)
    cos_doy = np.full((H, W), cd, np.float32)

    X = np.stack([
        b8, b11, b4,
        ndvi, ndmi, fvc,
        et,
        dem, slp,
        asp_sin, asp_cos,
        rain, rain_lag,
        sin_doy, cos_doy,
    ], axis=0).astype(np.float32)

    valid = np.isfinite(X).all(axis=0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X = normalize_X_chw(X)
    X = np.clip(X, -5.0, 5.0).astype(np.float32)
    return X, valid

def read_label_window(ds, bm, win: Window, label_name="b1"):
    nod = get_effective_nodata(ds)
    lb0 = get_label_band_index(bm, label_name)
    if lb0 is None:
        raise RuntimeError("Label band not found.")
    lb_idx = lb0 + 1
    y = ds.read(lb_idx, window=win).astype(np.float32)
    y[y == nod] = np.nan
    y[~np.isfinite(y)] = np.nan
    m = np.isfinite(y).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0).astype(np.float32)
    return y, m

def read_worldcover_full(ds, bm):
    if "WorldCover" not in bm:
        return None
    nod = get_effective_nodata(ds)
    wc = ds.read(bm["WorldCover"] + 1).astype(np.int16)
    wc = np.where(wc == int(nod), -9999, wc)
    return wc

# ============================================================
# Feature groups
# ============================================================
FEATURE_INDEX = {b: i for i, b in enumerate(FEATURE_BANDS)}

FEATURE_GROUPS = {
    "s2":      ["B8", "B11", "B4"],
    "idx":     ["NDVI", "NDMI", "FVC"],
    "eta300":  ["ETa300m"],
    "terrain": ["DEM", "Slope", "Aspect_sin", "Aspect_cos"],
    "rain":    ["RAIN_10d", "RAIN_10d_lag"],
    "doy":     ["SIN_DOY", "COS_DOY"],
}

def bands_to_cols(bands):
    cols = []
    for b in bands:
        if b not in FEATURE_INDEX:
            raise ValueError(f"Band not found in FEATURE_BANDS: {b}")
        cols.append(FEATURE_INDEX[b])
    return cols

def groups_to_cols(group_names):
    bands = []
    for g in group_names:
        bands += FEATURE_GROUPS[g]
    return bands_to_cols(bands), bands

# ============================================================
# Sampling
# ============================================================
def sample_training_from_files(
    files,
    n_samples_total,
    patch=256,
    seed=7,
    label_name="b1",
    min_valid_frac=0.05,
    max_slope_deg=25.0,
    exclude_urban=False,
    exclude_water=False,
    veg_sample_frac=0.7,
    veg_ndvi_thr=0.35,
    veg_fvc_thr=0.2,
    per_file_min=500,
    per_file_max=4000,
    et_stratify=False,
    et_q_low=0.33,
    et_q_high=0.66,
):
    """
    Returns:
      X: (N, C) float32
      y: (N,)  float32
      meta_df: per-sample metadata (file, row, col, date)
    """
    rng = np.random.default_rng(seed)
    gdal_opts = {"GDAL_CACHEMAX": 2048, "GDAL_NUM_THREADS": "ALL_CPUS"}

    files = list(files)
    if not files:
        raise RuntimeError("No input files to sample.")

    base_per_file = max(1, int(math.ceil(n_samples_total / len(files))))
    per_file_target = max(int(per_file_min), min(int(per_file_max), base_per_file))

    X_list, y_list = [], []
    meta = []

    t0 = time.time()
    got = 0

    def pick_indices_et_stratified(et_vals, take, rng_local):
        et_vals = et_vals.astype(np.float32)
        mask_finite = np.isfinite(et_vals)
        if mask_finite.sum() < 10:
            return None
        et_clean = et_vals[mask_finite]
        try:
            q_low = float(et_q_low)
            q_high = float(et_q_high)
            if not (0.0 < q_low < q_high < 1.0):
                q_low, q_high = 0.33, 0.66
            q1, q2 = np.nanquantile(et_clean, [q_low, q_high])
        except Exception:
            return None

        idx_all = np.arange(et_vals.size)

        lo_mask = (et_vals <= q1) & np.isfinite(et_vals)
        mid_mask = (et_vals > q1) & (et_vals <= q2) & np.isfinite(et_vals)
        hi_mask = (et_vals > q2) & np.isfinite(et_vals)

        idx_lo = idx_all[lo_mask]
        idx_mid = idx_all[mid_mask]
        idx_hi = idx_all[hi_mask]

        bins = [idx_lo, idx_mid, idx_hi]
        n_bins = len(bins)

        take = int(take)
        if take <= 0:
            return None

        base = take // n_bins
        rem = take - base * n_bins

        chosen = []
        for i, idx_bin in enumerate(bins):
            if idx_bin.size == 0:
                continue
            want = base + (1 if i < rem else 0)
            want = min(want, idx_bin.size)
            if want > 0:
                chosen.append(rng_local.choice(idx_bin, size=want, replace=False))

        if chosen:
            sel = np.concatenate(chosen)
            if sel.size < take and sel.size < et_vals.size:
                rest = np.setdiff1d(idx_all, sel, assume_unique=False)
                if rest.size > 0:
                    extra = rng_local.choice(rest, size=min(take - sel.size, rest.size), replace=False)
                    sel = np.concatenate([sel, extra])
            return sel
        else:
            return None

    with RasterioEnv(**gdal_opts):
        for fp in files:
            base = os.path.basename(fp)
            dstr, doy = parse_any_date(base)
            doy = 1 if doy is None else int(doy)

            with rasterio.open(fp) as ds:
                bm = band_map(ds)
                nod = get_effective_nodata(ds)

                lb0 = get_label_band_index(bm, label_name)
                if lb0 is None:
                    continue
                lb_idx = lb0 + 1

                lab = ds.read(lb_idx).astype(np.float32)
                lab[lab == nod] = np.nan
                lab[~np.isfinite(lab)] = np.nan
                ys, xs = np.where(np.isfinite(lab))
                if ys.size == 0:
                    continue

                mask_ok = np.ones((ds.height, ds.width), dtype=bool)

                if max_slope_deg is not None:
                    slp_full = ds.read(bm["Slope"] + 1).astype(np.float32)
                    slp_full[slp_full == nod] = np.nan
                    mask_ok &= np.isfinite(slp_full) & (slp_full <= float(max_slope_deg))

                if (exclude_urban or exclude_water) and ("WorldCover" in bm):
                    wc_full = read_worldcover_full(ds, bm)
                    if wc_full is not None:
                        if exclude_urban:
                            mask_ok &= (wc_full != WC_URBAN)
                        if exclude_water:
                            mask_ok &= (wc_full != WC_WATER)
                elif (exclude_urban or exclude_water) and ("WorldCover" not in bm):
                    print(f"[SAMPLE][WARN] {base}: exclude_urban/water requested but no WorldCover band found.")

                keep = mask_ok[ys, xs]
                ys2, xs2 = ys[keep], xs[keep]
                if ys2.size == 0:
                    continue

                veg = None
                try:
                    if "NDVI" in bm:
                        ndvi_full = ds.read(bm["NDVI"] + 1).astype(np.float32)
                        ndvi_full[ndvi_full == nod] = np.nan
                    else:
                        b8_full = ds.read(bm["B8"] + 1).astype(np.float32); b8_full[b8_full == nod] = np.nan
                        b4_full = ds.read(bm["B4"] + 1).astype(np.float32); b4_full[b4_full == nod] = np.nan
                        ndvi_full = (b8_full - b4_full) / (b8_full + b4_full + 1e-6)

                    if "FVC" in bm:
                        fvc_full = ds.read(bm["FVC"] + 1).astype(np.float32)
                        fvc_full[fvc_full == nod] = np.nan
                    else:
                        fvc_full = _fvc_from_ndvi(ndvi_full)

                    ndv = ndvi_full[ys2, xs2]
                    fvc = fvc_full[ys2, xs2]
                    veg = (np.isfinite(ndv) & (ndv >= float(veg_ndvi_thr))) | (np.isfinite(fvc) & (fvc >= float(veg_fvc_thr)))
                except Exception:
                    veg = None

                take = min(per_file_target, ys2.size)
                remaining = n_samples_total - got
                if remaining <= 0:
                    break
                take = min(take, remaining)
                if take <= 0:
                    continue

                if et_stratify:
                    et_vals_all = lab[ys2, xs2]
                    et_sel = pick_indices_et_stratified(et_vals_all, take, rng)
                    if et_sel is not None and et_sel.size > 0:
                        sel = et_sel
                    else:
                        if veg is not None:
                            veg_idx = np.where(veg)[0]
                            non_idx = np.where(~veg)[0]
                            n_veg = int(round(take * float(veg_sample_frac)))
                            n_non = take - n_veg
                            n_veg = min(n_veg, veg_idx.size)
                            n_non = min(n_non, non_idx.size)
                            pick = []
                            if n_veg > 0:
                                pick.append(rng.choice(veg_idx, size=n_veg, replace=False))
                            if n_non > 0:
                                pick.append(rng.choice(non_idx, size=n_non, replace=False))
                            if pick:
                                sel = np.concatenate(pick)
                                if sel.size < take:
                                    rem = take - sel.size
                                    rest = np.setdiff1d(np.arange(ys2.size), sel, assume_unique=False)
                                    if rest.size > 0:
                                        sel = np.concatenate(
                                            [sel, rng.choice(rest, size=min(rem, rest.size), replace=False)]
                                        )
                        else:
                            sel = rng.choice(ys2.size, size=take, replace=False)
                else:
                    if veg is None:
                        sel = rng.choice(ys2.size, size=take, replace=False)
                    else:
                        veg_idx = np.where(veg)[0]
                        non_idx = np.where(~veg)[0]
                        n_veg = int(round(take * float(veg_sample_frac)))
                        n_non = take - n_veg
                        n_veg = min(n_veg, veg_idx.size)
                        n_non = min(n_non, non_idx.size)
                        pick = []
                        if n_veg > 0:
                            pick.append(rng.choice(veg_idx, size=n_veg, replace=False))
                        if n_non > 0:
                            pick.append(rng.choice(non_idx, size=n_non, replace=False))
                        if pick:
                            sel = np.concatenate(pick)
                            if sel.size < take:
                                rem = take - sel.size
                                rest = np.setdiff1d(np.arange(ys2.size), sel, assume_unique=False)
                                if rest.size > 0:
                                    sel = np.concatenate(
                                        [sel, rng.choice(rest, size=min(rem, rest.size), replace=False)]
                                    )
                        else:
                            sel = rng.choice(ys2.size, size=take, replace=False)

                half = patch // 2
                for j in sel:
                    cy, cx = int(ys2[j]), int(xs2[j])
                    y0 = max(0, min(cy - half, ds.height - patch))
                    x0 = max(0, min(cx - half, ds.width  - patch))
                    win = Window(x0, y0, patch, patch)

                    X_chw, valid = read_features_window(ds, bm, win, doy)
                    y_patch, m_patch = read_label_window(ds, bm, win, label_name=label_name)

                    ry = cy - y0
                    rx = cx - x0
                    if ry < 0 or rx < 0 or ry >= patch or rx >= patch:
                        continue

                    if not valid[ry, rx]:
                        continue
                    if m_patch[ry, rx] < 0.5:
                        continue

                    m_both = (m_patch > 0.5) & valid
                    if float(m_both.mean()) < float(min_valid_frac):
                        continue

                    x_vec = X_chw[:, ry, rx].astype(np.float32)
                    y_val = float(y_patch[ry, rx])

                    X_list.append(x_vec)
                    y_list.append(y_val)
                    meta.append({"file": base, "date": dstr, "row": cy, "col": cx})

                    got += 1
                    if got >= n_samples_total:
                        break

            if got >= n_samples_total:
                break

    if not X_list:
        raise RuntimeError("Sampling produced 0 samples. Check masks / nodata / label band.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta)

    dt_min = (time.time() - t0) / 60.0
    print(f"[SAMPLE] got={len(y)} / requested={n_samples_total} | time={dt_min:.2f} min")
    return X, y, meta_df

# ============================================================
# Metrics
# ============================================================
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def rrmse_percent(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    denom = float(np.nanmean(y_true)) if y_true.size > 0 else float("nan")
    if (not np.isfinite(denom)) or denom == 0.0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)) / denom * 100.0)

# ============================================================
# Subset selection (CatBoost)
# ============================================================
def sample_cb_params(rng, base_params):
    """
    Sample a set of CatBoostRegressor hyperparameters.
    base_params is a dict with defaults (loss_function, eval_metric, etc.).
    """
    params = dict(base_params)

    # number of trees
    params["iterations"] = int(rng.choice([400, 800, 1200]))

    # depth
    params["depth"] = int(rng.choice([4, 6, 8, 10]))

    # learning rate
    params["learning_rate"] = float(rng.choice([0.03, 0.05, 0.1]))

    # L2 regularization
    params["l2_leaf_reg"] = float(rng.choice([1.0, 3.0, 5.0, 7.0, 10.0]))

    # bagging / Bayesian bootstrap strength
    params["bagging_temperature"] = float(rng.choice([0.0, 0.5, 1.0, 2.0]))

    return params

def subset_search(
    X_train, y_train,
    X_val, y_val,
    group_names_all,
    n_trials=40,
    min_groups=2,
    seed=7,
    base_params=None,
    verbose=True,
):
    rng = np.random.default_rng(seed)
    base_params = base_params or {}
    results = []
    best = None

    for t in range(1, int(n_trials) + 1):
        # random subset of feature groups
        k = int(rng.integers(min_groups, len(group_names_all) + 1))
        chosen_groups = sorted(rng.choice(group_names_all, size=k, replace=False).tolist())

        cols, bands = groups_to_cols(chosen_groups)

        # sample CatBoost params
        params = sample_cb_params(rng, base_params)
        params["random_seed"] = int(seed + t)

        model = CatBoostRegressor(**params)

        model.fit(
            X_train[:, cols],
            y_train,
            eval_set=(X_val[:, cols], y_val),
            use_best_model=True,
            verbose=False,
        )

        pred = model.predict(X_val[:, cols])
        s_rmse   = rmse(y_val, pred)
        s_r2     = float(r2_score(y_val, pred))
        s_mae    = mae(y_val, pred)
        s_rrmse  = rrmse_percent(y_val, pred)

        row = {
            "trial": t,
            "groups": ",".join(chosen_groups),
            "n_groups": len(chosen_groups),
            "bands": ",".join(bands),
            "n_bands": len(bands),
            "rmse": s_rmse,
            "mae": s_mae,
            "r2": s_r2,
            "rrmse_pct": s_rrmse,
            **{f"p_{k}": v for k, v in params.items()},
        }
        results.append(row)

        if verbose:
            print(
                f"[SEARCH] {t:03d}/{n_trials} "
                f"rmse={s_rmse:.4f} mae={s_mae:.4f} r2={s_r2:.3f} "
                f"rrmse%={s_rrmse:.2f} groups={chosen_groups}"
            )

        if best is None or s_rmse < best["rmse"]:
            best = row

    df = pd.DataFrame(results).sort_values("rmse", ascending=True)
    return best, df

# ============================================================
# Tile-wise prediction for per-file eval
# ============================================================
def predict_file_model(fp_in, model, cols, tile=2048, label_name="b1"):
    base = os.path.basename(fp_in)
    _, doy = parse_any_date(base)
    doy = 1 if doy is None else int(doy)

    with rasterio.open(fp_in) as ds:
        bm = band_map(ds)
        H, W = ds.height, ds.width

        pred_full = np.full((H, W), NODATA_OUT, np.float32)
        y_full = np.zeros((H, W), np.float32)
        m_full = np.zeros((H, W), np.float32)

        for y0 in range(0, H, tile):
            for x0 in range(0, W, tile):
                h0 = min(tile, H - y0)
                w0 = min(tile, W - x0)
                win = Window(x0, y0, w0, h0)

                X_chw, valid = read_features_window(ds, bm, win, doy)
                y_win, m_win = read_label_window(ds, bm, win, label_name=label_name)

                X_hwC = np.transpose(X_chw, (1, 2, 0))
                X_flat = X_hwC.reshape(-1, X_hwC.shape[-1]).astype(np.float32)
                valid_flat = valid.reshape(-1)
                m_flat = (m_win > 0.5).reshape(-1) & valid_flat

                out = np.full((h0 * w0,), NODATA_OUT, np.float32)
                if np.any(valid_flat):
                    pred_valid = model.predict(X_flat[valid_flat][:, cols]).astype(np.float32)
                    out_idx = np.where(valid_flat)[0]
                    out[out_idx] = pred_valid

                pred_tile = out.reshape(h0, w0)
                pred_full[y0:y0+h0, x0:x0+w0] = pred_tile

                y_full[y0:y0+h0, x0:x0+w0] = y_win
                m_full[y0:y0+h0, x0:x0+w0] = (m_flat.reshape(h0, w0)).astype(np.float32)

        m = m_full > 0.5
        if m.sum() < 2:
            return pred_full, float("nan"), float("nan"), float("nan"), float("nan"), int(m.sum())

        yy = y_full[m].astype(np.float32)
        pp = pred_full[m].astype(np.float32)
        s_rmse  = rmse(yy, pp)
        s_r2    = float(r2_score(yy, pp))
        s_mae   = mae(yy, pp)
        s_rrmse = rrmse_percent(yy, pp)
        return pred_full, s_rmse, s_mae, s_r2, s_rrmse, int(m.sum())

def plot_timeseries(csv_path: Path, out_prefix: Path, title_prefix="BAIXO (hold-out)"):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_csv(csv_path)

    # x-axis: date if present, else index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
        x = df["date"]
        xlab = "Date"
    else:
        x = np.arange(len(df))
        xlab = "File"

    def _style_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    outputs = {}

    # --- R² ---
    p = out_prefix.with_name(out_prefix.stem + "_r2.png")
    fig, ax = plt.subplots()
    ax.plot(x, df["r2"].values)
    ax.set_title(f"{title_prefix} per-file R²")
    ax.set_xlabel(xlab)
    ax.set_ylabel("R²")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    outputs["r2"] = p

    # --- RMSE ---
    p = out_prefix.with_name(out_prefix.stem + "_rmse.png")
    fig, ax = plt.subplots()
    ax.plot(x, df["rmse"].values)
    ax.set_title(f"{title_prefix} per-file RMSE")
    ax.set_xlabel(xlab)
    ax.set_ylabel("RMSE")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    outputs["rmse"] = p

    # --- Relative RMSE (%) ---
    # your CSV uses "rrmse_pct" (from earlier scripts)
    rrmse_col = "rrmse_pct" if "rrmse_pct" in df.columns else ("rrmse" if "rrmse" in df.columns else None)
    if rrmse_col:
        p = out_prefix.with_name(out_prefix.stem + "_rrmse_pct.png")
        fig, ax = plt.subplots()
        ax.plot(x, df[rrmse_col].values)
        ax.set_title(f"{title_prefix} per-file Relative RMSE (%)")
        ax.set_xlabel(xlab)
        ax.set_ylabel("Relative RMSE (%)")
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        outputs["rrmse_pct"] = p

    # --- MAE ---
    if "mae" in df.columns:
        p = out_prefix.with_name(out_prefix.stem + "_mae.png")
        fig, ax = plt.subplots()
        ax.plot(x, df["mae"].values)
        ax.set_title(f"{title_prefix} per-file MAE")
        ax.set_xlabel(xlab)
        ax.set_ylabel("MAE")
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        outputs["mae"] = p

    return outputs

# ============================================================
# W&B (optional)
# ============================================================
def setup_wandb(args):
    wants_artifact_download = bool(args.wandb_train_dataset or args.wandb_eval_dataset)
    mode = (args.wandb_mode or "online").lower()
    if mode not in ("online", "offline", "disabled"):
        mode = "online"
    wants_logging = (mode != "disabled")

    if not wants_logging and not wants_artifact_download:
        print("[W&B] disabled (set --wandb-mode online/offline to enable logging, or provide --wandb-train-dataset/--wandb-eval-dataset to download artifacts).")
        return None, None, None

    if (not args.wandb_entity):
        print("[W&B][ERROR] wandb-entity is missing.")
        print("          Please re-run with e.g.: --wandb-entity your_username_or_team")
        print("          W&B will be disabled for this run.")
        return None, None, None

    import wandb
    import requests
    from urllib3.exceptions import ProtocolError

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name if args.wandb_run_name else None,
        job_type=args.wandb_job_type,
        mode=mode,
        config={
            "feature_bands": list(FEATURE_BANDS),
            "feature_groups": FEATURE_GROUPS,
            "n_samples_total": args.n_samples_total,
            "val_frac": args.val_frac,
            "subset_trials": args.subset_trials,
            "min_groups": args.min_groups,
            "cb_base_params": {},
            "tile": args.tile,
            "exclude_urban": bool(args.exclude_urban),
            "exclude_water": bool(args.exclude_water),
            "max_slope_deg": args.max_slope_deg,
            "veg_sample_frac": args.veg_sample_frac,
            "veg_ndvi_thr": args.veg_ndvi_thr,
            "veg_fvc_thr": args.veg_fvc_thr,
            "per_file_min": args.per_file_min,
            "per_file_max": args.per_file_max,
            "et_stratify": args.et_stratify,
            "et_q_low": args.et_q_low,
            "et_q_high": args.et_q_high,
            "wandb_train_dataset": args.wandb_train_dataset,
            "wandb_eval_dataset": args.wandb_eval_dataset,
            "task_type": args.task_type,
            "gpu_devices": args.gpu_devices,
        },
    )
    print("[W&B] run url:", run.url)
    print("[W&B] entity:", args.wandb_entity, "| project:", args.wandb_project)
    if wants_logging and not args.wandb:
        print("[W&B] logging enabled by default; use --wandb-mode disabled to turn it off.")
    if wants_artifact_download and not wants_logging:
        print("[W&B] artifact download enabled without experiment logging.")
    if args.wandb_log_local_datasets:
        print("[W&B] local dataset artifact upload enabled.")
        print("[W&B] make the target project public if you want anyone to download these artifacts.")

    def safe_log_dataset_artifact(dir_path: str, artifact_name: str, label: str):
        if not args.wandb_log_local_datasets:
            return
        if not dir_path:
            return
        p = Path(dir_path)
        if not p.exists() or not p.is_dir():
            print(f"[W&B][WARN] skipping {label} dataset upload; directory not found:", p)
            return
        try:
            art = wandb.Artifact(artifact_name, type="dataset")
            art.add_dir(str(p))
            run.log_artifact(art)
            print(f"[W&B] uploaded {label} dataset artifact:", artifact_name)
        except Exception as e:
            print(f"[W&B][WARN] failed to upload {label} dataset artifact '{artifact_name}': {e}")

    def safe_download_artifact(name: str, label: str):
        if not name:
            return None
        try:
            print(f"[W&B] downloading {label} artifact:", name)
            art = run.use_artifact(name, type="dataset")
            d = art.download()
            print(f"[W&B] {label} dataset downloaded:", d)
            return d
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                ProtocolError) as e:
            print(f"[W&B][WARN] network error while downloading {label} artifact '{name}': {e}")
            print("[W&B][WARN] -> will fall back to local directories instead.")
            return None
        except Exception as e:
            print(f"[W&B][WARN] failed to download {label} artifact '{name}': {e}")
            print("[W&B][WARN] -> will fall back to local directories instead.")
            return None

    safe_log_dataset_artifact(args.data_dir, args.wandb_train_artifact_name, "train")
    safe_log_dataset_artifact(args.eval_dir, args.wandb_eval_artifact_name, "eval")

    train_dir = safe_download_artifact(args.wandb_train_dataset, "train")
    eval_dir  = safe_download_artifact(args.wandb_eval_dataset, "eval")

    return run, train_dir, eval_dir

# ============================================================
# TPOT helper (optional, still uses sklearn)
# ============================================================
def run_tpot_on_best_subset(X, y, best_cols, best_bands, out_dir, args):
    try:
        from tpot import TPOTRegressor
    except ImportError:
        print("[TPOT][WARN] tpot is not installed. Run `pip install tpot` and re-run with --run-tpot.")
        return None

    print("\n[TPOT] Starting TPOT search on best subset...")
    X_sel = X[:, best_cols]

    N = X_sel.shape[0]
    if N < 60:
        print(f"[TPOT][WARN] Too few samples for TPOT (N={N} < 60). Skipping TPOT.")
        return None

    max_samples = max(int(args.tpot_max_samples), 200)
    if N > max_samples:
        rs = np.random.RandomState(args.seed)
        idx = rs.choice(N, size=max_samples, replace=False)
        X_sel = X_sel[idx]
        y = y[idx]
        N = max_samples
        print(f"[TPOT] downsampled to {max_samples} samples")

    test_size = float(args.tpot_test_size)
    tr_size = int(N * (1.0 - test_size))
    if tr_size < 30:
        print(f"[TPOT][WARN] Too few training samples after split (train={tr_size}). Skipping TPOT.")
        return None

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_sel, y,
        test_size=test_size,
        random_state=int(args.seed),
        shuffle=True,
    )
    print("[TPOT] shapes: X_tr =", X_tr.shape, "| X_va =", X_va.shape)

    tpot = TPOTRegressor(
        generations=int(args.tpot_generations),
        population_size=int(args.tpot_population_size),
        random_state=int(args.seed),
        n_jobs=int(args.tpot_n_jobs),
    )

    t0 = time.time()
    tpot.fit(X_tr, y_tr)
    dt_min = (time.time() - t0) / 60.0
    print(f"[TPOT] finished search in {dt_min:.2f} min")

    y_pred = tpot.predict(X_va)
    val_rmse = rmse(y_va, y_pred)
    val_mae  = mae(y_va, y_pred)
    val_r2   = r2_score(y_va, y_pred)
    print(f"[TPOT] validation RMSE={val_rmse:.4f} MAE={val_mae:.4f} R2={val_r2:.3f}")

    model_path = out_dir / "tpot_best_model.joblib"
    joblib.dump(
        {
            "model": tpot.fitted_pipeline_,
            "selected_cols": best_cols,
            "selected_bands": best_bands,
            "val_rmse": float(val_rmse),
            "val_mae": float(val_mae),
            "val_r2": float(val_r2),
        },
        model_path,
    )
    print("[TPOT] saved fitted model:", model_path)

    return {
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_r2": val_r2,
        "pipeline_py": None,
        "model_path": str(model_path),
    }

# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-dir", type=str,
                    default="wapor_downscale_data/BAIXO_STACK_S2_MATCH_L3_20M_FULL_1")
    ap.add_argument("--out-dir", type=str, default="outputs_catboost_datesplit")

    # Sampling
    ap.add_argument("--n-samples-total", type=int, default=300000)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--label-name", type=str, default="b1")
    ap.add_argument("--min-valid-frac", type=float, default=0.05)

    ap.add_argument("--per-file-min", type=int, default=500)
    ap.add_argument("--per-file-max", type=int, default=4000)

    # Masks
    ap.add_argument("--max-slope-deg", type=float, default=25.0)
    ap.add_argument("--exclude-urban", action="store_true")
    ap.add_argument("--exclude-water", action="store_true")

    # Stratified sampling
    ap.add_argument("--veg-sample-frac", type=float, default=0.7)
    ap.add_argument("--veg-ndvi-thr", type=float, default=0.35)
    ap.add_argument("--veg-fvc-thr", type=float, default=0.2)

    # ET-based stratified sampling (optional; ETa, not ExtraTrees)
    ap.add_argument("--et-stratify", action="store_true")
    ap.add_argument("--et-q-low", type=float, default=0.33)
    ap.add_argument("--et-q-high", type=float, default=0.66)

    # Subset selection
    ap.add_argument("--subset-trials", type=int, default=40)
    ap.add_argument("--min-groups", type=int, default=2)
    ap.add_argument("--n-jobs", type=int, default=-1)

    # Eval on LAMEGO (and BAIXO hold-out)
    ap.add_argument("--tile", type=int, default=2048)
    ap.add_argument("--eval-max", type=int, default=0)
    ap.add_argument("--eval-dir", type=str,
                    default="wapor_downscale_data/LAMEGO_STACK_S2_MATCH_L3_20M_FULL_1")

    # Repro
    ap.add_argument("--seed", type=int, default=7)

    # CPU vs GPU
    ap.add_argument("--task-type", type=str, default="CPU", choices=["CPU", "GPU"],
                    help="CatBoost task_type: GPU or CPU")
    ap.add_argument("--gpu-devices", type=str, default="0",
                    help='GPU devices string for CatBoost, e.g. "0" or "0,1"')

    # W&B
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-entity", type=str, default="zolokiala-iwmi")
    ap.add_argument("--wandb-project", type=str, default="wapor-downscale-catboost")
    ap.add_argument("--wandb-run-name", type=str, default="")
    ap.add_argument("--wandb-mode", type=str, default="online")
    ap.add_argument("--wandb-job-type", type=str, default="train_catboost")
    ap.add_argument("--wandb-train-dataset", type=str, default="")
    ap.add_argument("--wandb-eval-dataset",  type=str, default="")
    ap.add_argument("--wandb-log-local-datasets", action="store_true")
    ap.add_argument("--wandb-train-artifact-name", type=str,
                    default="baixo_stack_s2_match_l3_20m_full_1")
    ap.add_argument("--wandb-eval-artifact-name", type=str,
                    default="lamego_stack_s2_match_l3_20m_full_1")

    # TPOT options
    ap.add_argument("--run-tpot", action="store_true")
    ap.add_argument("--tpot-max-samples", type=int, default=80000)
    ap.add_argument("--tpot-test-size", type=float, default=0.2)
    ap.add_argument("--tpot-generations", type=int, default=10)
    ap.add_argument("--tpot-population-size", type=int, default=30)
    ap.add_argument("--tpot-n-jobs", type=int, default=-1)

    args, _ = ap.parse_known_args()
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CATBOOST] task_type={args.task_type} | gpu_devices={args.gpu_devices}")

    # W&B (may override train/eval dirs via artifacts)
    wandb_run, train_dir_art, eval_dir_art = setup_wandb(args)

    # Train root = BAIXO (artifact or local)
    train_root = train_dir_art if train_dir_art else resolve_data_dir(args.data_dir)

    # Eval root = LAMEGO (artifact, explicit, or auto-discover)
    if eval_dir_art:
        eval_root = eval_dir_art
    elif args.eval_dir and os.path.exists(args.eval_dir):
        eval_root = args.eval_dir
    else:
        eval_root = resolve_eval_dir("")

    print("[TRAIN_DIR]", train_root, "| tifs:", count_tifs(train_root))
    if eval_root:
        print("[EVAL_DIR ]", eval_root, "| tifs:", count_tifs(eval_root))
    else:
        print("[EVAL_DIR ] <none> (LAMEGO disabled)")

    # -------------------------------
    # Build training & eval file sets
    # -------------------------------
    # BAIXO files from train_root
    all_baixo_files = [fp for fp in list_all_tifs(train_root)
                       if is_candidate_stack(fp) and file_has_required_bands(fp)]
    baixo_train_files, baixo_eval_files = split_files_by_date(
        all_baixo_files, BAIXO_TRAIN_START, BAIXO_TRAIN_END
    )

    # LAMEGO files (eval_root)
    lamego_train_files, lamego_eval_files = [], []
    if eval_root:
        all_lamego_files = [fp for fp in list_all_tifs(eval_root)
                            if is_candidate_stack(fp) and file_has_required_bands(fp)]
        lamego_train_files, lamego_eval_files = split_files_by_date(
            all_lamego_files, LAMEGO_TRAIN_START, LAMEGO_TRAIN_END
        )

    # Training: BAIXO_train + LAMEGO_train (union)
    train_files = sorted(set(baixo_train_files + lamego_train_files))

    # Eval: BAIXO_holdout + LAMEGO_holdout (separate lists)
    eval_files_baixo  = list(baixo_eval_files)
    eval_files_lamego = list(lamego_eval_files)

    if not baixo_train_files:
        raise RuntimeError("No valid BAIXO stacks found in training window to train.")
    if eval_root and not lamego_train_files:
        print("[WARN] No valid LAMEGO stacks found in training window; training only on BAIXO.")

    print(
        f"[FILES] BAIXO: train={len(baixo_train_files)} eval={len(eval_files_baixo)} | "
        f"LAMEGO: train={len(lamego_train_files)} eval={len(eval_files_lamego)} | "
        f"TRAIN UNION={len(train_files)}"
    )

    # 1) Sample BAIXO + LAMEGO (in training windows only)
    X, y, meta_df = sample_training_from_files(
        files=train_files,
        n_samples_total=int(args.n_samples_total),
        patch=int(args.patch),
        seed=int(args.seed),
        label_name=args.label_name,
        min_valid_frac=float(args.min_valid_frac),
        max_slope_deg=float(args.max_slope_deg) if args.max_slope_deg is not None else None,
        exclude_urban=bool(args.exclude_urban),
        exclude_water=bool(args.exclude_water),
        veg_sample_frac=float(args.veg_sample_frac),
        veg_ndvi_thr=float(args.veg_ndvi_thr),
        veg_fvc_thr=float(args.veg_fvc_thr),
        per_file_min=int(args.per_file_min),
        per_file_max=int(args.per_file_max),
        et_stratify=bool(args.et_stratify),
        et_q_low=float(args.et_q_low),
        et_q_high=float(args.et_q_high),
    )

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=float(args.val_frac),
        random_state=int(args.seed), shuffle=True
    )
    print(f"[SPLIT] X_tr={X_tr.shape} X_va={X_va.shape}")

    try:
        np.savez_compressed(
            out_dir / "baixo_lamego_samples_for_tpot.npz",
            X=X,
            y=y
        )
        print("[TPOT] saved sampled BAIXO+LAMEGO table:", out_dir / "baixo_lamego_samples_for_tpot.npz")
    except Exception as e:
        print("[TPOT][WARN] Could not save samples npz:", e)

    # 2) Subset selection with CatBoost
    group_names_all = sorted(list(FEATURE_GROUPS.keys()))

    base_params = dict(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=int(args.seed),
        thread_count=int(args.n_jobs),
        bootstrap_type="Bayesian",
        od_type="Iter",
        od_wait=40,
        task_type=args.task_type,
    )
    # Only relevant if GPU is used; safe for CPU
    if args.task_type.upper() == "GPU" and args.gpu_devices:
        base_params["devices"] = args.gpu_devices

    best, df_trials = subset_search(
        X_train=X_tr, y_train=y_tr,
        X_val=X_va, y_val=y_va,
        group_names_all=group_names_all,
        n_trials=int(args.subset_trials),
        min_groups=int(args.min_groups),
        seed=int(args.seed),
        base_params=base_params,
        verbose=True,
    )

    trials_csv = out_dir / "subset_trials.csv"
    df_trials.to_csv(trials_csv, index=False)
    print("[SUBSET] trials saved:", trials_csv)

    best_groups = best["groups"].split(",") if best["groups"] else []
    best_cols, best_bands = groups_to_cols(best_groups)

    # --- unpack CatBoost params from best row (fixing name quirks) ---
    raw_params = {k.replace("p_", ""): best[k] for k in best.keys() if k.startswith("p_")}

    # Map CatBoost's internal-ish names back to __init__ keyword args
    rename_map = {
        "bootstratype": "bootstrap_type",
        "odtype": "od_type",
        "randomseed": "random_seed",
        "learningrate": "learning_rate",
        "l2leafreg": "l2_leaf_reg",
        "baggingtemperature": "bagging_temperature",
        "threadcount": "thread_count",
        "tasktype": "task_type",
    }
    for old, new in rename_map.items():
        if old in raw_params and new not in raw_params:
            raw_params[new] = raw_params.pop(old)

    # Keep only known CatBoostRegressor __init__ params (defensive)
    allowed_keys = {
        "iterations",
        "depth",
        "learning_rate",
        "l2_leaf_reg",
        "bagging_temperature",
        "loss_function",
        "eval_metric",
        "random_seed",
        "thread_count",
        "bootstrap_type",
        "od_type",
        "od_wait",
        "task_type",
        "devices",
    }
    best_params = {k: v for k, v in raw_params.items() if k in allowed_keys}

    # ensure essentials exist
    best_params.setdefault("loss_function", "RMSE")
    best_params.setdefault("eval_metric", "RMSE")
    best_params.setdefault("thread_count", int(args.n_jobs))
    best_params.setdefault("random_seed", int(args.seed))
    best_params.setdefault("bootstrap_type", "Bayesian")
    best_params.setdefault("od_type", "Iter")
    best_params.setdefault("od_wait", 40)
    best_params.setdefault("task_type", args.task_type)
    if args.task_type.upper() == "GPU" and args.gpu_devices:
        best_params.setdefault("devices", args.gpu_devices)

    best_json = out_dir / "best_subset.json"
    best_json.write_text(json.dumps({
        "best_rmse": best["rmse"],
        "best_mae": best["mae"],
        "best_r2": best["r2"],
        "best_rrmse_pct": best.get("rrmse_pct", None),
        "groups": best_groups,
        "bands": best_bands,
        "cols": best_cols,
        "cb_params": best_params,
    }, indent=2))
    print("[SUBSET] best saved:", best_json)
    print(
        "[SUBSET] BEST:",
        f"rmse={best['rmse']:.4f}",
        f"mae={best['mae']:.4f}",
        f"r2={best['r2']:.3f}",
        f"rrmse%={best.get('rrmse_pct', float('nan')):.2f}",
        "groups=", best_groups,
    )

    # 3) Train final CatBoost model on ALL BAIXO+LAMEGO samples
    model = CatBoostRegressor(**best_params)
    model.fit(X[:, best_cols], y, verbose=False)

    model_path = out_dir / "catboost_best.joblib"
    joblib.dump({
        "model": model,
        "feature_bands": list(FEATURE_BANDS),
        "feature_groups": FEATURE_GROUPS,
        "selected_groups": best_groups,
        "selected_bands": best_bands,
        "selected_cols": best_cols,
        "cb_params": best_params,
        "seed": int(args.seed),
    }, model_path)
    print("[MODEL] saved:", model_path)

    # Feature importance
    fi = None
    if hasattr(model, "feature_importances_"):
        fi = np.array(model.feature_importances_, dtype=float)
    elif hasattr(model, "get_feature_importance"):
        fi = np.array(model.get_feature_importance(), dtype=float)

    imp_df = None
    if fi is not None:
        imp_df = pd.DataFrame({
            "band": best_bands,
            "importance": fi.astype(float),
        }).sort_values("importance", ascending=False)
        imp_csv = out_dir / "feature_importance_selected.csv"
        imp_df.to_csv(imp_csv, index=False)
        print("[MODEL] importance saved:", imp_csv)

    # 4) Optional TPOT search
    tpot_info = None
    if args.run_tpot:
        tpot_info = run_tpot_on_best_subset(X, y, best_cols, best_bands, out_dir, args)

    # W&B logging
    if wandb_run is not None:
        import wandb
        wandb.log({
            "subset/best_rmse": float(best["rmse"]),
            "subset/best_mae": float(best["mae"]),
            "subset/best_r2": float(best["r2"]),
            "subset/best_rrmse_pct": float(best.get("rrmse_pct", float("nan"))),
            "subset/best_groups": best["groups"],
            "subset/n_selected_bands": int(len(best_bands)),
        })
        wandb.save(str(trials_csv))
        wandb.save(str(best_json))
        wandb.save(str(model_path))
        if imp_df is not None:
            wandb.save(str(imp_csv))
            wandb.log({"subset/feature_importance": wandb.Table(dataframe=imp_df)})

        if tpot_info is not None:
            wandb.log({
                "tpot/val_rmse": float(tpot_info["val_rmse"]),
                "tpot/val_mae": float(tpot_info["val_mae"]),
                "tpot/val_r2": float(tpot_info["val_r2"]),
            })
            wandb.save(tpot_info["model_path"])

    # 5) Per-file evaluation on LAMEGO & BAIXO (hold-out dates)
    # ---------------------------------------------------------
    def run_perfile_eval(tag_label, files, csv_name, png_prefix):
        if not files:
            return None, None, None

        f_list = list(files)
        if args.eval_max and int(args.eval_max) > 0:
            f_list = f_list[: int(args.eval_max)]

        rows = []
        print(f"\n[PER-FILE EVAL] {tag_label} file_count =", len(f_list))
        for fp in f_list:
            tag = Path(fp).stem
            dstr, _ = parse_any_date(tag)
            _, s_rmse, s_mae, s_r2, s_rrmse, n_pix = predict_file_model(
                fp_in=fp,
                model=model,
                cols=best_cols,
                tile=int(args.tile),
                label_name=args.label_name,
            )
            print(
                f"[EVAL] {tag_label} {tag} | "
                f"RMSE={s_rmse:.4f} MAE={s_mae:.4f} R2={s_r2:.3f} "
                f"RRMSE%={s_rrmse:.2f} | n_pix={n_pix}"
            )
            rows.append({
                "file": str(fp),
                "tag": tag,
                "date": dstr,
                "rmse": s_rmse,
                "mae": s_mae,
                "r2": s_r2,
                "rrmse_pct": s_rrmse,
                "n_pix": n_pix,
            })

        per_csv = out_dir / csv_name
        df_rows = pd.DataFrame(rows)
        df_rows.to_csv(per_csv, index=False)
        print("[EVAL] saved:", per_csv)

        plot_outputs = plot_timeseries(
            per_csv,
            out_dir / f"{png_prefix}_timeseries",
            title_prefix=f"{tag_label} (CatBoost)"
        )
        r2_png = plot_outputs["r2"]
        rmse_png = plot_outputs["rmse"]
        print("[EVAL] plots:", r2_png, rmse_png)

        if wandb_run is not None:
            import wandb
            wandb.save(str(per_csv))
            wandb.log({f"eval_{tag_label.lower()}/per_file_metrics_catboost": wandb.Table(dataframe=df_rows)})
            wandb.log({f"eval_{tag_label.lower()}/r2_timeseries_catboost": wandb.Image(str(r2_png))})
            wandb.log({f"eval_{tag_label.lower()}/rmse_timeseries_catboost": wandb.Image(str(rmse_png))})

        return per_csv, r2_png, rmse_png

    # LAMEGO hold-out
    run_perfile_eval(
        tag_label="LAMEGO",
        files=eval_files_lamego,
        csv_name="per_file_metrics_lamego_catboost.csv",
        png_prefix="per_file_lamego_catboost",
    )

    # BAIXO hold-out
    run_perfile_eval(
        tag_label="BAIXO",
        files=eval_files_baixo,
        csv_name="per_file_metrics_baixo_catboost.csv",
        png_prefix="per_file_baixo_catboost",
    )

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()
