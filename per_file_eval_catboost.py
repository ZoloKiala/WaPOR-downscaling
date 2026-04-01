#!/usr/bin/env python3
# ============================================================
# WaPOR 20m Downscaling — Per-file Evaluation ONLY (CatBoost)
#
# ✅ EDIT: Per-file evaluation runs ONLY on the TEST period.
#         Works for BOTH BAIXO and LAMEGO.
#
# You can set ONE global test window (applies to both sites):
#   --test-start YYYY-MM-DD --test-end YYYY-MM-DD
#   OR --test-years 2023,2024
#
# Or set site-specific overrides:
#   --baixo-test-start/--baixo-test-end or --baixo-test-years
#   --lamego-test-start/--lamego-test-end or --lamego-test-years
#
# Per-file metrics:
#   (A) Pred20m -> WaPOR300m (ETa300m): RMSE / MAE / Bias / RRMSE%   (after ~300m block-mean)
#   (B) Native20m (b1) -> WaPOR300m    : RMSE / MAE / Bias / RRMSE%  (after ~300m block-mean)
#   (C) Pred20m vs Native20m (b1)      : RMSE / MAE / R² / RRMSE% (20m)
#
# Example (global test window for both sites):
#   python per_file_eval_catboost.py \
#     --model-bundle outputs_catboost_datesplit/catboost_best.joblib \
#     --lamego-eval-dir wapor_downscale_data/LAMEGO_STACK_S2_MATCH_L3_20M_FULL_1 \
#     --baixo-eval-dir  wapor_downscale_data/BAIXO_STACK_S2_MATCH_L3_20M_FULL_1 \
#     --out-dir outputs_perfile_eval \
#     --tile 2048 \
#     --test-start 2023-01-01 --test-end 2024-12-31 \
#     --save-preds
#
# Example (site-specific windows):
#   ... --baixo-test-years 2023,2024 --lamego-test-years 2021,2022
# ============================================================

import os
import re
import glob
import math
import time
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import rasterio
from rasterio.windows import Window
from rasterio.env import Env as RasterioEnv

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib


# ---------------------------
# Constants
# ---------------------------
FALLBACK_NODATA = -9999.0
NODATA_OUT = -9999.0

REQUIRED_BANDS = ("B4", "B8", "B11", "ETa300m", "DEM", "Slope")


# ============================================================
# Utils
# ============================================================
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


def parse_date_from_any_metadata(fp: str):
    """
    Robust date discovery for test-period filtering:
      1) filename
      2) dataset tags: DATE/date/time_start/system:time_start/start_date/etc

    Returns (iso_date_str, doy) or (None, None)
    """
    stem = Path(fp).stem
    dstr, doy = parse_any_date(stem)
    if dstr:
        return dstr, doy

    try:
        with rasterio.open(fp) as ds:
            tags = ds.tags() or {}
            cand_keys = [
                "DATE", "date", "Date",
                "time_start", "TIME_START",
                "system:time_start", "SYSTEM:TIME_START",
                "start_date", "START_DATE",
            ]
            for k in cand_keys:
                if k in tags and tags[k]:
                    v = str(tags[k]).strip()

                    # millis since epoch (GEE-like)
                    if v.isdigit():
                        try:
                            ms = int(v)
                            dte = dt.datetime.utcfromtimestamp(ms / 1000.0).date()
                            doy = (dte - dt.date(dte.year, 1, 1)).days + 1
                            return dte.isoformat(), doy
                        except Exception:
                            pass

                    dd, doy2 = parse_any_date(v)
                    if dd:
                        return dd, doy2
    except Exception:
        pass

    return None, None


def parse_years_list(s: str):
    if not s:
        return None
    yrs = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        yrs.append(int(part))
    return yrs if yrs else None


def in_test_window(date_iso: str, test_start: str = None, test_end: str = None, test_years=None) -> bool:
    """
    Decide whether date_iso belongs to test period.
    - If test_years provided: keep those years
    - Else use inclusive [test_start, test_end]
    """
    if not date_iso:
        return False
    try:
        d = dt.date.fromisoformat(date_iso)
    except Exception:
        return False

    if test_years:
        return d.year in set(test_years)

    if test_start:
        s = dt.date.fromisoformat(test_start)
        if d < s:
            return False
    if test_end:
        e = dt.date.fromisoformat(test_end)
        if d > e:
            return False
    return True


def sincos_doy(doy: int):
    ang = 2.0 * math.pi * (doy / 365.0)
    return float(math.sin(ang)), float(math.cos(ang))


def list_all_tifs(root):
    patterns = ["**/*.tif", "**/*.tiff", "**/*.TIF", "**/*.TIFF"]
    files = []
    for pat in patterns:
        files += glob.glob(os.path.join(root, pat), recursive=True)
    return sorted(set(files))


def is_candidate_stack(fp: str) -> bool:
    p = fp.replace("\\", "/").lower()
    if "/preds" in p or "/tb" in p or "/cache_tfrecord" in p or "/cache_npz" in p:
        return False
    return fp.lower().endswith((".tif", ".tiff"))


def get_effective_nodata(ds):
    nod = ds.nodata
    if nod is None or (not np.isfinite(nod)):
        return FALLBACK_NODATA
    return float(nod)


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
        tag = ds.tags(i + 1) or {}
        for key in ("BANDNAME", "band_name", "NAME", "BandName", "DESCRIPTION", "description", "name"):
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
        "B4":  ["b4", "B04", "red", "RED", "band4"],
        "B8":  ["b8", "B08", "nir", "NIR", "band8"],
        "B11": ["b11", "swir", "SWIR", "swir1", "SWIR1", "band11"],

        "ETa300m": ["eta300m", "ETa_300m", "ETa300", "ETa300m", "ET300m", "AETI300", "AETI_300m", "L1-AETI-D", "L1_AETI_D"],
        "DEM":   ["dem", "elevation", "ELEVATION", "elev"],
        "Slope": ["slope", "SLOPE", "terrain_slope", "SLOPE_DEG"],

        "NDVI": ["ndvi", "NDVI"],
        "NDMI": ["ndmi", "NDMI"],
        "FVC":  ["fvc", "FVC"],

        "Aspect":     ["aspect", "ASPECT", "Aspect_deg", "aspect_deg"],
        "Aspect_sin": ["aspect_sin", "ASPECT_SIN"],
        "Aspect_cos": ["aspect_cos", "ASPECT_COS"],

        "RAIN_10d":     ["rain_10d", "RAIN_10D", "precip_10d", "P10D", "RAIN10D"],
        "RAIN_10d_lag": ["rain_10d_lag", "RAIN_10D_LAG", "precip_10d_lag", "P10D_LAG", "RAIN10D_LAG"],

        "b1": ["B1", "b1", "label", "LABEL", "target", "y", "ETa20", "ETa20m", "AETI20"],
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


def file_has_required_bands(fp: str, label_name="b1") -> bool:
    try:
        with rasterio.open(fp) as ds:
            bm = band_map(ds)
            if not bm:
                return False
            ok_feats = all(k in bm for k in REQUIRED_BANDS)
            lb = get_label_band_index(bm, label_name)
            return ok_feats and (lb is not None)
    except Exception:
        return False


# ============================================================
# Feature engineering (must match training)
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

    b8  = _sanitize_band(ds.read(bm["B8"] + 1,      window=win), nod)
    b11 = _sanitize_band(ds.read(bm["B11"] + 1,     window=win), nod)
    b4  = _sanitize_band(ds.read(bm["B4"] + 1,      window=win), nod)
    et  = _sanitize_band(ds.read(bm["ETa300m"] + 1, window=win), nod)
    dem = _sanitize_band(ds.read(bm["DEM"] + 1,     window=win), nod)
    slp = _sanitize_band(ds.read(bm["Slope"] + 1,   window=win), nod)

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

    # Aspect
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

    # Rain
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
    y = ds.read(lb0 + 1, window=win).astype(np.float32)
    y[y == nod] = np.nan
    y[~np.isfinite(y)] = np.nan
    m = np.isfinite(y)
    y = np.nan_to_num(y, nan=0.0).astype(np.float32)
    return y, m


def read_et300_window(ds, bm, win: Window):
    nod = get_effective_nodata(ds)
    et = ds.read(bm["ETa300m"] + 1, window=win).astype(np.float32)
    et[et == nod] = np.nan
    et[~np.isfinite(et)] = np.nan
    return et


# ============================================================
# Metrics helpers
# ============================================================
def rmse_only(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae_only(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def bias(y_true, y_pred):
    # ✅ FIX (your original had y_pred = y_true)
    y_true = np.asarray(y_true, np.float32)
    y_pred = np.asarray(y_pred, np.float32)
    return float(np.nanmean(y_pred - y_true))


def rrmse_percent(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    denom = float(np.nanmean(y_true)) if y_true.size > 0 else float("nan")
    if (not np.isfinite(denom)) or denom == 0.0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)) / denom * 100.0)


# ============================================================
# GeoTIFF writer
# ============================================================
def write_pred_tif(ds_in, pred_arr, out_path):
    out_path = str(out_path)
    profile = ds_in.profile.copy()
    profile.update(
        count=1,
        dtype="float32",
        nodata=NODATA_OUT,
        compress="deflate",
        predictor=2,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(pred_arr.astype(np.float32), 1)
        dst.set_band_description(1, "ETa20m_pred")


# ============================================================
# Block aggregation (mean) for ~300m evaluation
# ============================================================
def infer_block_factor(ds, target_m=300.0):
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


# ============================================================
# Tile-wise prediction + per-file metrics
# ============================================================
def predict_file_and_metrics(fp_in, model, cols, tile=2048, label_name="b1", save_pred_tif_path=None):
    # robust date (filename or metadata)
    dstr, doy = parse_date_from_any_metadata(fp_in)
    doy = 1 if doy is None else int(doy)

    gdal_opts = {"GDAL_CACHEMAX": 2048, "GDAL_NUM_THREADS": "ALL_CPUS"}

    with RasterioEnv(**gdal_opts):
        with rasterio.open(fp_in) as ds:
            bm = band_map(ds)
            H, W = ds.height, ds.width

            pred_full = np.full((H, W), NODATA_OUT, np.float32)
            y_full    = np.full((H, W), np.nan, np.float32)  # native 20m label (b1)
            et_full   = np.full((H, W), np.nan, np.float32)  # ETa300m values on 20m grid
            m_full    = np.zeros((H, W), np.uint8)           # mask for 20m pred-vs-native

            for y0 in range(0, H, tile):
                for x0 in range(0, W, tile):
                    h0 = min(tile, H - y0)
                    w0 = min(tile, W - x0)
                    win = Window(x0, y0, w0, h0)

                    X_chw, valid_feat = read_features_window(ds, bm, win, doy)
                    y_win, m_lab = read_label_window(ds, bm, win, label_name=label_name)
                    et_win = read_et300_window(ds, bm, win)

                    X_hwC = np.transpose(X_chw, (1, 2, 0))
                    X_flat = X_hwC.reshape(-1, X_hwC.shape[-1]).astype(np.float32)

                    feat_ok = valid_feat.reshape(-1)
                    lab_ok  = m_lab.reshape(-1)
                    both_ok = feat_ok & lab_ok

                    out = np.full((h0 * w0,), NODATA_OUT, np.float32)
                    if np.any(feat_ok):
                        pred_valid = model.predict(X_flat[feat_ok][:, cols]).astype(np.float32)
                        out[np.where(feat_ok)[0]] = pred_valid

                    pred_tile = out.reshape(h0, w0)
                    pred_full[y0:y0+h0, x0:x0+w0] = pred_tile

                    y_full[y0:y0+h0, x0:x0+w0]  = y_win
                    et_full[y0:y0+h0, x0:x0+w0] = et_win
                    m_full[y0:y0+h0, x0:x0+w0]  = both_ok.reshape(h0, w0).astype(np.uint8)

            if save_pred_tif_path:
                write_pred_tif(ds, pred_full, save_pred_tif_path)

            # 20m: Pred vs Native
            m20 = m_full.astype(bool)
            n_pix_20 = int(m20.sum())
            if n_pix_20 >= 2:
                y20 = y_full[m20].astype(np.float32)
                p20 = pred_full[m20].astype(np.float32)
                pvn_rmse  = rmse_only(y20, p20)
                pvn_mae   = mae_only(y20, p20)
                pvn_r2    = float(r2_score(y20, p20))
                pvn_rrmse = rrmse_percent(y20, p20)
            else:
                pvn_rmse = pvn_mae = pvn_r2 = pvn_rrmse = float("nan")

            # 300m comparisons
            bf = infer_block_factor(ds, target_m=300.0)
            if bf is None:
                pvE_rmse = pvE_mae = pvE_bias = pvE_rrmse = float("nan")
                nvE_rmse = nvE_mae = nvE_bias = nvE_rrmse = float("nan")
                n_pix_300_pred = 0
                n_pix_300_nat  = 0
                fx = fy = None
            else:
                fx, fy = bf

                pred_nan = pred_full.astype(np.float32)
                pred_nan[pred_nan == NODATA_OUT] = np.nan

                nat_nan = y_full.astype(np.float32)
                et_nan  = et_full.astype(np.float32)

                # require finite ET for comparisons
                pred_nan[~np.isfinite(et_nan)] = np.nan
                nat_nan[~np.isfinite(et_nan)]  = np.nan

                pred_300 = block_mean_2d(pred_nan, fx=fx, fy=fy)
                nat_300  = block_mean_2d(nat_nan,  fx=fx, fy=fy)
                et_300   = block_mean_2d(et_nan,   fx=fx, fy=fy)

                if pred_300 is None or nat_300 is None or et_300 is None:
                    pvE_rmse = pvE_mae = pvE_bias = pvE_rrmse = float("nan")
                    nvE_rmse = nvE_mae = nvE_bias = nvE_rrmse = float("nan")
                    n_pix_300_pred = 0
                    n_pix_300_nat  = 0
                else:
                    m_pred = np.isfinite(pred_300) & np.isfinite(et_300)
                    m_nat  = np.isfinite(nat_300)  & np.isfinite(et_300)

                    n_pix_300_pred = int(m_pred.sum())
                    n_pix_300_nat  = int(m_nat.sum())

                    if n_pix_300_pred >= 2:
                        y300 = et_300[m_pred].astype(np.float32)
                        p300 = pred_300[m_pred].astype(np.float32)
                        pvE_rmse  = rmse_only(y300, p300)
                        pvE_mae   = mae_only(y300, p300)
                        pvE_bias  = bias(y300, p300)
                        pvE_rrmse = rrmse_percent(y300, p300)
                    else:
                        pvE_rmse = pvE_mae = pvE_bias = pvE_rrmse = float("nan")

                    if n_pix_300_nat >= 2:
                        y300 = et_300[m_nat].astype(np.float32)
                        n300 = nat_300[m_nat].astype(np.float32)
                        nvE_rmse  = rmse_only(y300, n300)
                        nvE_mae   = mae_only(y300, n300)
                        nvE_bias  = bias(y300, n300)
                        nvE_rrmse = rrmse_percent(y300, n300)
                    else:
                        nvE_rmse = nvE_mae = nvE_bias = nvE_rrmse = float("nan")

            return {
                "date": dstr or "",
                "pvn_rmse": pvn_rmse,
                "pvn_mae": pvn_mae,
                "pvn_r2": pvn_r2,
                "pvn_rrmse_pct": pvn_rrmse,
                "n_pix_20": n_pix_20,

                "pvE_rmse": pvE_rmse,
                "pvE_mae": pvE_mae,
                "pvE_bias": pvE_bias,
                "pvE_rrmse_pct": pvE_rrmse,
                "n_pix_300_pred": int(n_pix_300_pred),

                "nvE_rmse": nvE_rmse,
                "nvE_mae": nvE_mae,
                "nvE_bias": nvE_bias,
                "nvE_rrmse_pct": nvE_rrmse,
                "n_pix_300_nat": int(n_pix_300_nat),

                "agg_fx": int(fx) if fx is not None else "",
                "agg_fy": int(fy) if fy is not None else "",
            }


# ============================================================
# Plotting
# ============================================================
def make_plots(csv_path: Path, out_dir: Path, tag: str):
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.read_csv(csv_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
        x = df["date"]
        xlab = "Date"
    else:
        x = np.arange(len(df))
        xlab = "File"

    def _style(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    outputs = {}

    comparisons_300 = [
        ("pvE_rmse",      "nvE_rmse",      "RMSE (20→300)",   f"{tag} 20m→300m RMSE (Pred vs Native)"),
        ("pvE_mae",       "nvE_mae",       "MAE (20→300)",    f"{tag} 20m→300m MAE (Pred vs Native)"),
        ("pvE_bias",      "nvE_bias",      "Bias (20→300)",   f"{tag} 20m→300m Bias (Pred vs Native)"),
        ("pvE_rrmse_pct", "nvE_rrmse_pct", "RRMSE% (20→300)", f"{tag} 20m→300m Relative RMSE % (Pred vs Native)"),
    ]
    for c_pred, c_nat, ylab, title in comparisons_300:
        if c_pred not in df.columns or c_nat not in df.columns:
            continue
        p = out_dir / f"{tag.lower()}_{c_pred}_vs_{c_nat}.png"
        fig, ax = plt.subplots()
        ax.plot(x, df[c_pred].values, label="Pred20 → WaPOR300")
        ax.plot(x, df[c_nat].values,  label="Native20 → WaPOR300")
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        _style(ax)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        outputs[p.name] = str(p)

    if "pvn_rmse" in df.columns and "pvn_mae" in df.columns:
        p = out_dir / f"{tag.lower()}_pred20_vs_native20_rmse_mae.png"
        fig, ax = plt.subplots()
        ax.plot(x, df["pvn_rmse"].values, label="RMSE (Pred20 vs Native20)")
        ax.plot(x, df["pvn_mae"].values,  label="MAE (Pred20 vs Native20)")
        ax.set_title(f"{tag} Pred20 vs Native20 (20m): RMSE & MAE")
        ax.set_xlabel(xlab)
        ax.set_ylabel("Error")
        _style(ax)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        outputs[p.name] = str(p)

    if "pvn_r2" in df.columns:
        p = out_dir / f"{tag.lower()}_pred20_vs_native20_r2.png"
        fig, ax = plt.subplots()
        ax.plot(x, df["pvn_r2"].values, label="R² (Pred20 vs Native20)")
        ax.set_title(f"{tag} Pred20 vs Native20 (20m): R²")
        ax.set_xlabel(xlab)
        ax.set_ylabel("R²")
        _style(ax)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        outputs[p.name] = str(p)

    if "pvn_rrmse_pct" in df.columns:
        p = out_dir / f"{tag.lower()}_pred20_vs_native20_rrmse_pct.png"
        fig, ax = plt.subplots()
        ax.plot(x, df["pvn_rrmse_pct"].values, label="RRMSE% (Pred20 vs Native20)")
        ax.set_title(f"{tag} Pred20 vs Native20 (20m): Relative RMSE %")
        ax.set_xlabel(xlab)
        ax.set_ylabel("RRMSE (%)")
        _style(ax)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        outputs[p.name] = str(p)

    return outputs


# ============================================================
# Run evaluation for one dataset (tag + dir)
# ============================================================
def run_dataset(tag: str, eval_dir: str, out_root: Path, model, selected_cols, args,
                test_start: str, test_end: str, test_years):
    if not eval_dir:
        return None

    eval_dir = str(eval_dir)
    if not os.path.exists(eval_dir):
        print(f"[SKIP] {tag}: eval_dir not found -> {eval_dir}")
        return None

    out_dir = out_root / tag.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    files_all = [fp for fp in list_all_tifs(eval_dir)
                 if is_candidate_stack(fp) and file_has_required_bands(fp, label_name=args.label_name)]

    if not files_all:
        print(f"[SKIP] {tag}: no valid stacks found in -> {eval_dir}")
        return None

    # ✅ Filter to TEST only (per-site window)
    files = []
    skipped_no_date = 0
    skipped_not_test = 0
    for fp in files_all:
        dstr, _ = parse_date_from_any_metadata(fp)
        if not dstr:
            skipped_no_date += 1
            continue
        if in_test_window(dstr, test_start, test_end, test_years):
            files.append(fp)
        else:
            skipped_not_test += 1

    files = sorted(files)

    print(f"\n[{tag}] EVAL_DIR={eval_dir}")
    print(f"[{tag}] candidates={len(files_all)} | kept_test={len(files)} | "
          f"skipped_no_date={skipped_no_date} | skipped_not_test={skipped_not_test}")
    print(f"[{tag}] TEST window: start={test_start} end={test_end} years={test_years}")

    if not files:
        print(f"[SKIP] {tag}: no TEST-period stacks found in -> {eval_dir}")
        return None

    if args.eval_max and int(args.eval_max) > 0:
        files = files[: int(args.eval_max)]

    preds_dir = out_dir / f"preds_{tag.lower()}"
    if args.save_preds:
        preds_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()

    for i, fp in enumerate(files, 1):
        stem = Path(fp).stem

        pred_tif = None
        if args.save_preds:
            pred_tif = preds_dir / f"{stem}_pred.tif"

        mets = predict_file_and_metrics(
            fp_in=fp,
            model=model,
            cols=selected_cols,
            tile=int(args.tile),
            label_name=args.label_name,
            save_pred_tif_path=pred_tif,
        )

        print(
            f"[{tag}] {i:03d}/{len(files)} {stem} ({mets['date']}) | "
            f"Pred20→WaPOR300: RMSE={mets['pvE_rmse']:.3f} MAE={mets['pvE_mae']:.3f} Bias={mets['pvE_bias']:.3f} RRMSE%={mets['pvE_rrmse_pct']:.2f} (n={mets['n_pix_300_pred']}) | "
            f"Native20→WaPOR300: RMSE={mets['nvE_rmse']:.3f} MAE={mets['nvE_mae']:.3f} Bias={mets['nvE_bias']:.3f} RRMSE%={mets['nvE_rrmse_pct']:.2f} (n={mets['n_pix_300_nat']}) | "
            f"Pred20↔Native20: R2={mets['pvn_r2']:.3f} RMSE={mets['pvn_rmse']:.3f} MAE={mets['pvn_mae']:.3f} RRMSE%={mets['pvn_rrmse_pct']:.2f} (n={mets['n_pix_20']}) | "
            f"agg=({mets['agg_fx']}x{mets['agg_fy']})"
        )

        rows.append({
            "file": str(fp),
            "tag": stem,
            "date": mets["date"],

            "pvn_rmse": mets["pvn_rmse"],
            "pvn_mae": mets["pvn_mae"],
            "pvn_r2": mets["pvn_r2"],
            "pvn_rrmse_pct": mets["pvn_rrmse_pct"],
            "n_pix_20": mets["n_pix_20"],

            "pvE_rmse": mets["pvE_rmse"],
            "pvE_mae": mets["pvE_mae"],
            "pvE_bias": mets["pvE_bias"],
            "pvE_rrmse_pct": mets["pvE_rrmse_pct"],
            "n_pix_300_pred": mets["n_pix_300_pred"],

            "nvE_rmse": mets["nvE_rmse"],
            "nvE_mae": mets["nvE_mae"],
            "nvE_bias": mets["nvE_bias"],
            "nvE_rrmse_pct": mets["nvE_rrmse_pct"],
            "n_pix_300_nat": mets["n_pix_300_nat"],

            "agg_fx": mets["agg_fx"],
            "agg_fy": mets["agg_fy"],

            "pred_tif": str(pred_tif) if pred_tif else "",
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"per_file_metrics_{tag.lower()}_TESTONLY.csv"
    df.to_csv(csv_path, index=False)
    print(f"[{tag}] [OUT] metrics CSV: {csv_path}")

    plots = make_plots(csv_path, out_dir, tag.upper())
    print(f"[{tag}] [OUT] plots:")
    for k, v in plots.items():
        print("   ", k, "->", v)

    dt_min = (time.time() - t0) / 60.0
    print(f"[{tag}] [DONE] time: {dt_min:.2f} min")

    return str(csv_path)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model-bundle", type=str, required=True,
                    help="Path to catboost_best.joblib produced by training script.")
    ap.add_argument("--out-dir", type=str, default="outputs_perfile_eval",
                    help="Output root directory; subfolders LAMEGO/ and BAIXO/ will be created.")

    ap.add_argument("--lamego-eval-dir", type=str, default="",
                    help="Folder with LAMEGO stack GeoTIFFs to evaluate (optional).")
    ap.add_argument("--baixo-eval-dir", type=str, default="",
                    help="Folder with BAIXO stack GeoTIFFs to evaluate (optional).")

    ap.add_argument("--label-name", type=str, default="b1")
    ap.add_argument("--tile", type=int, default=2048)
    ap.add_argument("--eval-max", type=int, default=0)
    ap.add_argument("--save-preds", action="store_true")

    # Global test window (applies to both, unless overridden)
    ap.add_argument("--test-start", type=str, default="",
                    help="Global test start date (YYYY-MM-DD).")
    ap.add_argument("--test-end", type=str, default="",
                    help="Global test end date (YYYY-MM-DD).")
    ap.add_argument("--test-years", type=str, default="",
                    help="Global test years, comma-separated (e.g., 2023,2024). Overrides --test-start/--test-end.")

    # Site overrides
    ap.add_argument("--baixo-test-start", type=str, default="", help="BAIXO test start (YYYY-MM-DD).")
    ap.add_argument("--baixo-test-end", type=str, default="", help="BAIXO test end (YYYY-MM-DD).")
    ap.add_argument("--baixo-test-years", type=str, default="", help="BAIXO test years (e.g., 2023,2024).")

    ap.add_argument("--lamego-test-start", type=str, default="", help="LAMEGO test start (YYYY-MM-DD).")
    ap.add_argument("--lamego-test-end", type=str, default="", help="LAMEGO test end (YYYY-MM-DD).")
    ap.add_argument("--lamego-test-years", type=str, default="", help="LAMEGO test years (e.g., 2023,2024).")

    args = ap.parse_args()

    # Normalize global test
    test_start_g = args.test_start.strip() or None
    test_end_g = args.test_end.strip() or None
    test_years_g = parse_years_list(args.test_years)

    # Normalize BAIXO test (override if provided)
    test_start_b = args.baixo_test_start.strip() or test_start_g
    test_end_b = args.baixo_test_end.strip() or test_end_g
    test_years_b = parse_years_list(args.baixo_test_years) or test_years_g

    # Normalize LAMEGO test (override if provided)
    test_start_l = args.lamego_test_start.strip() or test_start_g
    test_end_l = args.lamego_test_end.strip() or test_end_g
    test_years_l = parse_years_list(args.lamego_test_years) or test_years_g

    # Require *some* test definition for each site that will run
    def _has_test(ts, te, ty):
        return (ty is not None) or (ts is not None) or (te is not None)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load model bundle
    bundle = joblib.load(args.model_bundle)
    model = bundle["model"]
    selected_cols = bundle.get("selected_cols", None)
    selected_bands = bundle.get("selected_bands", None)

    if selected_cols is None:
        raise RuntimeError("Model bundle missing 'selected_cols'. Re-save bundle from training script.")
    selected_cols = list(map(int, selected_cols))

    print("[MODEL] loaded:", args.model_bundle)
    print("[MODEL] selected bands:", selected_bands)
    print("[MODEL] selected cols:", selected_cols)
    print("[OUT] root:", out_root)

    ran_any = False

    if args.baixo_eval_dir:
        if not _has_test(test_start_b, test_end_b, test_years_b):
            raise SystemExit("BAIXO eval requested but no test window provided (use global --test-* or --baixo-test-*).")
        run_dataset("BAIXO", args.baixo_eval_dir, out_root, model, selected_cols, args,
                    test_start=test_start_b, test_end=test_end_b, test_years=test_years_b)
        ran_any = True

    if args.lamego_eval_dir:
        if not _has_test(test_start_l, test_end_l, test_years_l):
            raise SystemExit("LAMEGO eval requested but no test window provided (use global --test-* or --lamego-test-*).")
        run_dataset("LAMEGO", args.lamego_eval_dir, out_root, model, selected_cols, args,
                    test_start=test_start_l, test_end=test_end_l, test_years=test_years_l)
        ran_any = True

    if not ran_any:
        raise SystemExit("No eval dirs provided. Use --lamego-eval-dir and/or --baixo-eval-dir.")


if __name__ == "__main__":
    main()
