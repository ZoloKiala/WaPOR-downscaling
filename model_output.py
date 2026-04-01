import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------
# Paths (edit if needed)
# -----------------------------
EVAL_DIR = Path("outputs_perfile_eval")               # where per_file_eval_catboost.py writes
MODEL_DIR = Path("outputs_catboost_datesplit")        # where training script writes subset_trials.csv etc.

# 300m per-file CSVs (from per_file_eval_catboost.py)
BAIXO_300_CSV  = EVAL_DIR / "BAIXO"  / "per_file_metrics_baixo.csv"
LAMEGO_300_CSV = EVAL_DIR / "LAMEGO" / "per_file_metrics_lamego.csv"

# Subset search outputs (from training script)
BEST_JSON  = MODEL_DIR / "best_subset.json"
TRIALS_CSV = MODEL_DIR / "subset_trials.csv"
FI_CSV     = MODEL_DIR / "feature_importance_selected.csv"

# -----------------------------
# Plot styling helper
# -----------------------------
def _despine(ax):
    """Remove top/right axes spines (journal-style)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# -----------------------------
# Stats helpers
# -----------------------------
def bootstrap_mean_ci(x, n_boot=2000, seed=7):
    """Mean + 95% bootstrap CI of the mean."""
    x = np.asarray(pd.Series(x).dropna().values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = rng.choice(x, size=len(x), replace=True)
        boots.append(np.mean(s))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.mean(x)), float(lo), float(hi), int(len(x))

def paired_tests(a, b):
    """Paired t-test and Wilcoxon signed-rank (two-sided)."""
    a = pd.Series(a).astype(float)
    b = pd.Series(b).astype(float)
    df = pd.concat([a, b], axis=1).dropna()
    a2, b2 = df.iloc[:, 0].values, df.iloc[:, 1].values
    if len(a2) < 5:
        return {"n": int(len(a2)), "p_t": np.nan, "p_w": np.nan}
    t_stat, p_t = stats.ttest_rel(a2, b2, nan_policy="omit")
    try:
        w_stat, p_w = stats.wilcoxon(a2, b2, zero_method="wilcox", alternative="two-sided")
    except Exception:
        p_w = np.nan
    return {"n": int(len(a2)), "p_t": float(p_t), "p_w": float(p_w)}

def summarize_300m(csv_path, site_name):
    df = pd.read_csv(csv_path)

    # per_file_eval_catboost.py columns for 300m:
    pred = df["pvE_rmse"].dropna()   # Pred20 -> WaPOR300 RMSE
    nat  = df["nvE_rmse"].dropna()   # Native20 -> WaPOR300 RMSE

    mean_p, lo_p, hi_p, n_p = bootstrap_mean_ci(pred)
    mean_n, lo_n, hi_n, n_n = bootstrap_mean_ci(nat)

    tests = paired_tests(pred, nat)

    diff = mean_n - mean_p
    impr = 100.0 * diff / mean_n if mean_n != 0 else np.nan

    return {
        "site": site_name,
        "pred_mean": mean_p, "pred_lo": lo_p, "pred_hi": hi_p,
        "nat_mean": mean_n,  "nat_lo": lo_n, "nat_hi": hi_n,
        "n_pairs": tests["n"],
        "p_t": tests["p_t"],
        "p_w": tests["p_w"],
        "rmse_diff": diff,
        "improv_pct": impr,
    }

# -----------------------------
# 1) 300 m aggregation stats (Baixo + Lamego)
# -----------------------------
rows = []
rows.append(summarize_300m(BAIXO_300_CSV, "Baixo"))
rows.append(summarize_300m(LAMEGO_300_CSV, "Lamego"))
summary = pd.DataFrame(rows)
print("\n=== 300 m RMSE bootstrap + paired tests ===")
print(summary.to_string(index=False))

out_csv = Path("agg300m_rmse_bootstrap_and_tests.csv")
summary.to_csv(out_csv, index=False)
print("\nSaved:", out_csv.resolve())

# -----------------------------
# 2) Bar chart: feature-group selection frequency
# -----------------------------
trials = pd.read_csv(TRIALS_CSV)

def parse_groups(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [g.strip() for g in str(s).split(",") if g.strip()]

groups_all = sorted({g for gs in trials["groups"].apply(parse_groups) for g in gs})
freq = {g: 0 for g in groups_all}
for gs in trials["groups"].apply(parse_groups):
    for g in set(gs):
        freq[g] += 1

freq_df = pd.DataFrame({"group": list(freq.keys()), "n_selected": list(freq.values())})
freq_df["pct_selected"] = 100.0 * freq_df["n_selected"] / len(trials)
freq_df = freq_df.sort_values("pct_selected", ascending=False)

fig, ax = plt.subplots()
ax.bar(freq_df["group"], freq_df["pct_selected"])
ax.set_ylabel("Selection frequency (%)")
ax.set_xlabel("Feature group")
ax.set_title("Feature-group selection frequency across subset-search trials")
ax.tick_params(axis="x", rotation=30)
for lbl in ax.get_xticklabels():
    lbl.set_ha("right")
_despine(ax)
fig.tight_layout()
fig.savefig("feature_group_selection_frequency.png", dpi=200)
plt.close(fig)
print("Saved: feature_group_selection_frequency.png")

# -----------------------------
# 3) Bar chart: top feature importances (selected model)
# -----------------------------
if FI_CSV.exists():
    fi = pd.read_csv(FI_CSV).sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots()
    ax.bar(fi["band"], fi["importance"])
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature (band)")
    ax.set_title("Top 15 feature importances (selected CatBoost model)")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    _despine(ax)
    fig.tight_layout()
    fig.savefig("top15_feature_importance.png", dpi=200)
    plt.close(fig)
    print("Saved: top15_feature_importance.png")
else:
    print("[WARN] feature_importance_selected.csv not found. Skipping importance plot.")

# -----------------------------
# 4) Print best subset headline (for reporting)
# -----------------------------
best = json.loads(BEST_JSON.read_text())
print("\n=== Best subset (validation) ===")
print(f"RMSE   : {best['best_rmse']:.3f} mm/dec")
print(f"MAE    : {best['best_mae']:.3f} mm/dec")
print(f"R²     : {best['best_r2']:.3f}")
if best.get("best_rrmse_pct") is not None:
    print(f"RRMSE% : {best['best_rrmse_pct']:.2f}%")
print("Groups :", best["groups"])
print("Bands  :", best["bands"])
