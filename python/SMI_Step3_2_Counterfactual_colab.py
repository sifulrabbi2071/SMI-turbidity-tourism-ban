"""
Saint Martin's Island — Phase 3, Step 3.2
Tide covariates + machine-learning counterfactual models

What this script does
  1. Tide at every Sentinel-2 overpass (pyTMD, GOT4.10 model):
     height at overpass, 24-h tidal range (spring/neap), rising/falling
  2. Builds the modelling table: one row per usable date for each
     treatment zone; outcome = log(turbidity median)
  3. Trains 4 models on PRE-restriction data only (2017–Oct 2024,
     COVID closure excluded):
        M0 ratio baseline  (control turbidity + constant offset)
        M1 Ridge regression
        M2 Random Forest
        M3 XGBoost
     and evaluates them with leave-one-year-out cross-validation
  4. Predicts the counterfactual ("what turbidity would have been
     without the restriction") for the restriction period
  5. Effect = observed − counterfactual (log scale -> % change),
     95% bootstrap CI, placebo test against pre-restriction years,
     minimum detectable effect (MDE)

Inputs : SMI_project/outputs/SMI_analysis_dataset_v1.csv  (Step 3.1)
Outputs: SMI_project/outputs/step3_2/...
Run in Google Colab after mounting Drive (see instructions).
Version 1.0 (September 2026)
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import pyTMD

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 0. SETTINGS
# ------------------------------------------------------------------
BASE = os.environ.get("SMI_BASE", "/content/drive/MyDrive/SMI_project/outputs")
IN_FILE = os.path.join(BASE, "SMI_analysis_dataset_v1.csv")
OUT = os.path.join(BASE, "step3_2")
os.makedirs(OUT, exist_ok=True)

ISLAND = (92.32, 20.62)          # lon, lat for tide prediction
TIDE_MODEL = "GOT4.10_nc"
RNG = np.random.default_rng(42)  # fixed seed -> reproducible
N_BOOT = 2000

ZONES = ["T_north", "T_east", "T_west", "T_south",
         "T_north_deep", "T_east_deep", "T_west_deep", "T_south_deep"]

FEATURES = ["log_ctrl", "ctrl_EW", "ws10_24h", "ws10_72h", "u10_24h", "v10_24h",
            "swh_24h", "sst_C", "era5_rain_mm_catch_72h", "chirps_catch_7d",
            "chirps_catch_30d", "tide_now", "tide_range", "tide_trend",
            "doy_sin", "doy_cos", "sun_zenith"]

# ------------------------------------------------------------------
# 1. LOAD
# ------------------------------------------------------------------
d = pd.read_csv(IN_FILE, parse_dates=["date"])
d["overpass"] = pd.to_datetime(d["date"].dt.strftime("%Y-%m-%d") + " " + d["time_utc"])

# ------------------------------------------------------------------
# 2. TIDES (cached after the first run)
# ------------------------------------------------------------------
tide_file = os.path.join(OUT, "tides_GOT410.csv")
if os.path.exists(tide_file):
    tides = pd.read_csv(tide_file, parse_dates=["overpass"])
else:
    pyTMD.datasets.fetch_gsfc_got("GOT4.10")          # downloads once (~ minutes)
    ov = np.sort(d["overpass"].unique())
    offsets = np.arange(-12, 13)                       # -12 h ... +12 h, hourly
    t_all = (ov[:, None] + offsets[None, :].astype("timedelta64[h]")).ravel()
    h = pyTMD.compute.tide_elevations(
        ISLAND[0], ISLAND[1], t_all, model=TIDE_MODEL, crs="4326",
        standard="datetime", method="linear", type="time series",
        extrapolate=True, cutoff=50.0)                 # nearest ocean cell within 50 km
    h = np.asarray(h.values, dtype=float).reshape(len(ov), len(offsets))
    tides = pd.DataFrame({
        "overpass": ov,
        "tide_now": h[:, 12],                          # m, at overpass
        "tide_range": h.max(axis=1) - h.min(axis=1),   # m, 24-h range (spring/neap)
        "tide_trend": h[:, 13] - h[:, 11]})            # m/2h, >0 rising
    tides.to_csv(tide_file, index=False)
print("Tides: n =", len(tides), "| NaN:", int(tides.tide_now.isna().sum()),
      "| range of tide_now (m):", round(tides.tide_now.min(), 2), "to", round(tides.tide_now.max(), 2))

d = d.merge(tides, on="overpass", how="left")

# ------------------------------------------------------------------
# 3. MODELLING TABLE
# ------------------------------------------------------------------
ok = d[d["ok"]].copy()
ctrl = ok[ok.zone.isin(["C_east", "C_west"])].pivot_table(
    index="date", columns="zone", values="turbidity_median").dropna()
ctrl_feat = pd.DataFrame({"log_ctrl": np.log(ctrl).mean(axis=1),
                          "ctrl_EW": np.log(ctrl["C_east"] / ctrl["C_west"])})

doy = ok["date"].dt.dayofyear
ok["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
ok["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
ok["y"] = np.log(ok["turbidity_median"])
ok["hyear"] = np.where(ok.date.dt.month >= 5, ok.date.dt.year, ok.date.dt.year - 1)  # May–Apr year
ok = ok.merge(ctrl_feat, left_on="date", right_index=True, how="inner")
ok = ok.dropna(subset=FEATURES + ["y"])


def group_of(r):
    if r.period == "capped":
        return "Capped (Nov–Jan)"
    if r.period == "closed":
        return "Closed (Feb–Apr)" if r.date.month in (2, 3, 4) else "Closed (May–Oct)"
    if r.period == "covid":
        return "COVID closure 2020"
    return "pre"


ok["group"] = ok.apply(group_of, axis=1)
ok.to_csv(os.path.join(OUT, "modelling_table.csv"), index=False)

# ------------------------------------------------------------------
# 4. MODELS
# ------------------------------------------------------------------
def make_models():
    return {
        "M1_Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "M2_RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                                 max_features=0.5, n_jobs=-1, random_state=42),
        "M3_XGBoost": XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                   random_state=42, n_jobs=-1),
    }


def fit_predict(name, model, train, test):
    if name == "M0_Ratio":
        offset = (train.y - train.log_ctrl).mean()
        return test.log_ctrl.values + offset
    model.fit(train[FEATURES], train.y)
    return model.predict(test[FEATURES])


MODEL_NAMES = ["M0_Ratio", "M1_Ridge", "M2_RandomForest", "M3_XGBoost"]
cv_rows, effect_rows, oof_all, post_all = [], [], [], []

for zone in ZONES:
    z = ok[ok.zone == zone]
    pre = z[z.group == "pre"].copy()
    post = z[z.group != "pre"].copy()

    # --- 4a. leave-one-year-out cross-validation on PRE data ---
    for name in MODEL_NAMES:
        oof = pd.Series(np.nan, index=pre.index)
        for yr in sorted(pre.hyear.unique()):
            tr, te = pre[pre.hyear != yr], pre[pre.hyear == yr]
            oof.loc[te.index] = fit_predict(name, make_models().get(name), tr, te)
        res = pre.y - oof
        cv_rows.append({"zone": zone, "model": name, "n_pre": len(pre),
                        "R2": r2_score(pre.y, oof),
                        "RMSE_log": np.sqrt(mean_squared_error(pre.y, oof)),
                        "bias_log": res.mean()})
        tmp = pre[["date", "hyear", "y"]].copy()
        tmp["pred"], tmp["zone"], tmp["model"] = oof, zone, name
        tmp["season"] = np.where(tmp.date.dt.month.isin([11, 12, 1]), "Nov-Jan",
                                 np.where(tmp.date.dt.month.isin([2, 3, 4]), "Feb-Apr", "May-Oct"))
        oof_all.append(tmp)

    # --- 4b. counterfactual for restriction period (+ COVID) ---
    for name in MODEL_NAMES:
        pred = fit_predict(name, make_models().get(name), pre, post)
        tmp = post[["date", "group", "y"]].copy()
        tmp["pred"], tmp["zone"], tmp["model"] = pred, zone, name
        post_all.append(tmp)

cv = pd.DataFrame(cv_rows)
oof_all = pd.concat(oof_all)
post_all = pd.concat(post_all)
cv.to_csv(os.path.join(OUT, "cv_metrics.csv"), index=False)
oof_all.to_csv(os.path.join(OUT, "oof_predictions.csv"), index=False)
post_all.to_csv(os.path.join(OUT, "counterfactual_predictions.csv"), index=False)

print("\nCross-validation (leave-one-year-out, log scale):")
print(cv.pivot_table(index="zone", columns="model", values="R2").round(3).reindex(ZONES))

# ------------------------------------------------------------------
# 5. EFFECTS, BOOTSTRAP CI, PLACEBO, MDE
# ------------------------------------------------------------------
SEASON_OF = {"Capped (Nov–Jan)": "Nov-Jan", "Closed (Feb–Apr)": "Feb-Apr",
             "Closed (May–Oct)": "May-Oct", "COVID closure 2020": None}

for (zone, name, grp), g in post_all.groupby(["zone", "model", "group"]):
    r = (g.y - g.pred).values
    est = r.mean()
    boot = np.array([RNG.choice(r, len(r)).mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # placebo: mean out-of-fold residual of the same season in each pre year
    o = oof_all[(oof_all.zone == zone) & (oof_all.model == name)]
    s = SEASON_OF[grp]
    if s is not None:
        o = o[o.season == s]
    plac = (o.y - o.pred).groupby(o.hyear).mean()
    plac = plac[o.groupby(o.hyear).size() >= 3]          # years with ≥3 obs
    p_plac = (np.sum(np.abs(plac) >= abs(est)) + 1) / (len(plac) + 1)
    mde = 2.8 * plac.std(ddof=1) if len(plac) > 2 else np.nan
    effect_rows.append({
        "zone": zone, "model": name, "group": grp, "n_obs": len(r),
        "effect_pct": 100 * (np.exp(est) - 1),
        "ci_low_pct": 100 * (np.exp(lo) - 1), "ci_high_pct": 100 * (np.exp(hi) - 1),
        "placebo_years": len(plac), "placebo_p": p_plac,
        "MDE_pct": 100 * (np.exp(mde) - 1) if np.isfinite(mde) else np.nan})

eff = pd.DataFrame(effect_rows)
eff.to_csv(os.path.join(OUT, "effects.csv"), index=False)

best = (cv[cv.model != "M0_Ratio"].groupby("model").R2.mean().idxmax())
print("\nBest model by mean CV R2:", best)
show = eff[(eff.model == best) & (eff.group != "COVID closure 2020")]
print("\nEffect of restrictions on turbidity (%), model =", best)
print(show.pivot_table(index="zone", columns="group", values="effect_pct").round(1).reindex(ZONES))
print("\n95% CI and placebo p-value, T_north:")
print(show[show.zone == "T_north"][["group", "n_obs", "effect_pct", "ci_low_pct",
                                    "ci_high_pct", "placebo_p", "MDE_pct"]].round(2).to_string(index=False))

# ------------------------------------------------------------------
# 6. FIGURES
# ------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

# Fig A: CV observed vs predicted (T_north, best model)
o = oof_all[(oof_all.zone == "T_north") & (oof_all.model == best)]
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.scatter(np.exp(o.pred), np.exp(o.y), s=8, alpha=0.6)
lim = [np.exp(min(o.y.min(), o.pred.min())), np.exp(max(o.y.max(), o.pred.max()))]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
r2 = cv[(cv.zone == "T_north") & (cv.model == best)].R2.iloc[0]
ax.set_xlabel("Predicted turbidity (FNU)"); ax.set_ylabel("Observed turbidity (FNU)")
ax.set_title(f"T_north, {best}\nleave-one-year-out CV, R² = {r2:.2f}")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figA_cv_T_north.png")); plt.close(fig)

# Fig B: observed vs counterfactual, T_north, restriction period (Nov–Apr)
p = post_all[(post_all.zone == "T_north") & (post_all.model == best) &
             (post_all.group.isin(["Capped (Nov–Jan)", "Closed (Feb–Apr)"]))].sort_values("date")
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(p.date, np.exp(p.pred), "o", mfc="none", color="#2c7bb6", ms=5, label="Counterfactual (no restriction)")
ax.plot(p.date, np.exp(p.y), "o", color="#d7301f", ms=4, label="Observed")
for _, r in p.iterrows():
    ax.plot([r.date, r.date], [np.exp(r.pred), np.exp(r.y)], color="grey", lw=0.5)
ax.set_yscale("log"); ax.set_ylabel("Turbidity (FNU)")
ax.set_title(f"T_north during restrictions: observed vs counterfactual ({best})")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figB_counterfactual_T_north.png")); plt.close(fig)

# Fig C: effect estimates with CI for all zones (best model)
groups = ["Capped (Nov–Jan)", "Closed (Feb–Apr)"]
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, grp in zip(axes, groups):
    e = show[show.group == grp].set_index("zone").reindex(ZONES)
    yy = np.arange(len(ZONES))[::-1]
    ax.errorbar(e.effect_pct, yy, xerr=[e.effect_pct - e.ci_low_pct, e.ci_high_pct - e.effect_pct],
                fmt="o", color="k", ecolor="grey", capsize=3)
    ax.axvline(0, color="red", lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels(ZONES)
    ax.set_xlabel("Effect on turbidity (%)"); ax.set_title(grp); ax.grid(alpha=0.3, axis="x")
fig.suptitle(f"Estimated effect of tourism restrictions (95% bootstrap CI), {best}", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figC_effects.png")); plt.close(fig)

print("\nAll outputs saved in:", OUT)
try:
    from IPython.display import Image, display
    for f in ["figA_cv_T_north.png", "figB_counterfactual_T_north.png", "figC_effects.png"]:
        display(Image(filename=os.path.join(OUT, f)))
except Exception:
    pass
