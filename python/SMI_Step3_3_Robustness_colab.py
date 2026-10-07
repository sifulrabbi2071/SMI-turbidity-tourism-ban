"""
Saint Martin's Island — Phase 3, Step 3.3
Robustness checks for the counterfactual estimates

  R1  Pixel-quality threshold: valid fraction >= 0.10 / 0.30 (main) / 0.50
  R2  Train only on Nov–Apr (drops noisy post-monsoon days)
  R3  Model choice: Ridge (main) vs XGBoost
  R4  Placebo-based 95% interval (year-to-year model error)
  R5  Known-effect injection: add an artificial −10% / −20% change to
      pre-restriction years and check that the method recovers it
      (forward-in-time: train on earlier years only)
  R6  Which drivers matter? (permutation importance, T_north)
  R7  COVID-2020 signal: date-by-date look
  R8  Algorithm-independent outcome: median red reflectance (B4) in
      place of turbidity, for both the response and the control
      predictors (added in v1.1)

Inputs : SMI_project/outputs/SMI_analysis_dataset_v1.csv       (Step 3.1)
         SMI_project/outputs/step3_2/tides_GOT410.csv           (Step 3.2)
Outputs: SMI_project/outputs/step3_3/...
Version 1.1 (October 2026) — R8 red-reflectance check added
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE = os.environ.get("SMI_BASE", "/content/drive/MyDrive/SMI_project/outputs")
OUT = os.path.join(BASE, "step3_3")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)

MAIN_ZONES = ["T_north", "T_east", "T_west", "T_south"]
FEATURES = ["log_ctrl", "ctrl_EW", "ws10_24h", "ws10_72h", "u10_24h", "v10_24h",
            "swh_24h", "sst_C", "era5_rain_mm_catch_72h", "chirps_catch_7d",
            "chirps_catch_30d", "tide_now", "tide_range", "tide_trend",
            "doy_sin", "doy_cos", "sun_zenith"]
GROUPS = {"Capped (Nov–Jan)": [11, 12, 1], "Closed (Feb–Apr)": [2, 3, 4]}

raw = pd.read_csv(os.path.join(BASE, "SMI_analysis_dataset_v1.csv"), parse_dates=["date"])
raw["overpass"] = pd.to_datetime(raw["date"].dt.strftime("%Y-%m-%d") + " " + raw["time_utc"])
tides = pd.read_csv(os.path.join(BASE, "step3_2", "tides_GOT410.csv"), parse_dates=["overpass"])
raw = raw.merge(tides, on="overpass", how="left")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def build_table(valid_min=0.30, var="turbidity_median"):
    """Modelling table for a given pixel-quality threshold and optical variable."""
    d = raw.copy()
    d["ok"] = (d.valid_mean >= valid_min) & d[var].notna() & (d[var] > 0)
    d = d[d.ok]
    ctrl = d[d.zone.isin(["C_east", "C_west"])].pivot_table(
        index="date", columns="zone", values=var).dropna()
    cf = pd.DataFrame({"log_ctrl": np.log(ctrl).mean(axis=1),
                       "ctrl_EW": np.log(ctrl.C_east / ctrl.C_west)})
    d = d[d.zone.isin(MAIN_ZONES)].merge(cf, left_on="date", right_index=True)
    doy = d.date.dt.dayofyear
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    d["y"] = np.log(d[var])
    d["hyear"] = np.where(d.date.dt.month >= 5, d.date.dt.year, d.date.dt.year - 1)
    d["month"] = d.date.dt.month
    return d.dropna(subset=FEATURES + ["y"])


def model(kind):
    if kind == "Ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
    return XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=42, n_jobs=-1)


def placebo_errors(pre, kind, months):
    """Leave-one-year-out mean residual for the given months in each pre year."""
    out = {}
    for yr in sorted(pre.hyear.unique()):
        te = pre[(pre.hyear == yr) & pre.month.isin(months)]
        if len(te) < 3:
            continue
        m = model(kind).fit(pre[pre.hyear != yr][FEATURES], pre[pre.hyear != yr].y)
        out[yr] = (te.y - m.predict(te[FEATURES])).mean()
    return pd.Series(out)


def estimate(d, kind="Ridge", train_months=None):
    """Effects for each zone and restriction group."""
    rows = []
    for zone in MAIN_ZONES:
        z = d[d.zone == zone]
        pre = z[z.period == "pre"]
        if train_months is not None:
            pre = pre[pre.month.isin(train_months)]
        m = model(kind).fit(pre[FEATURES], pre.y)
        for grp, months in GROUPS.items():
            per = "capped" if grp.startswith("Capped") else "closed"
            post = z[(z.period == per) & z.month.isin(months)]
            r = (post.y - m.predict(post[FEATURES])).values
            est = r.mean()
            boot = np.array([RNG.choice(r, len(r)).mean() for _ in range(1000)])
            plac = placebo_errors(pre, kind, months)
            sd = plac.std(ddof=1)
            rows.append({"zone": zone, "group": grp, "n": len(r),
                         "effect_pct": 100 * (np.exp(est) - 1),
                         "boot_lo": 100 * (np.exp(np.percentile(boot, 2.5)) - 1),
                         "boot_hi": 100 * (np.exp(np.percentile(boot, 97.5)) - 1),
                         "plac_lo": 100 * (np.exp(est - 1.96 * sd) - 1),
                         "plac_hi": 100 * (np.exp(est + 1.96 * sd) - 1),
                         "placebo_p": (np.sum(np.abs(plac) >= abs(est)) + 1) / (len(plac) + 1),
                         "n_placebo_years": len(plac)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# R1–R4: specifications
# ------------------------------------------------------------------
specs = [
    ("Main (valid≥0.30, Ridge)", dict(valid_min=0.30), dict(kind="Ridge")),
    ("valid≥0.10", dict(valid_min=0.10), dict(kind="Ridge")),
    ("valid≥0.50", dict(valid_min=0.50), dict(kind="Ridge")),
    ("Train Nov–Apr only", dict(valid_min=0.30), dict(kind="Ridge", train_months=[11, 12, 1, 2, 3, 4])),
    ("XGBoost", dict(valid_min=0.30), dict(kind="XGB")),
    ("Red reflectance (B4)", dict(valid_min=0.30, var="rho_red_median"), dict(kind="Ridge")),   # R8
]
res = []
for name, tkw, ekw in specs:
    e = estimate(build_table(**tkw), **ekw)
    e.insert(0, "spec", name)
    res.append(e)
    print("done:", name)
rob = pd.concat(res)
rob.to_csv(os.path.join(OUT, "robustness_effects.csv"), index=False)

print("\nEffect (%) by specification:")
print(rob.pivot_table(index=["group", "spec"], columns="zone", values="effect_pct",
                      sort=False)[MAIN_ZONES].round(1))
print("\nMain spec: bootstrap vs placebo-based 95% intervals (%):")
print(rob[rob.spec.str.startswith("Main")][["zone", "group", "n", "effect_pct", "boot_lo", "boot_hi",
                                              "plac_lo", "plac_hi", "placebo_p"]].round(1).to_string(index=False))

# ------------------------------------------------------------------
# R5: known-effect injection (forward in time)
# ------------------------------------------------------------------
main = build_table(0.30)
inj_rows = []
for zone in MAIN_ZONES:
    z = main[(main.zone == zone) & (main.period == "pre")]
    for test_year in [2021, 2022, 2023]:                # pseudo-restriction season (Nov–Apr)
        train = z[z.hyear < test_year]
        test = z[(z.hyear == test_year) & z.month.isin([11, 12, 1, 2, 3, 4])]
        if len(test) < 5:
            continue
        m = model("Ridge").fit(train[FEATURES], train.y)
        for true in [0.0, -10.0, -20.0]:
            y_inj = test.y + np.log(1 + true / 100)
            est = 100 * (np.exp((y_inj - m.predict(test[FEATURES])).mean()) - 1)
            inj_rows.append({"zone": zone, "pseudo_season": f"{test_year}/{test_year + 1}",
                             "true_effect_pct": true, "estimated_pct": est})
inj = pd.DataFrame(inj_rows)
inj.to_csv(os.path.join(OUT, "injection_test.csv"), index=False)
print("\nKnown-effect injection (train on earlier years, predict pseudo-season):")
print(inj.groupby("true_effect_pct").estimated_pct.agg(["mean", "std", "min", "max", "count"]).round(1))

# ------------------------------------------------------------------
# R6: permutation importance (T_north, Ridge, pre data)
# ------------------------------------------------------------------
tn = main[(main.zone == "T_north") & (main.period == "pre")]
m = model("Ridge").fit(tn[FEATURES], tn.y)
pi = permutation_importance(m, tn[FEATURES], tn.y, n_repeats=30, random_state=42)
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
imp.to_csv(os.path.join(OUT, "importance_T_north.csv"))
print("\nTop drivers of T_north turbidity (drop in R² when shuffled):")
print(imp.sort_values(ascending=False).head(8).round(3))

# ------------------------------------------------------------------
# R7: COVID-2020 dates
# ------------------------------------------------------------------
cov_rows = []
for zone in MAIN_ZONES:
    z = main[main.zone == zone]
    pre = z[z.period == "pre"]
    m = model("Ridge").fit(pre[FEATURES], pre.y)
    c = z[z.period == "covid"]
    for _, r in c.iterrows():
        cov_rows.append({"date": r.date.date(), "zone": zone,
                         "observed": r.turbidity_median,
                         "counterfactual": float(np.exp(m.predict(r[FEATURES].to_frame().T.astype(float))[0])),
                         "control_ref": float(np.exp(r.log_ctrl))})
cov = pd.DataFrame(cov_rows)
cov["effect_pct"] = 100 * (cov.observed / cov.counterfactual - 1)
cov.to_csv(os.path.join(OUT, "covid_dates.csv"), index=False)
print("\nCOVID-2020, effect (%) by date and zone:")
print(cov.pivot_table(index="date", columns="zone", values="effect_pct")[MAIN_ZONES].round(1))

# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
spec_names = [s[0] for s in specs]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
for ax, grp in zip(axes, GROUPS):
    for i, zone in enumerate(MAIN_ZONES):
        for j, s in enumerate(spec_names):
            r = rob[(rob.spec == s) & (rob.zone == zone) & (rob.group == grp)].iloc[0]
            yv = i * (len(spec_names) + 1) + j
            ax.plot([r.plac_lo, r.plac_hi], [yv, yv], color="lightgrey", lw=3)
            ax.plot([r.boot_lo, r.boot_hi], [yv, yv], color="grey", lw=1)
            ax.plot(r.effect_pct, yv, "o", color="k" if j == 0 else "#2c7bb6", ms=4)
    ticks = [i * (len(spec_names) + 1) + j for i in range(4) for j in range(len(spec_names))]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{z.replace('T_', '')}: {s}" for z in MAIN_ZONES for s in spec_names], fontsize=6)
    ax.invert_yaxis(); ax.axvline(0, color="red", lw=0.8)
    ax.set_title(grp); ax.set_xlabel("Effect on turbidity (%)"); ax.grid(alpha=0.3, axis="x")
fig.suptitle("Robustness: dark bar = bootstrap 95% CI, light bar = placebo-based 95% interval", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figD_robustness.png")); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4))
imp.tail(10).plot.barh(ax=ax, color="#2c7bb6")
ax.set_xlabel("Permutation importance (drop in R²)")
ax.set_title("Drivers of nearshore turbidity, T_north (pre-restriction)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figE_importance.png")); plt.close(fig)

print("\nAll outputs saved in:", OUT)
try:
    from IPython.display import Image, display
    for f in ["figD_robustness.png", "figE_importance.png"]:
        display(Image(filename=os.path.join(OUT, f)))
except Exception:
    pass
