"""
Saint Martin's Island — Phase 3, Step 3.5
Manuscript numbers, tables and figures

Recomputes, from your own Drive files, every number in the manuscript that
was not already printed by Steps 3.1–3.4, and redraws all data figures.
Every value is written to a report so that it can be checked line by line
against the manuscript.

  A  Data availability and turbidity patterns       (Results 4.1)
  B  Physical setting: waves, rainfall, tides       (Study area 2.1)
  C  Study-zone areas                               (Methods 3.2)
  D  Night-time lights: model details and effects   (Results 4.2)
  E  Cross-validation table                         (Table 5)
  F  Effects, placebo p-values, minimum detectable reductions (Table 6)
  G  Robustness table incl. red reflectance         (Table 7, Results 4.5)
  H  Capped season split: November vs Dec–Jan       (Results 4.5)
  I  Known-effect injection, permutation importance (Results 4.3, 4.5)
  J  COVID-19 dates and published field comparison  (Results 4.6)
  K  Oxygen solubility (Weiss, 1970)                (Discussion 5.3)
  L  Figures 1, 3, 4, 5, 6, 7                       (600 dpi PNG + PDF)

RUN ORDER: Steps 3.1, 3.2, 3.3 (version 1.1, with the red-reflectance check)
and 3.4 must have been run first. Then:
    1. from google.colab import drive; drive.mount('/content/drive')
    2. paste this script into a new cell and run it (2–5 minutes).
Outputs: SMI_project/outputs/step3_5/
    manuscript_numbers.txt   all numbers, grouped by manuscript section
    manuscript_numbers.csv   the same, as a table
    table5_cv.csv, table6_effects.csv, table7_robustness.csv
    Fig1_study_area, Fig3_availability_seasonality, Fig4_night_lights,
    Fig5_counterfactual_north, Fig6_effects, Fig7_robustness (.png, .pdf)
Version 1.0 (October 2026)
"""
import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 0. PATHS
# ------------------------------------------------------------------
PROJECT = os.environ.get("SMI_DATA", "/content/drive/MyDrive/SMI_project")
BASE = os.environ.get("SMI_BASE", os.path.join(PROJECT, "outputs"))
OUT = os.path.join(BASE, "step3_5")
os.makedirs(OUT, exist_ok=True)

P = {
    "zones": os.path.join(PROJECT, "SMI_zones_v2_geojson.geojson"),
    "cov": os.path.join(PROJECT, "SMI_covariates_v1.csv"),
    "lights": os.path.join(PROJECT, "SMI_night_lights_v1.csv"),
    "data": os.path.join(BASE, "SMI_analysis_dataset_v1.csv"),
    "tides": os.path.join(BASE, "step3_2", "tides_GOT410.csv"),
    "model": os.path.join(BASE, "step3_2", "modelling_table.csv"),
    "cv": os.path.join(BASE, "step3_2", "cv_metrics.csv"),
    "eff": os.path.join(BASE, "step3_2", "effects.csv"),
    "oof": os.path.join(BASE, "step3_2", "oof_predictions.csv"),
    "cf": os.path.join(BASE, "step3_2", "counterfactual_predictions.csv"),
    "rob": os.path.join(BASE, "step3_3", "robustness_effects.csv"),
    "inj": os.path.join(BASE, "step3_3", "injection_test.csv"),
    "imp": os.path.join(BASE, "step3_3", "importance_T_north.csv"),
    "covid": os.path.join(BASE, "step3_3", "covid_dates.csv"),
    "nl_eff": os.path.join(BASE, "step3_4", "night_lights_effects.csv"),
}
missing = [f"{k}: {v}" for k, v in P.items() if not os.path.exists(v)]
if missing:
    raise FileNotFoundError("These input files were not found:\n  " + "\n  ".join(missing))

rob = pd.read_csv(P["rob"])
if not rob["spec"].str.startswith("Red").any():
    raise RuntimeError("robustness_effects.csv has no red-reflectance rows.\n"
                       "Please re-run Step 3.3 version 1.1 first, then run this script again.")

# ------------------------------------------------------------------
# Report helper
# ------------------------------------------------------------------
rows = []


def rep(section, label, value):
    rows.append({"section": section, "item": label, "value": value})
    print(f"[{section}] {label}: {value}")


def pct(x):
    return f"{x:+.1f}%".replace("+-", "−").replace("-", "−")


def rng(a, b, d=1, unit=""):
    return f"{a:.{d}f}–{b:.{d}f}{unit}"


SEC = ["T_north", "T_east", "T_west", "T_south"]
LAB = {"T_north": "North", "T_east": "East", "T_west": "West", "T_south": "South"}
COL = {"T_north": "#D55E00", "T_east": "#E69F00", "T_west": "#CC79A7", "T_south": "#009E73",
       "C_east": "#56B4E9", "C_west": "#0072B2"}
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FEATURES = ["log_ctrl", "ctrl_EW", "ws10_24h", "ws10_72h", "u10_24h", "v10_24h", "swh_24h", "sst_C",
            "era5_rain_mm_catch_72h", "chirps_catch_7d", "chirps_catch_30d", "tide_now", "tide_range",
            "tide_trend", "doy_sin", "doy_cos", "sun_zenith"]

d = pd.read_csv(P["data"], parse_dates=["date"])
ok = d[d["ok"]]

# ------------------------------------------------------------------
# A. Data availability and turbidity patterns (Results 4.1)
# ------------------------------------------------------------------
S = "A Results 4.1"
dates = d.drop_duplicates("date")
rep(S, "Acquisition dates", f"{len(dates)} ({dates.date.min().date()} to {dates.date.max().date()})")
for sc, n in dates.groupby("spacecraft").size().items():
    first = dates[dates.spacecraft == sc].date.min().date()
    rep(S, f"Dates from {sc}", f"{n} (first {first})")
rep(S, "Overpass times (UTC)", ", ".join(sorted(dates.time_utc.astype(str).unique())))
rep(S, "Zone–date records / usable (valid ≥ 0.30)", f"{len(d)} / {len(ok)} ({100*len(ok)/len(d):.1f}%)")
tn = ok[ok.zone == "T_north"]
bym = tn.groupby(tn.date.dt.month).size().reindex(range(1, 13), fill_value=0)
rep(S, "Usable dates, northern sector", len(tn))
rep(S, "  of which Nov–Mar", f"{bym[[11,12,1,2,3]].sum()} ({100*bym[[11,12,1,2,3]].sum()/len(tn):.0f}%)")
rep(S, "  of which May–Aug", int(bym[[5, 6, 7, 8]].sum()))
rep(S, "  by month (Jan..Dec)", ", ".join(str(int(v)) for v in bym.values))
rep(S, "Usable dates, other sectors",
    ", ".join(f"{LAB[z]} {len(ok[ok.zone == z])}" for z in SEC[1:]))
pre = ok[ok.period == "pre"]
clim = pre.pivot_table(index=pre.date.dt.month, columns="zone", values="turbidity_median", aggfunc="median")
six = SEC + ["C_east", "C_west"]
for m in [11, 3, 4]:
    v = clim.loc[m, six]
    rep(S, f"Pre-restriction median turbidity, {MON[m-1]} (six zones, FNU)", rng(v.min(), v.max()))
rep(S, "Pre-restriction usable April dates, northern sector", int((pre[pre.zone == "T_north"].date.dt.month == 4).sum()))
ratio = pre[pre.zone.isin(SEC)].groupby("zone")["log_ratio"].median().apply(np.exp)
rep(S, "Median nearshore/offshore ratio", ", ".join(f"{LAB[z]} {ratio[z]:.2f}" for z in SEC))
tnp = pre[pre.zone == "T_north"]
rep(S, "Spearman rho, northern sector vs control", f"{tnp[['turbidity_median','control_ref']].corr('spearman').iloc[0,1]:.2f}")
rep(S, "Maximum possible reduction if all nearshore excess removed",
    rng(100 * (1 - 1 / ratio.min()), 100 * (1 - 1 / ratio.max()), 1, "%"))

# ------------------------------------------------------------------
# B. Physical setting (Study area 2.1)
# ------------------------------------------------------------------
S = "B Study area 2.1"
cov = pd.read_csv(P["cov"], parse_dates=["date"])
cov["m"] = cov.date.dt.month
mm = cov.groupby("m").agg(swh=("swh_24h", "mean"), rain30=("chirps_catch_30d", "mean"))
rep(S, "Mean significant wave height, Nov–Mar (m)", rng(mm.loc[[11, 12, 1, 2, 3], "swh"].min(), mm.loc[[11, 12, 1, 2, 3], "swh"].max(), 2))
rep(S, "Mean significant wave height, Jun–Sep (m)", rng(mm.loc[[6, 7, 8, 9], "swh"].min(), mm.loc[[6, 7, 8, 9], "swh"].max(), 2))
rep(S, "Mean 30-day catchment rainfall, Dec–Mar (mm)", f"max {mm.loc[[12, 1, 2, 3], 'rain30'].max():.1f}")
rep(S, "Mean 30-day catchment rainfall, Jun–Sep (mm)", rng(mm.loc[[6, 7, 8, 9], "rain30"].min(), mm.loc[[6, 7, 8, 9], "rain30"].max(), 0))
tides = pd.read_csv(P["tides"])
rep(S, "Tide height at overpass, all overpasses (m)", rng(tides.tide_now.min(), tides.tide_now.max(), 2))
rep(S, "24-h tidal range, all overpasses (m)", rng(tides.tide_range.min(), tides.tide_range.max(), 2))

# ------------------------------------------------------------------
# C. Zone areas (Methods 3.2)
# ------------------------------------------------------------------
S = "C Methods 3.2"
zones = gpd.read_file(P["zones"]).set_crs(4326, allow_override=True)
area = dict(zip(zones.zone, zones.area_km2))
rep(S, "Mapped island area (km²)", f"{area['island_land']:.1f}")
rep(S, "Treatment sector areas (km²)", ", ".join(f"{LAB[z]} {area[z]:.1f}" for z in SEC))
rep(S, "Control sector areas (km²)", f"East {area['C_east']:.1f}, West {area['C_west']:.1f}")
rep(S, "Excluded ferry corridor (km²)", f"{area['ferry_corridor']:.1f}")
deep = d[d.zone.str.endswith("_deep")].groupby("zone").area_km2.first()
rep(S, "Deep-variant areas (km²)", rng(deep.min(), deep.max()))

# ------------------------------------------------------------------
# D. Night-time lights (Results 4.2)
# ------------------------------------------------------------------
S = "D Results 4.2"
v = pd.read_csv(P["lights"], parse_dates=["month"])
rep(S, "Monthly composites", f"{len(v)} ({v.month.min():%Y-%m} to {v.month.max():%Y-%m})")
v = v[(v.island_cloudfree_nights >= 3) & (v.island_light_sum > 0) & (v.teknaf_light_sum > 0)].copy()
rep(S, "Months meeting inclusion criteria", len(v))
v["m"] = v.month.dt.month
v["t"] = (v.month - pd.Timestamp("2017-01-01")).dt.days / 365.25
v["y"], v["x"] = np.log(v.island_light_sum), np.log(v.teknaf_light_sum)
v["hyear"] = np.where(v.m >= 5, v.month.dt.year, v.month.dt.year - 1)


def nl_group(r):
    if r.month >= pd.Timestamp("2024-11-01"):
        return "Capped" if r.m in (11, 12, 1) else ("Closed Feb–Apr" if r.m in (2, 3, 4) else "Closed May–Oct")
    if pd.Timestamp("2020-04-01") <= r.month < pd.Timestamp("2020-11-01"):
        return "COVID"
    return "pre"


v["g"] = v.apply(nl_group, axis=1)


def Xn(dd):
    D = pd.get_dummies(dd.m.astype(str), prefix="m").reindex(columns=[f"m_{i}" for i in range(2, 13)], fill_value=0)
    return pd.concat([dd[["x", "t"]], D], axis=1).astype(float)


vp = v[v.g == "pre"].copy()
vp["res"] = np.nan
for yr in vp.hyear.unique():
    tr, te = vp[vp.hyear != yr], vp[vp.hyear == yr]
    vp.loc[te.index, "res"] = te.y - LinearRegression().fit(Xn(tr), tr.y).predict(Xn(te))
nl_model = LinearRegression().fit(Xn(vp), vp.y)
vpo = v[v.g != "pre"].copy()
vpo["res"] = vpo.y - nl_model.predict(Xn(vpo))
rep(S, "Pre-restriction months", len(vp))
rep(S, "Leave-one-year-out R²", f"{r2_score(vp.y, vp.y - vp.res):.2f}")
rep(S, "Coefficient of log Teknaf radiance", f"{nl_model.coef_[0]:.2f}")
rep(S, "Trend independent of Teknaf (% per year)", f"{100*(np.exp(nl_model.coef_[1])-1):.1f}")
nle = pd.read_csv(P["nl_eff"])
for _, r in nle.iterrows():
    rep(S, f"Effect, {r.group}", f"{pct(r.effect_pct)} (placebo 95% interval {pct(r.placebo_lo_pct)} to "
        f"{pct(r.placebo_hi_pct)}; p = {r.placebo_p:.2f}; months below counterfactual {r.months_negative})")
fa = vpo[vpo.g == "Closed Feb–Apr"]
rep(S, "Closure (Feb–Apr) monthly effects", rng(100*(np.exp(fa.res.max())-1), 100*(np.exp(fa.res.min())-1), 1, "%"))
for g in ["Capped"]:
    q = vpo[vpo.g == g]
    rep(S, "Capped-season monthly effects",
        "; ".join(f"{r.month:%b %Y} {pct(100*(np.exp(r.res)-1))}" for _, r in q.iterrows()))
npl = vp[vp.m.isin([2, 3, 4])].groupby("hyear").res.mean()
rep(S, "Placebo years, Feb–Apr (smallest attainable p)", f"{len(npl)} ({1/(len(npl)+1):.2f})")

# ------------------------------------------------------------------
# E. Cross-validation (Table 5, Results 4.3)
# ------------------------------------------------------------------
S = "E Table 5"
cv = pd.read_csv(P["cv"])
cv = cv[cv.zone.isin(SEC)]
t5 = cv.pivot_table(index="zone", columns="model", values="R2").reindex(SEC)
t5r = cv.pivot_table(index="zone", columns="model", values="RMSE_log").reindex(SEC)
tab5 = pd.DataFrame({"Sector": [LAB[z] for z in SEC],
                     "Pre-restriction dates": cv[cv.model == "M1_Ridge"].set_index("zone").reindex(SEC).n_pre.values})
for mname in ["M0_Ratio", "M1_Ridge", "M2_RandomForest", "M3_XGBoost"]:
    tab5[mname] = [f"{t5.loc[z, mname]:.3f} ({t5r.loc[z, mname]:.3f})" for z in SEC]
tab5.to_csv(os.path.join(OUT, "table5_cv.csv"), index=False)
best = t5.idxmax(axis=1)
rep(S, "Best model per sector", ", ".join(f"{LAB[z]} {best[z]}" for z in SEC))
rep(S, "Ridge R² range", rng(t5["M1_Ridge"].min(), t5["M1_Ridge"].max(), 2))
rep(S, "Ridge RMSE range", rng(t5r["M1_Ridge"].min(), t5r["M1_Ridge"].max(), 2))
rep(S, "Ridge max |bias|", f"{cv[cv.model=='M1_Ridge'].bias_log.abs().max():.4f}")
rep(S, "Ratio-baseline R² range", rng(t5["M0_Ratio"].min(), t5["M0_Ratio"].max(), 2))
imp = pd.read_csv(P["imp"], index_col=0).iloc[:, 0].sort_values(ascending=False)
rep(S, "Permutation importance (northern sector, top 6)", ", ".join(f"{k} {val:.3f}" for k, val in imp.head(6).items()))

# ------------------------------------------------------------------
# F. Effects (Table 6, Results 4.4)
# ------------------------------------------------------------------
S = "F Table 6"
main = rob[rob.spec.str.startswith("Main")].copy()
main["sd"] = np.log((100 + main.plac_hi) / (100 + main.effect_pct)) / 1.96
main["MDE_reduction"] = 100 * (1 - np.exp(-2.8 * main.sd))
tab6 = main[["group", "zone", "n", "effect_pct", "boot_lo", "boot_hi", "plac_lo", "plac_hi", "placebo_p", "MDE_reduction"]].round(3)
tab6.to_csv(os.path.join(OUT, "table6_effects.csv"), index=False)
for grp in ["Capped (Nov–Jan)", "Closed (Feb–Apr)"]:
    q = main[main.group == grp].set_index("zone").reindex(SEC)
    for z in SEC:
        r = q.loc[z]
        rep(S, f"{grp}, {LAB[z]}", f"n {int(r.n)}; {pct(r.effect_pct)}; bootstrap {r.boot_lo:.1f} to {r.boot_hi:.1f}; "
            f"placebo {r.plac_lo:.1f} to {r.plac_hi:.1f}; p {r.placebo_p:.3f}; MDE {r.MDE_reduction:.1f}%")
    rep(S, f"{grp}: effect range / MDE range",
        f"{q.effect_pct.min():.1f} to {q.effect_pct.max():.1f}% / {q.MDE_reduction.min():.1f}–{q.MDE_reduction.max():.1f}%")
eff = pd.read_csv(P["eff"])
mo = eff[(eff.model == "M1_Ridge") & (eff.zone.isin(SEC)) & (eff.group == "Closed (May–Oct)")].set_index("zone").reindex(SEC)
rep(S, "Monsoon closure (May–Oct) effects", ", ".join(f"{LAB[z]} {pct(mo.loc[z,'effect_pct'])} (n {int(mo.loc[z,'n_obs'])})" for z in SEC))
rep(S, "Monsoon closure bootstrap span", f"{mo.ci_low_pct.min():.1f}% to {mo.ci_high_pct.max():.1f}%")
md = pd.read_csv(P["model"], parse_dates=["date"])
mo_dates = md[(md.group == "Closed (May–Oct)") & (md.zone == "T_north")].date
rep(S, "Monsoon closure usable dates (northern sector)", f"{mo_dates.min().date()} to {mo_dates.max().date()}")

# ------------------------------------------------------------------
# G. Robustness (Table 7, Results 4.5)
# ------------------------------------------------------------------
S = "G Table 7"
spec_order = [("Main (valid≥0.30, Ridge)", "Main analysis"), ("valid≥0.10", "Valid fraction ≥ 0.10"),
              ("valid≥0.50", "Valid fraction ≥ 0.50"), ("Train Nov–Apr only", "Training on Nov–Apr only"),
              ("XGBoost", "XGBoost"), ("deep", "Deep sectors (300–1000 m)"), ("Red reflectance (B4)", "Red reflectance (B4)")]
deep_eff = eff[(eff.model == "M1_Ridge") & eff.zone.str.endswith("_deep")].copy()
deep_eff["zone"] = deep_eff.zone.str.replace("_deep", "")
t7 = []
for key, label in spec_order:
    row = {"Specification": label}
    for grp, short in [("Closed (Feb–Apr)", "Closure"), ("Capped (Nov–Jan)", "Capped")]:
        for z in SEC:
            if key == "deep":
                val = float(deep_eff[(deep_eff.zone == z) & (deep_eff.group == grp)].effect_pct.iloc[0])
            else:
                val = float(rob[(rob.spec == key) & (rob.zone == z) & (rob.group == grp)].effect_pct.iloc[0])
            row[f"{short} {LAB[z][0]}"] = round(val, 1)
    t7.append(row)
tab7 = pd.DataFrame(t7)
tab7.to_csv(os.path.join(OUT, "table7_robustness.csv"), index=False)
clos = tab7[[c for c in tab7.columns if c.startswith("Closure")]].values
rep(S, "All closure estimates (7 specifications × 4 sectors)", f"{clos.min():.1f}% to {clos.max():.1f}% (n = {clos.size})")
pmin = rob[rob.group == "Closed (Feb–Apr)"].placebo_p.min()
pdeep = eff[(eff.model == "M1_Ridge") & eff.zone.str.endswith("_deep") & (eff.group == "Closed (Feb–Apr)")].placebo_p.min()
rep(S, "Smallest closure placebo p (all specifications incl. deep)", f"{min(pmin, pdeep):.3f}")
ws = tab7[tab7.Specification.isin(["Valid fraction ≥ 0.10", "Valid fraction ≥ 0.50", "Training on Nov–Apr only",
                                   "Deep sectors (300–1000 m)", "Main analysis"])][["Capped W", "Capped S"]].values
rep(S, "Capped W and S across thresholds, training windows, deep sectors", f"{ws.min():.1f}% to {ws.max():.1f}%")
red = rob[rob.spec.str.startswith("Red")].copy()
red["sd"] = np.log((100 + red.plac_hi) / (100 + red.effect_pct)) / 1.96
red["MDE_reduction"] = 100 * (1 - np.exp(-2.8 * red.sd))
rc = red[red.group == "Closed (Feb–Apr)"]
rep(S, "Red reflectance, closure effects", f"{rc.effect_pct.min():.1f}% to {rc.effect_pct.max():.1f}% (p {rc.placebo_p.min():.3f}–{rc.placebo_p.max():.3f})")
rep(S, "Red reflectance, closure MDE (reduction)", rng(rc.MDE_reduction.min(), rc.MDE_reduction.max(), 1, "%"))
for z in SEC:
    r = red[(red.group == "Capped (Nov–Jan)") & (red.zone == z)].iloc[0]
    rep(S, f"Red reflectance, capped, {LAB[z]}", f"{pct(r.effect_pct)} (placebo {r.plac_lo:.1f} to {r.plac_hi:.1f}; p {r.placebo_p:.3f})")
xg = rob[(rob.spec == "XGBoost") & (rob.group == "Capped (Nov–Jan)")].set_index("zone")
rep(S, "XGBoost, capped W and S", f"{pct(xg.loc['T_west','effect_pct'])} (p {xg.loc['T_west','placebo_p']:.3f}), "
    f"{pct(xg.loc['T_south','effect_pct'])} (p {xg.loc['T_south','placebo_p']:.3f})")

# ------------------------------------------------------------------
# H. Capped season split: November vs December–January
# ------------------------------------------------------------------
S = "H Results 4.5"
md["month"] = md.date.dt.month


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))


split_rows = []
for z in SEC:
    zz = md[md.zone == z]
    zp = zz[zz.group == "pre"]
    mod = ridge().fit(zp[FEATURES], zp.y)
    for name, months in [("November", [11]), ("December–January", [12, 1])]:
        post = zz[(zz.group == "Capped (Nov–Jan)") & zz.month.isin(months)]
        est = (post.y - mod.predict(post[FEATURES])).mean()
        plac = {}
        for yr in sorted(zp.hyear.unique()):
            te = zp[(zp.hyear == yr) & zp.month.isin(months)]
            if len(te) < 3:
                continue
            mm_ = ridge().fit(zp[zp.hyear != yr][FEATURES], zp[zp.hyear != yr].y)
            plac[yr] = (te.y - mm_.predict(te[FEATURES])).mean()
        plac = pd.Series(plac)
        p = (np.sum(np.abs(plac) >= abs(est)) + 1) / (len(plac) + 1)
        split_rows.append({"zone": z, "part": name, "n": len(post), "effect_pct": 100 * (np.exp(est) - 1),
                           "placebo_p": p, "placebo_years": len(plac)})
sp = pd.DataFrame(split_rows)
sp.to_csv(os.path.join(OUT, "capped_split_nov_decjan.csv"), index=False)
for name in ["November", "December–January"]:
    q = sp[sp.part == name]
    rep(S, f"{name} effects", f"{q.effect_pct.min():.1f}% to {q.effect_pct.max():.1f}% "
        f"(p {q.placebo_p.min():.2f}–{q.placebo_p.max():.2f}; placebo years {q.placebo_years.min()}–{q.placebo_years.max()})")

# ------------------------------------------------------------------
# I. Known-effect injection (Results 4.5)
# ------------------------------------------------------------------
S = "I Results 4.5"
inj = pd.read_csv(P["inj"]).groupby("true_effect_pct").estimated_pct.agg(["mean", "std", "count"])
for t, r in inj.sort_index(ascending=False).iterrows():
    rep(S, f"Injection, true effect {t:.0f}%", f"mean {r['mean']:.1f}% (SD {r['std']:.1f}; n {int(r['count'])})")

# ------------------------------------------------------------------
# J. COVID-19 dates and published field comparison (Results 4.6)
# ------------------------------------------------------------------
S = "J Results 4.6"
cdates = pd.read_csv(P["covid"], parse_dates=["date"])
rep(S, "COVID sector–dates below counterfactual", f"{(cdates.effect_pct < 0).sum()} of {len(cdates)}")
rep(S, "COVID dates", ", ".join(sorted(cdates.date.dt.strftime("%Y-%m-%d").unique())))
big = cdates.nsmallest(3, "effect_pct")
rep(S, "Three largest COVID deficits", "; ".join(f"{r.date:%d %b %Y} {r.zone} {pct(r.effect_pct)}" for _, r in big.iterrows()))
ce = eff[(eff.model == "M1_Ridge") & eff.zone.isin(SEC) & (eff.group == "COVID closure 2020")]
rep(S, "COVID mean effects across sectors", f"{ce.effect_pct.min():.1f}% to {ce.effect_pct.max():.1f}% (placebo p {ce.placebo_p.min():.2f})")
# Published field values (Rahman et al., 2026; typed from the paper)
FIELD = {"2025-10-14": 3.2, "2025-10-18": 0.7}
sat = ok[ok.zone.isin(["T_east", "T_north"]) & ok.date.isin(pd.to_datetime(["2025-10-14", "2025-10-19"]))]
sat = sat.pivot_table(index="date", columns="zone", values="turbidity_median")
e14, e19 = sat.loc["2025-10-14", "T_east"], sat.loc["2025-10-19", "T_east"]
rep(S, "Sentinel-2, 14 Oct 2025 (FNU)", f"East {e14:.2f}, North {sat.loc['2025-10-14','T_north']:.2f}")
rep(S, "Sentinel-2, 19 Oct 2025 (FNU)", f"East {e19:.2f}, North {sat.loc['2025-10-19','T_north']:.2f}")
rep(S, "Satellite / field ratio (eastern sector)", f"{e14/FIELD['2025-10-14']:.1f} and {e19/FIELD['2025-10-18']:.1f}")
rep(S, "Decline between the two dates", f"field {100*(1-FIELD['2025-10-18']/FIELD['2025-10-14']):.0f}%, "
    f"satellite east {100*(1-e19/e14):.0f}%")

# ------------------------------------------------------------------
# K. Oxygen solubility (Weiss, 1970) at salinity 33
# ------------------------------------------------------------------
S = "K Discussion 5.3"


def o2_mg_per_l(temp_c, sal):
    tk = temp_c + 273.15
    ln_c = (-173.4292 + 249.6339 * (100 / tk) + 143.3483 * np.log(tk / 100) - 21.8492 * (tk / 100)
            + sal * (-0.033096 + 0.014259 * (tk / 100) - 0.0017 * (tk / 100) ** 2))
    return np.exp(ln_c) * 1.42905          # mL/L -> mg/L


rep(S, "O2 solubility at 27 °C / 31.4 °C, S = 33 (mg/L)", f"{o2_mg_per_l(27, 33):.2f} / {o2_mg_per_l(31.4, 33):.2f} "
    f"(difference {o2_mg_per_l(27, 33)-o2_mg_per_l(31.4, 33):.2f})")

# ------------------------------------------------------------------
# Save report
# ------------------------------------------------------------------
rep_df = pd.DataFrame(rows)
rep_df.to_csv(os.path.join(OUT, "manuscript_numbers.csv"), index=False)
with open(os.path.join(OUT, "manuscript_numbers.txt"), "w", encoding="utf-8") as fh:
    cur = None
    for r in rows:
        if r["section"] != cur:
            cur = r["section"]
            fh.write(f"\n=== {cur} ===\n")
        fh.write(f"{r['item']}: {r['value']}\n")

# ------------------------------------------------------------------
# L. FIGURES
# ------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
GRID = "#e6e6e6"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# --- Fig. 1 study area ---
utm = zones.to_crs(32646)
U = {r.zone: r.geometry for r in utm.itertuples()}
isl = U["island_land"]
ring = isl.buffer(6000).difference(isl.buffer(3000))
geo = lambda gm: gpd.GeoSeries([gm], crs=32646).to_crs(4326)
Z = {k: geo(val) for k, val in U.items()}
fig, ax = plt.subplots(figsize=(5.6, 6.2))
for s_ in ax.spines.values():
    s_.set_visible(True)
for z in ["C_east", "C_west"]:
    Z[z].plot(ax=ax, color=COL[z], alpha=0.35, edgecolor=COL[z], linewidth=0.6)
geo(U["ferry_corridor"].intersection(ring)).plot(ax=ax, facecolor="none", edgecolor="0.45", hatch="////", linewidth=0.5)
for z in SEC:
    Z[z].plot(ax=ax, color=COL[z], alpha=0.85, edgecolor="white", linewidth=0.4)
geo(isl.buffer(300).boundary).plot(ax=ax, color="k", linewidth=0.4, linestyle=(0, (2, 2)))
Z["island_land"].plot(ax=ax, color="0.82", edgecolor="0.35", linewidth=0.4)
tb = gpd.GeoSeries([U["C_east"].union(U["C_west"])], crs=32646).to_crs(4326).total_bounds
ax.set_xlim(tb[0] - 0.006, tb[2] + 0.006); ax.set_ylim(tb[1] - 0.006, tb[3] + 0.006)
ax.set_aspect(1 / np.cos(np.deg2rad(20.6)))
ib = Z["island_land"].total_bounds; cx, cy = (ib[0] + ib[2]) / 2, (ib[1] + ib[3]) / 2
for z, xy, c in [("T_north", (cx - 0.030, ib[3] + 0.015), COL["T_north"]), ("T_east", (ib[2] + 0.025, cy + 0.008), "#9a6a00"),
                 ("T_west", (ib[0] - 0.025, cy + 0.008), COL["T_west"]), ("T_south", (cx + 0.025, ib[1] - 0.016), COL["T_south"])]:
    p_ = Z[z].iloc[0].representative_point()
    ax.annotate(z.replace("_", "-"), xy=(p_.x, p_.y), xytext=xy, fontsize=7.5, fontweight="bold", color=c,
                ha="center", va="center", arrowprops=dict(arrowstyle="-", color="0.3", lw=0.5))
ax.text(tb[2] - 0.024, cy - 0.035, "C-east\n(control)", ha="center", fontsize=8, color="#0b4f75", fontweight="bold")
ax.text(tb[0] + 0.024, cy - 0.035, "C-west\n(control)", ha="center", fontsize=8, color="#0b4f75", fontweight="bold")
ax.set_xticks([92.25, 92.30, 92.35, 92.40]); ax.set_xticklabels([f"{x:.2f}°E" for x in [92.25, 92.30, 92.35, 92.40]])
ax.set_yticks([20.55, 20.60, 20.65]); ax.set_yticklabels([f"{y:.2f}°N" for y in [20.55, 20.60, 20.65]])
ax.tick_params(labelsize=7, direction="out", length=3)
km = 1 / (111.32 * np.cos(np.deg2rad(20.6)))
sx, sy = tb[0] + 0.004, tb[1] + 0.002
ax.plot([sx, sx + 2 * km], [sy, sy], color="k", lw=2, solid_capstyle="butt"); ax.text(sx + km, sy + 0.003, "2 km", ha="center", fontsize=7)
nx, ny = tb[2] - 0.002, tb[3] - 0.018
ax.annotate("", xy=(nx, ny + 0.012), xytext=(nx, ny), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.8))
ax.text(nx, ny + 0.0145, "N", ha="center", fontsize=8, fontweight="bold")
handles = [Patch(facecolor="white", edgecolor="0.3", label="Nearshore treatment sectors, 0–1 km (labelled)"),
           Line2D([0], [0], color="k", lw=0.6, ls=(0, (2, 2)), label="300 m from shore (inner edge of deep variants)"),
           Patch(facecolor=COL["C_west"], alpha=0.35, edgecolor=COL["C_west"], label="Offshore control sectors, 3–6 km"),
           Patch(facecolor="none", edgecolor="0.45", hatch="////", label="Ferry corridor (excluded from controls)"),
           Patch(facecolor="0.82", edgecolor="0.35", label="Island (dry-season Sentinel-2 shoreline)")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=6.6, frameon=False,
          handlelength=2.2, columnspacing=1.2)
save(fig, "Fig1_study_area")

# --- Fig. 3 availability and seasonal cycle ---
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
tab = pd.crosstab(tn.date.dt.year, tn.date.dt.month).reindex(columns=range(1, 13), fill_value=0)
im = a.imshow(tab.values, cmap="Blues", aspect="auto", vmin=0, vmax=tab.values.max())
a.set_xticks(range(12)); a.set_xticklabels([m_[0] for m_ in MON]); a.set_yticks(range(len(tab))); a.set_yticklabels(tab.index)
for i in range(tab.shape[0]):
    for j in range(12):
        val = tab.values[i, j]
        if val > 0:
            a.text(j, i, val, ha="center", va="center", fontsize=6.5, color="white" if val > 5 else "#222")
a.set_title("(a) Usable observations, northern sector", fontsize=8, loc="left")
for s_ in a.spines.values():
    s_.set_visible(False)
a.tick_params(length=0)
cb = fig.colorbar(im, ax=a, fraction=0.04, pad=0.02); cb.set_label("Dates per month", fontsize=7); cb.ax.tick_params(labelsize=6.5)
order = [10, 11, 12, 1, 2, 3]; xx = np.arange(len(order))
for z in ["C_east", "C_west"]:
    b.plot(xx, clim.loc[order, z], "--", color=COL[z], lw=1.2, marker="s", ms=3.5, label="Control " + z[2:])
for z in SEC:
    b.plot(xx, clim.loc[order, z], "-", color=COL[z], lw=1.4, marker="o", ms=3.5, label=LAB[z])
b.set_xticks(xx); b.set_xticklabels([MON[m_ - 1] for m_ in order]); b.set_ylabel("Median turbidity (FNU)")
b.set_ylim(0, 14); b.grid(axis="y", color=GRID, lw=0.6); b.set_axisbelow(True)
b.set_title("(b) Pre-restriction seasonal cycle", fontsize=8, loc="left")
b.legend(ncol=2, fontsize=6.5, frameon=False, loc="lower right")
fig.tight_layout(); save(fig, "Fig3_availability_seasonality")

# --- Fig. 4 night lights ---
sd_nl = vp.res.std()
fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.axhspan(100 * (np.exp(-1.96 * sd_nl) - 1), 100 * (np.exp(1.96 * sd_nl) - 1), color="#f0f0f0", zorder=0)
for s0, e0 in [("2025-02-01", "2025-11-01"), ("2026-02-01", "2026-09-01")]:
    ax.axvspan(pd.Timestamp(s0), pd.Timestamp(e0), color="#0072B2", alpha=0.08, lw=0)
ax.axvspan(pd.Timestamp("2020-04-01"), pd.Timestamp("2020-11-01"), color="#999999", alpha=0.12, lw=0)
ax.axhline(0, color="#555", lw=0.6)
ax.plot(vp.month, 100 * (np.exp(vp.res) - 1), "o", ms=3, color="#9a9a9a", mfc="white", mew=0.8, label="Pre-restriction (cross-validated)")
sty = {"COVID": ("#555555", "s", "COVID-19 shutdown 2020"), "Capped": ("#E69F00", "D", "Capped season"),
       "Closed Feb–Apr": ("#0072B2", "o", "Closure, Feb–Apr"), "Closed May–Oct": ("#56B4E9", "o", "Closure, May–Oct")}
for g_, (c, mk, lab) in sty.items():
    q = vpo[vpo.g == g_]
    ax.plot(q.month, 100 * (np.exp(q.res) - 1), mk, ms=4.2, color=c, mec="white", mew=0.5, label=lab)
ax.set_ylabel("Island night lights vs\ncounterfactual (%)"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
top = ax.get_ylim()[1] * 0.92
ax.text(pd.Timestamp("2020-07-15"), top, "COVID-19", ha="center", fontsize=6.5, color="#555")
ax.text(pd.Timestamp("2025-06-15"), top, "Closure", ha="center", fontsize=6.5, color="#0072B2")
ax.text(pd.Timestamp("2026-05-01"), top, "Closure", ha="center", fontsize=6.5, color="#0072B2")
ax.legend(ncol=5, fontsize=6.3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), handletextpad=0.3, columnspacing=1.0)
fig.tight_layout(); save(fig, "Fig4_night_lights")

# --- Fig. 5 counterfactual, northern sector ---
oof = pd.read_csv(P["oof"], parse_dates=["date"]); oof = oof[(oof.model == "M1_Ridge") & (oof.zone == "T_north")]
cfp = pd.read_csv(P["cf"], parse_dates=["date"]); cfp = cfp[(cfp.model == "M1_Ridge") & (cfp.zone == "T_north")]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.9]})
a.loglog(np.exp(oof.pred), np.exp(oof.y), "o", ms=2.6, color="#D55E00", alpha=0.6, mew=0)
lim = [0.5, 60]; a.plot(lim, lim, color="#555", lw=0.7); a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel("Predicted turbidity (FNU)"); a.set_ylabel("Observed turbidity (FNU)")
a.set_title("(a) Cross-validation, north", fontsize=8, loc="left")
a.text(0.6, 40, f"R² = {t5.loc['T_north','M1_Ridge']:.2f} (log scale)\nn = {len(oof)}", fontsize=6.8, va="top")
a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)
allp = pd.concat([oof[oof.date >= pd.Timestamp("2023-10-01")][["date", "y", "pred"]],
                  cfp[cfp.group != "COVID closure 2020"][["date", "y", "pred"]]]).sort_values("date")
for s0, e0 in [("2025-02-01", "2025-11-01"), ("2026-02-01", "2026-09-30")]:
    b.axvspan(pd.Timestamp(s0), pd.Timestamp(e0), color="#0072B2", alpha=0.08, lw=0)
for s0, e0 in [("2024-11-01", "2025-02-01"), ("2025-11-01", "2026-02-01")]:
    b.axvspan(pd.Timestamp(s0), pd.Timestamp(e0), color="#E69F00", alpha=0.10, lw=0)
b.axvline(pd.Timestamp("2024-11-01"), color="#555", lw=0.7, ls="--")
b.semilogy(allp.date, np.exp(allp.pred), "o", ms=3.2, mfc="white", mec="#555", mew=0.7, label="Counterfactual (predicted)")
b.semilogy(allp.date, np.exp(allp.y), "o", ms=2.6, color="#D55E00", mew=0, label="Observed")
for _, r in allp.iterrows():
    b.plot([r.date, r.date], [np.exp(r.pred), np.exp(r.y)], color="#bbbbbb", lw=0.5, zorder=0)
b.set_ylabel("Turbidity (FNU)"); b.set_title("(b) Observed and counterfactual turbidity, north", fontsize=8, loc="left")
b.grid(axis="y", color=GRID, lw=0.5); b.set_axisbelow(True)
yl = b.get_ylim()
b.text(pd.Timestamp("2024-11-06"), yl[1] * 0.55, "Capped", fontsize=6.0, ha="left", color="#9a6a00")
b.text(pd.Timestamp("2025-06-15"), yl[1] * 0.75, "Closure", fontsize=6.3, ha="center", color="#0072B2")
b.text(pd.Timestamp("2025-11-06"), yl[1] * 0.55, "Capped", fontsize=6.0, ha="left", color="#9a6a00")
b.text(pd.Timestamp("2026-05-15"), yl[1] * 0.75, "Closure", fontsize=6.3, ha="center", color="#0072B2")
b.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
b.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); b.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
fig.tight_layout(); save(fig, "Fig5_counterfactual_north")

# --- Fig. 6 effects ---
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
for ax, grp, title in zip(axs, ["Capped (Nov–Jan)", "Closed (Feb–Apr)"], ["(a) Capped season (Nov–Jan)", "(b) Closure (Feb–Apr)"]):
    for i, z in enumerate(SEC):
        r = main[(main.zone == z) & (main.group == grp)].iloc[0]
        ax.plot([r.plac_lo, r.plac_hi], [-i, -i], color="#d9d9d9", lw=7, solid_capstyle="butt", zorder=1)
        ax.plot([r.boot_lo, r.boot_hi], [-i, -i], color="#444", lw=1.4, zorder=2)
        ax.plot(r.effect_pct, -i, "o", ms=6, color=COL[z], mec="white", mew=0.8, zorder=3)
        ax.plot([-r.MDE_reduction] * 2, [-i - 0.28, -i + 0.28], color="#0072B2", lw=1.2, zorder=2)
    ax.axvline(0, color="#555", lw=0.7)
    ax.set_title(title, fontsize=8, loc="left"); ax.set_xlabel("Effect on turbidity (%)")
    ax.set_xlim(-35, 35); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
axs[0].set_yticks([0, -1, -2, -3]); axs[0].set_yticklabels([LAB[z] for z in SEC])
h = [Line2D([0], [0], marker="o", color="none", mfc="#777", mec="white", ms=6, label="Estimate"),
     Line2D([0], [0], color="#444", lw=1.4, label="Bootstrap 95% CI"),
     Line2D([0], [0], color="#d9d9d9", lw=7, label="Placebo-based 95% interval"),
     Line2D([0], [0], color="#0072B2", lw=1.2, label="Minimum detectable reduction")]
fig.legend(handles=h, ncol=4, fontsize=6.6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, "Fig6_effects")

# --- Fig. 7 robustness ---
mk = {"Main analysis": "o", "Valid fraction ≥ 0.10": "^", "Valid fraction ≥ 0.50": "v", "Training on Nov–Apr only": "s",
      "XGBoost": "D", "Deep sectors (300–1000 m)": "P", "Red reflectance (B4)": "X"}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, (short, title) in zip(axs, [("Capped", "(a) Capped season (Nov–Jan)"), ("Closure", "(b) Closure (Feb–Apr)")]):
    for i, z in enumerate(SEC):
        for j, (_, lab) in enumerate(spec_order):
            val = float(tab7.loc[tab7.Specification == lab, f"{short} {LAB[z][0]}"].iloc[0])
            ax.plot(val, -i - (j - 3) * 0.11, mk[lab], ms=4.2 if j else 5.5, color=COL[z],
                    mec="#222" if j == 0 else "white", mew=0.6, alpha=1 if j == 0 else 0.85)
    ax.axvline(0, color="#555", lw=0.7); ax.set_xlim(-12, 20)
    ax.set_title(title, fontsize=8, loc="left"); ax.set_xlabel("Effect on turbidity (%)")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
axs[0].set_yticks([0, -1, -2, -3]); axs[0].set_yticklabels([LAB[z] for z in SEC])
h = [Line2D([0], [0], marker=mk[l], color="none", mfc="#888", mec="white", ms=5, label=l) for _, l in spec_order]
fig.legend(handles=h, ncol=4, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.13, 1, 1)); save(fig, "Fig7_robustness")

print("\nAll outputs saved in:", OUT)
print("Open manuscript_numbers.txt and compare each value with the manuscript.")
