"""
Saint Martin's Island — Step 3.6: Supplementary material

Builds the supplementary tables (Tables S1–S7) and Fig. S1 from the outputs of
Steps 3.2–3.5. Only Table S3 involves new model fits: it lists the effect in
each placebo year, refitting the ridge models of Step 3.3 (main specification)
exactly as Step 3.3 does, and checks that the result reproduces the placebo
intervals in robustness_effects.csv. Ridge regression involves no random
numbers, so the reproduction is exact.

Inputs  : SMI_project/outputs/  (Steps 3.1–3.5)
Outputs : SMI_project/outputs/step3_6/
    TableS1_night_lights_monthly.csv   monthly night-light effects after the restrictions
    TableS2_importance_north.csv       permutation importance, northern sector
    TableS3_placebo_years.csv          effect in each placebo year (turbidity, ridge)
    TableS4_block_bootstrap.csv        bootstrap intervals resampling whole months (v1.1)
    TableS5_monsoon_closure.csv        effects for the monsoon months of the closure
    TableS6_capped_split.csv           capped season split: November vs December–January
    TableS7_injection.csv              known-effect injection test
    TableS8_covid_dates.csv            individual dates of the 2020 COVID-19 shutdown
    FigS1_counterfactual_other_sectors.png (600 dpi) and .pdf
    supplementary_check.txt            consistency checks against the manuscript files

HOW TO RUN IN COLAB
    1. from google.colab import drive; drive.mount('/content/drive')
    2. paste this script into a new cell and run it (about 1 minute).
Version 1.1 (October 2026): adds Table S4 (month-block bootstrap)
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 0. PATHS
# ------------------------------------------------------------------
PROJECT = os.environ.get("SMI_DATA", "/content/drive/MyDrive/SMI_project")
BASE = os.environ.get("SMI_BASE", os.path.join(PROJECT, "outputs"))
OUT = os.path.join(BASE, "step3_6")
os.makedirs(OUT, exist_ok=True)
for _f in os.listdir(OUT):                      # remove tables from earlier versions
    if _f.startswith("TableS") and _f.endswith(".csv"):
        os.remove(os.path.join(OUT, _f))

P = {
    "data": os.path.join(BASE, "SMI_analysis_dataset_v1.csv"),
    "tides": os.path.join(BASE, "step3_2", "tides_GOT410.csv"),
    "cv": os.path.join(BASE, "step3_2", "cv_metrics.csv"),
    "eff": os.path.join(BASE, "step3_2", "effects.csv"),
    "oof": os.path.join(BASE, "step3_2", "oof_predictions.csv"),
    "cf": os.path.join(BASE, "step3_2", "counterfactual_predictions.csv"),
    "rob": os.path.join(BASE, "step3_3", "robustness_effects.csv"),
    "inj": os.path.join(BASE, "step3_3", "injection_test.csv"),
    "imp": os.path.join(BASE, "step3_3", "importance_T_north.csv"),
    "covid": os.path.join(BASE, "step3_3", "covid_dates.csv"),
    "nl_month": os.path.join(BASE, "step3_4", "night_lights_monthly_effects.csv"),
    "split": os.path.join(BASE, "step3_5", "capped_split_nov_decjan.csv"),
}
missing = [f"{k}: {v}" for k, v in P.items() if not os.path.exists(v)]
if missing:
    raise FileNotFoundError("These input files were not found:\n  " + "\n  ".join(missing))

SEC = ["T_north", "T_east", "T_west", "T_south"]
LAB = {"T_north": "North", "T_east": "East", "T_west": "West", "T_south": "South"}
COL = {"T_north": "#D55E00", "T_east": "#E69F00", "T_west": "#CC79A7", "T_south": "#009E73"}
FEATURES = ["log_ctrl", "ctrl_EW", "ws10_24h", "ws10_72h", "u10_24h", "v10_24h", "swh_24h", "sst_C",
            "era5_rain_mm_catch_72h", "chirps_catch_7d", "chirps_catch_30d", "tide_now", "tide_range",
            "tide_trend", "doy_sin", "doy_cos", "sun_zenith"]
GROUPS = {"Capped (Nov–Jan)": [11, 12, 1], "Closed (Feb–Apr)": [2, 3, 4]}
checks = []


def check(label, ok, detail=""):
    line = f"[{'OK' if ok else 'CHECK'}] {label}" + (f" — {detail}" if detail else "")
    checks.append(line)
    print(line)


def pct(x):
    return 100 * (np.exp(x) - 1)


# ------------------------------------------------------------------
# Table S1. Monthly night-light effects after the restrictions (Results 4.2)
# ------------------------------------------------------------------
nl = pd.read_csv(P["nl_month"], parse_dates=["month"])
order = {"COVID 2020": 0, "Capped (Nov–Jan)": 1, "Closed (Feb–Apr)": 2, "Closed (May–Oct)": 3}
s1 = nl[["month", "group", "island_cloudfree_nights", "effect_pct"]].copy()
s1["Month"] = s1.month.dt.strftime("%b %Y")
s1 = s1.sort_values("month")
s1 = s1.rename(columns={"group": "Period", "island_cloudfree_nights": "Mean cloud-free observations",
                        "effect_pct": "Effect (%)"})[["Month", "Period", "Mean cloud-free observations", "Effect (%)"]]
s1.to_csv(os.path.join(OUT, "TableS1_night_lights_monthly.csv"), index=False)
fa = nl[nl.group == "Closed (Feb–Apr)"].effect_pct
check("Table S1: Feb–Apr closure months all negative", bool((fa < 0).all()),
      f"{len(fa)} months, {fa.max():.1f}% to {fa.min():.1f}%")

# ------------------------------------------------------------------
# Table S2. Permutation importance, northern sector (Results 4.3)
# ------------------------------------------------------------------
imp = pd.read_csv(P["imp"], index_col=0).iloc[:, 0].sort_values(ascending=False)
SHORT = {"era5_rain_mm_catch_72h": "era5_rain_72h", "chirps_catch_7d": "chirps_7d",
         "chirps_catch_30d": "chirps_30d"}                 # names as in Table 4 of the article
s2 = pd.DataFrame({"Rank": range(1, len(imp) + 1), "Predictor": [SHORT.get(k, k) for k in imp.index],
                   "Decrease in R²": imp.values})
s2.to_csv(os.path.join(OUT, "TableS2_importance_north.csv"), index=False)
check("Table S2: 17 predictors", len(s2) == 17, f"top: {imp.index[0]} {imp.iloc[0]:.2f}")

# ------------------------------------------------------------------
# Table S3. Effect in each placebo year (Results 4.4) — refit as in Step 3.3
# ------------------------------------------------------------------
raw = pd.read_csv(P["data"], parse_dates=["date"])
raw["overpass"] = pd.to_datetime(raw["date"].dt.strftime("%Y-%m-%d") + " " + raw["time_utc"])
tides = pd.read_csv(P["tides"], parse_dates=["overpass"])
raw = raw.merge(tides, on="overpass", how="left")


def build_table(valid_min=0.30, var="turbidity_median"):
    d = raw.copy()
    d["ok"] = (d.valid_mean >= valid_min) & d[var].notna() & (d[var] > 0)
    d = d[d.ok]
    ctrl = d[d.zone.isin(["C_east", "C_west"])].pivot_table(index="date", columns="zone", values=var).dropna()
    cf = pd.DataFrame({"log_ctrl": np.log(ctrl).mean(axis=1), "ctrl_EW": np.log(ctrl.C_east / ctrl.C_west)})
    d = d[d.zone.isin(SEC)].merge(cf, left_on="date", right_index=True)
    doy = d.date.dt.dayofyear
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    d["y"] = np.log(d[var])
    d["hyear"] = np.where(d.date.dt.month >= 5, d.date.dt.year, d.date.dt.year - 1)
    d["month"] = d.date.dt.month
    return d.dropna(subset=FEATURES + ["y"])


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))


main_tab = build_table(0.30)
rob = pd.read_csv(P["rob"])
rob_main = rob[rob.spec.str.startswith("Main")].set_index(["zone", "group"])
s3_rows = []
for zone in SEC:
    pre = main_tab[(main_tab.zone == zone) & (main_tab.period == "pre")]
    for grp, months in GROUPS.items():
        plac = {}
        for yr in sorted(pre.hyear.unique()):
            te = pre[(pre.hyear == yr) & pre.month.isin(months)]
            if len(te) < 3:
                continue
            m = ridge().fit(pre[pre.hyear != yr][FEATURES], pre[pre.hyear != yr].y)
            plac[yr] = ((te.y - m.predict(te[FEATURES])).mean(), len(te))
        r = rob_main.loc[(zone, grp)]
        est = np.log(1 + r.effect_pct / 100)
        sd = np.std([v[0] for v in plac.values()], ddof=1)
        lo, hi = pct(est - 1.96 * sd), pct(est + 1.96 * sd)
        check(f"Table S3: placebo interval reproduced, {LAB[zone]}, {grp}",
              abs(lo - r.plac_lo) < 0.05 and abs(hi - r.plac_hi) < 0.05 and len(plac) == r.n_placebo_years,
              f"{lo:.1f} to {hi:.1f} (Step 3.3: {r.plac_lo:.1f} to {r.plac_hi:.1f}); {len(plac)} years")
        for yr, (e, n) in plac.items():
            s3_rows.append({"Period": grp, "Sector": LAB[zone], "Placebo year": f"{yr}/{str(yr + 1)[-2:]}",
                            "n": n, "Effect (%)": pct(e)})
        s3_rows.append({"Period": grp, "Sector": LAB[zone], "Placebo year": "Restriction (observed)",
                        "n": int(r.n), "Effect (%)": r.effect_pct})
s3 = pd.DataFrame(s3_rows)
s3.to_csv(os.path.join(OUT, "TableS3_placebo_years.csv"), index=False)

# ------------------------------------------------------------------
# Table S5. Monsoon months of the closure (Results 4.4)
# ------------------------------------------------------------------
eff = pd.read_csv(P["eff"])
s4 = eff[(eff.model == "M1_Ridge") & eff.zone.isin(SEC) & (eff.group == "Closed (May–Oct)")].copy()
s4["Sector"] = s4.zone.map(LAB)
s4 = s4.set_index("zone").reindex(SEC)
s4 = s4.rename(columns={"n_obs": "n", "effect_pct": "Effect (%)", "ci_low_pct": "Bootstrap 95% CI, lower",
                        "ci_high_pct": "Bootstrap 95% CI, upper"})
s4 = s4[["Sector", "n", "Effect (%)", "Bootstrap 95% CI, lower", "Bootstrap 95% CI, upper"]]
s4.to_csv(os.path.join(OUT, "TableS5_monsoon_closure.csv"), index=False)
check("Table S5: monsoon closure n per sector", True, ", ".join(f"{r.Sector} {int(r.n)}" for _, r in s4.iterrows()))

# ------------------------------------------------------------------
# Table S6. Capped season split (Results 4.5)
# ------------------------------------------------------------------
sp = pd.read_csv(P["split"])
sp["Sector"] = sp.zone.map(LAB)
s5 = sp.rename(columns={"part": "Part", "effect_pct": "Effect (%)", "placebo_p": "Placebo p",
                        "placebo_years": "Placebo years"})[["Part", "Sector", "n", "Effect (%)", "Placebo p", "Placebo years"]]
s5["_p"] = s5.Part.map({"November": 0, "December–January": 1})
s5["_s"] = sp.zone.map({z: i for i, z in enumerate(SEC)})
s5 = s5.sort_values(["_p", "_s"]).drop(columns=["_p", "_s"])
s5.to_csv(os.path.join(OUT, "TableS6_capped_split.csv"), index=False)

# ------------------------------------------------------------------
# Table S7. Known-effect injection (Results 4.5)
# ------------------------------------------------------------------
inj = pd.read_csv(P["inj"])
s6 = inj.pivot_table(index=["zone", "pseudo_season"], columns="true_effect_pct", values="estimated_pct").reset_index()
s6 = s6.rename(columns={0.0: "Estimate, true 0%", -10.0: "Estimate, true −10%", -20.0: "Estimate, true −20%"})
s6["Sector"] = s6.zone.map(LAB)
s6["_o"] = s6.zone.map({z: i for i, z in enumerate(SEC)})
s6 = s6.sort_values(["_o", "pseudo_season"])
s6["Pseudo-restriction season"] = s6.pseudo_season.str.replace(r"/20(\d\d)$", r"/\1", regex=True)   # 2021/2022 -> 2021/22
s6 = s6[["Sector", "Pseudo-restriction season", "Estimate, true 0%", "Estimate, true −10%", "Estimate, true −20%"]]
s6.to_csv(os.path.join(OUT, "TableS7_injection.csv"), index=False)
mean0 = inj[inj.true_effect_pct == 0].estimated_pct
check("Table S7: injection, true 0%", len(mean0) == 12, f"mean {mean0.mean():.1f}%, SD {mean0.std():.1f}, n {len(mean0)}")

# ------------------------------------------------------------------
# Table S8. COVID-19 shutdown dates (Results 4.6)
# ------------------------------------------------------------------
cv19 = pd.read_csv(P["covid"], parse_dates=["date"])
cv19["Sector"] = cv19.zone.map(LAB)
cv19["_o"] = cv19.zone.map({z: i for i, z in enumerate(SEC)})
s7 = cv19.sort_values(["date", "_o"]).copy()
s7["Date"] = s7.date.dt.strftime("%d %b %Y")
s7 = s7.rename(columns={"control_ref": "Control (FNU)", "observed": "Observed (FNU)",
                        "counterfactual": "Counterfactual (FNU)", "effect_pct": "Effect (%)"})
s7 = s7[["Date", "Sector", "Control (FNU)", "Observed (FNU)", "Counterfactual (FNU)", "Effect (%)"]]
s7.to_csv(os.path.join(OUT, "TableS8_covid_dates.csv"), index=False)
check("Table S8: sector–dates below counterfactual", True, f"{(cv19.effect_pct < 0).sum()} of {len(cv19)}")

# ------------------------------------------------------------------
# Table S4. Month-block bootstrap (Section 4.4) — allows for serial
# correlation of residuals within a month by resampling whole calendar months
# ------------------------------------------------------------------
cfb = pd.read_csv(P["cf"], parse_dates=["date"])
cfb = cfb[(cfb.model == "M1_Ridge") & cfb.zone.isin(SEC)]
RNG = np.random.default_rng(42)
s8_rows = []
for zone in SEC:
    for grp in GROUPS:
        q = cfb[(cfb.zone == zone) & (cfb.group == grp)].sort_values("date")
        res = (q.y - q.pred).values
        month = q.date.dt.to_period("M").values
        blocks = [res[month == m] for m in sorted(set(month))]
        est = res.mean()
        r = rob_main.loc[(zone, grp)]
        check(f"Table S4: effect matches Table 6, {LAB[zone]}, {grp}",
              abs(pct(est) - r.effect_pct) < 0.01 and len(res) == r.n, f"{pct(est):.2f}% (n {len(res)})")
        boot = np.array([np.concatenate([blocks[i] for i in RNG.integers(0, len(blocks), len(blocks))]).mean()
                         for _ in range(1000)])
        s8_rows.append({"Period": grp, "Sector": LAB[zone], "n": len(res), "Months": len(blocks),
                        "Effect (%)": pct(est),
                        "Date bootstrap, lower": r.boot_lo, "Date bootstrap, upper": r.boot_hi,
                        "Month-block bootstrap, lower": pct(np.percentile(boot, 2.5)),
                        "Month-block bootstrap, upper": pct(np.percentile(boot, 97.5)),
                        "Placebo interval, lower": r.plac_lo, "Placebo interval, upper": r.plac_hi})
s8 = pd.DataFrame(s8_rows)
s8.to_csv(os.path.join(OUT, "TableS4_block_bootstrap.csv"), index=False)

# ------------------------------------------------------------------
# Fig. S1. Counterfactuals for the eastern, western and southern sectors
# ------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
GRID = "#e6e6e6"
cvm = pd.read_csv(P["cv"]).set_index(["zone", "model"])
oof_all = pd.read_csv(P["oof"], parse_dates=["date"])
cf_all = pd.read_csv(P["cf"], parse_dates=["date"])
other = ["T_east", "T_west", "T_south"]
fig, axs = plt.subplots(3, 2, figsize=(7.2, 8.4), gridspec_kw={"width_ratios": [1, 1.9]})
letters = iter("abcdef")
for row, z in enumerate(other):
    a, b = axs[row]
    oof = oof_all[(oof_all.model == "M1_Ridge") & (oof_all.zone == z)]
    cfp = cf_all[(cf_all.model == "M1_Ridge") & (cf_all.zone == z)]
    a.loglog(np.exp(oof.pred), np.exp(oof.y), "o", ms=2.6, color=COL[z], alpha=0.6, mew=0)
    lim = [0.5, 250]; a.plot(lim, lim, color="#555", lw=0.7); a.set_xlim(lim); a.set_ylim(lim)   # all points shown
    a.set_xlabel("Predicted turbidity (FNU)"); a.set_ylabel("Observed turbidity (FNU)")
    a.set_title(f"({next(letters)}) Cross-validation, {LAB[z].lower()}", fontsize=8, loc="left")
    a.text(0.6, 150, f"R² = {cvm.loc[(z, 'M1_Ridge'), 'R2']:.2f} (log scale)\nn = {len(oof)}", fontsize=6.8, va="top")
    a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)
    allp = pd.concat([oof[oof.date >= pd.Timestamp("2023-10-01")][["date", "y", "pred"]],
                      cfp[cfp.group != "COVID closure 2020"][["date", "y", "pred"]]]).sort_values("date")
    for s0, e0 in [("2025-02-01", "2025-11-01"), ("2026-02-01", "2026-09-30")]:
        b.axvspan(pd.Timestamp(s0), pd.Timestamp(e0), color="#0072B2", alpha=0.08, lw=0)
    for s0, e0 in [("2024-11-01", "2025-02-01"), ("2025-11-01", "2026-02-01")]:
        b.axvspan(pd.Timestamp(s0), pd.Timestamp(e0), color="#E69F00", alpha=0.10, lw=0)
    b.axvline(pd.Timestamp("2024-11-01"), color="#555", lw=0.7, ls="--")
    b.semilogy(allp.date, np.exp(allp.pred), "o", ms=3.2, mfc="white", mec="#555", mew=0.7, label="Counterfactual (predicted)")
    b.semilogy(allp.date, np.exp(allp.y), "o", ms=2.6, color=COL[z], mew=0, label="Observed")
    for _, r in allp.iterrows():
        b.plot([r.date, r.date], [np.exp(r.pred), np.exp(r.y)], color="#bbbbbb", lw=0.5, zorder=0)
    b.set_ylim(0.5, 250); b.set_ylabel("Turbidity (FNU)")
    b.set_title(f"({next(letters)}) Observed and counterfactual turbidity, {LAB[z].lower()}", fontsize=8, loc="left")
    b.grid(axis="y", color=GRID, lw=0.5); b.set_axisbelow(True)
    b.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); b.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    if row == 0:
        yl = b.get_ylim()
        b.text(pd.Timestamp("2024-11-06"), yl[1] * 0.55, "Capped", fontsize=6.0, ha="left", color="#9a6a00")
        b.text(pd.Timestamp("2025-06-15"), yl[1] * 0.75, "Closure", fontsize=6.3, ha="center", color="#0072B2")
        b.text(pd.Timestamp("2025-11-06"), yl[1] * 0.55, "Capped", fontsize=6.0, ha="left", color="#9a6a00")
        b.text(pd.Timestamp("2026-05-15"), yl[1] * 0.75, "Closure", fontsize=6.3, ha="center", color="#0072B2")
    if row == 2:
        from matplotlib.lines import Line2D
        hd = [Line2D([], [], ls="none", marker="o", ms=3.2, mfc="white", mec="#555", mew=0.7),
              Line2D([], [], ls="none", marker="o", ms=2.6, color="#777", mew=0)]
        b.legend(hd, ["Counterfactual (predicted)", "Observed (colour as in panels a, c, e)"], fontsize=6.5,
                 frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.3))
fig.tight_layout()
for ext, kw in [("png", dict(dpi=600)), ("pdf", {})]:
    fig.savefig(os.path.join(OUT, f"FigS1_counterfactual_other_sectors.{ext}"), bbox_inches="tight", pad_inches=0.05, **kw)
plt.close(fig)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
with open(os.path.join(OUT, "supplementary_check.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(checks) + "\n")
n_bad = sum(c.startswith("[CHECK]") for c in checks)
print(f"\nSaved in {OUT}: Tables S1–S8 (CSV) and Fig. S1 (PNG, PDF).")
print("All checks passed." if n_bad == 0 else f"{n_bad} check(s) need attention — see supplementary_check.txt")

try:
    from IPython.display import Image, display
    display(Image(filename=os.path.join(OUT, "FigS1_counterfactual_other_sectors.png"), width=800))
except Exception:
    pass
