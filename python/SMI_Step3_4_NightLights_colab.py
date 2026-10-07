"""
Saint Martin's Island — Step 3.4
Night lights (VIIRS) as a "manipulation check": did human activity on the
island actually fall during the tourism restrictions?

Model (trained on pre-restriction months only, COVID 2020 excluded):
  log(island lights) ~ log(Teknaf lights) + linear trend + calendar month
Effect = observed − counterfactual (log scale -> %), placebo test with
leave-one-year-out residuals of the same months in pre years.
Months with < 3 cloud-free nights are excluded (monsoon gaps).

Input : SMI_project/SMI_night_lights_v1.csv  (GEE Step 2.6)
Output: SMI_project/outputs/step3_4/...
Version 1.0 (October 2026)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DATA = os.environ.get("SMI_DATA", "/content/drive/MyDrive/SMI_project")
OUT = os.path.join(os.environ.get("SMI_BASE", "/content/drive/MyDrive/SMI_project/outputs"), "step3_4")
os.makedirs(OUT, exist_ok=True)
MIN_NIGHTS = 3

v = pd.read_csv(os.path.join(DATA, "SMI_night_lights_v1.csv"), parse_dates=["month"])
v = v[(v.island_cloudfree_nights >= MIN_NIGHTS) & (v.island_light_sum > 0) & (v.teknaf_light_sum > 0)].copy()
v["m"] = v.month.dt.month
v["t"] = (v.month - pd.Timestamp("2017-01-01")).dt.days / 365.25
v["y"], v["x"] = np.log(v.island_light_sum), np.log(v.teknaf_light_sum)
v["hyear"] = np.where(v.m >= 5, v.month.dt.year, v.month.dt.year - 1)


def group(r):
    d = r.month
    if d >= pd.Timestamp("2024-11-01"):
        if r.m in (11, 12, 1):
            return "Capped (Nov–Jan)"
        return "Closed (Feb–Apr)" if r.m in (2, 3, 4) else "Closed (May–Oct)"
    if pd.Timestamp("2020-04-01") <= d < pd.Timestamp("2020-11-01"):
        return "COVID 2020"
    return "pre"


v["group"] = v.apply(group, axis=1)
MONTHS = {"Capped (Nov–Jan)": [11, 12, 1], "Closed (Feb–Apr)": [2, 3, 4],
          "Closed (May–Oct)": [5, 6, 7, 8, 9, 10], "COVID 2020": [4, 5, 6, 7, 8, 9, 10]}


def X(d):
    D = pd.get_dummies(d.m.astype(str), prefix="m").reindex(
        columns=[f"m_{i}" for i in range(2, 13)], fill_value=0)
    return pd.concat([d[["x", "t"]], D], axis=1).astype(float)


pre = v[v.group == "pre"].copy()
oof = pd.Series(index=pre.index, dtype=float)
for yr in pre.hyear.unique():
    tr, te = pre[pre.hyear != yr], pre[pre.hyear == yr]
    oof[te.index] = te.y - LinearRegression().fit(X(tr), tr.y).predict(X(te))
pre["res"] = oof
print("Pre-restriction months:", len(pre), "| leave-one-year-out R² =", round(r2_score(pre.y, pre.y - oof), 2))

model = LinearRegression().fit(X(pre), pre.y)
post = v[v.group != "pre"].copy()
post["pred"] = model.predict(X(post))
post["res"] = post.y - post.pred
post["effect_pct"] = 100 * (np.exp(post.res) - 1)

rows = []
for g, d in post.groupby("group"):
    est = d.res.mean()
    plac = pre[pre.m.isin(MONTHS[g])].groupby("hyear").res.mean()
    sd = plac.std(ddof=1)
    rows.append({"group": g, "n_months": len(d), "effect_pct": 100 * (np.exp(est) - 1),
                 "placebo_lo_pct": 100 * (np.exp(est - 1.96 * sd) - 1),
                 "placebo_hi_pct": 100 * (np.exp(est + 1.96 * sd) - 1),
                 "placebo_p": (np.sum(np.abs(plac) >= abs(est)) + 1) / (len(plac) + 1),
                 "months_negative": f"{(d.res < 0).sum()}/{len(d)}"})
eff = pd.DataFrame(rows)
eff.to_csv(os.path.join(OUT, "night_lights_effects.csv"), index=False)
post.to_csv(os.path.join(OUT, "night_lights_monthly_effects.csv"), index=False)
print("\nEffect on island night lights (%):")
print(eff.round(2).to_string(index=False))
print("\nMonth by month:")
print(post[["month", "group", "island_light_sum", "effect_pct"]].round(1).to_string(index=False))

# Figure: residuals over time
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.axhspan(100 * (np.exp(-1.96 * pre.res.std()) - 1), 100 * (np.exp(1.96 * pre.res.std()) - 1),
           color="grey", alpha=0.15, label="95% range of pre-restriction errors")
ax.plot(pre.month, 100 * (np.exp(pre.res) - 1), "o", color="grey", ms=3, label="Pre-restriction (out-of-sample)")
col = {"Capped (Nov–Jan)": "#fdae61", "Closed (Feb–Apr)": "#d7191c", "Closed (May–Oct)": "#7b3294", "COVID 2020": "#2c7bb6"}
for g, d in post.groupby("group"):
    ax.plot(d.month, d.effect_pct, "o", color=col[g], ms=5, label=g)
ax.axhline(0, color="k", lw=0.7)
ax.axvline(pd.Timestamp("2024-11-01"), color="green", ls="--", lw=1)
ax.set_ylabel("Night lights vs counterfactual (%)")
ax.set_title("Saint Martin's Island night lights relative to counterfactual (Teknaf-, trend- and season-adjusted)")
ax.legend(fontsize=7, ncol=3, loc="lower left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figF_night_lights.png")); plt.close(fig)
print("\nSaved in:", OUT)
try:
    from IPython.display import Image, display
    display(Image(filename=os.path.join(OUT, "figF_night_lights.png")))
except Exception:
    pass
