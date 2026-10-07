"""
Saint Martin's Island — Phase 3, Step 3.1
Quality control, merging and exploratory figures

Inputs  (raw exports from Google Earth Engine, never edited):
    SMI_turbidity_timeseries_v2.csv   (Step 2.3)
    SMI_covariates_v1.csv             (Step 2.4)
Outputs:
    SMI_analysis_dataset_v1.csv       analysis-ready table (date x zone)
    fig1_availability.png ... fig4_ratio_by_period.png

COLAB VERSION: reads the CSVs directly from Google Drive (folder
SMI_project, where Earth Engine exported them) and saves all outputs to
SMI_project/outputs. Run the Drive-mount cell first.
Version 1.0-colab (September 2026)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

DATA_DIR = "/content/drive/MyDrive/SMI_project"
OUT_DIR = "/content/drive/MyDrive/SMI_project/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

VALID_MIN = 0.30                         # min fraction of zone with valid pixels
RESTRICTION_START = pd.Timestamp("2024-11-01")
COVID = (pd.Timestamp("2020-03-26"), pd.Timestamp("2020-11-01"))
T_ZONES = ["T_north", "T_east", "T_west", "T_south"]
C_ZONES = ["C_east", "C_west"]

# ------------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------------
turb = pd.read_csv(os.path.join(DATA_DIR, "SMI_turbidity_timeseries_v2.csv"), parse_dates=["date"])
cov = pd.read_csv(os.path.join(DATA_DIR, "SMI_covariates_v1.csv"), parse_dates=["date"])

# ------------------------------------------------------------------
# 2. Covariates: units and small numerical artefacts
# ------------------------------------------------------------------
cov["sst_C"] = cov["sst_24h"] - 273.15
for col in ["tp_local_24h", "tp_catch_24h", "tp_catch_72h", "tp_catch_168h"]:
    cov[col.replace("tp_", "era5_rain_mm_")] = (cov[col].clip(lower=0) * 1000)   # m -> mm, negatives = 0
cov["wind_dir_from_deg"] = (np.degrees(np.arctan2(-cov["u10_24h"], -cov["v10_24h"])) + 360) % 360
cov = cov.drop(columns=["time_utc", "sst_24h", "tp_local_24h", "tp_catch_24h", "tp_catch_72h", "tp_catch_168h"])

# ------------------------------------------------------------------
# 3. Turbidity quality flag
# ------------------------------------------------------------------
turb["ok"] = (turb["valid_mean"] >= VALID_MIN) & turb["turbidity_median"].notna()


def period(d):
    if d >= RESTRICTION_START:
        return "capped" if d.month in (11, 12, 1) else "closed"
    if COVID[0] <= d < COVID[1]:
        return "covid"
    return "pre"


def season(m):
    if m in (11, 12, 1):
        return "Nov-Jan"
    if m in (2, 3, 4):
        return "Feb-Apr"
    return "May-Oct"


turb["period"] = turb["date"].map(period)
turb["season"] = turb["date"].dt.month.map(season)

# Same-day control reference (geometric mean of the two control sectors)
ctrl = (turb[turb.ok & turb.zone.isin(C_ZONES)]
        .pivot_table(index="date", columns="zone", values="turbidity_median"))
ctrl = ctrl.dropna()
ref = np.exp(np.log(ctrl).mean(axis=1)).rename("control_ref")
turb = turb.merge(ref, left_on="date", right_index=True, how="left")
turb["log_ratio"] = np.log(turb["turbidity_median"] / turb["control_ref"])

data = turb.merge(cov, on="date", how="left")
data.to_csv(os.path.join(OUT_DIR, "SMI_analysis_dataset_v1.csv"), index=False)

ok = data[data.ok]
print("Rows:", len(data), "| usable rows:", len(ok), f"({len(ok)/len(data):.0%})")

# ------------------------------------------------------------------
# 4. Figures
# ------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Fig 1: data availability (T_north)
a = ok[ok.zone == "T_north"]
tab = pd.crosstab(a.date.dt.year, a.date.dt.month).reindex(columns=range(1, 13), fill_value=0)
fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(tab.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(12)); ax.set_xticklabels(MONTHS)
ax.set_yticks(range(len(tab))); ax.set_yticklabels(tab.index)
for i in range(tab.shape[0]):
    for j in range(12):
        v = tab.values[i, j]
        ax.text(j, i, v, ha="center", va="center", fontsize=8, color="white" if v > 4 else "black")
ax.set_title(f"Usable Sentinel-2 observations per month (T_north, valid ≥ {VALID_MIN:.0%})")
fig.colorbar(im, ax=ax, label="observations")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig1_availability.png")); plt.close(fig)

# Fig 2: seasonal cycle before restrictions
pre = ok[ok.period == "pre"]
clim = pre.pivot_table(index=pre.date.dt.month, columns="zone", values="turbidity_median", aggfunc="median")
counts = pre[pre.zone == "T_north"].groupby(pre.date.dt.month).size()
months_ok = counts[counts >= 10].index
fig, ax = plt.subplots(figsize=(8, 4))
styles = {"T_north": ("#d7301f", "-"), "T_east": ("#fc8d59", "-"), "T_west": ("#7b3294", "-"),
          "T_south": ("#1a9850", "-"), "C_east": ("#2c7bb6", "--"), "C_west": ("#00a6ca", "--")}
order = [m for m in [11, 12, 1, 2, 3, 4] if m in months_ok]
x = range(len(order))
for z, (col, ls) in styles.items():
    ax.plot(x, clim.loc[order, z].values, ls, color=col, marker="o", label=z)
ax.set_xticks(list(x)); ax.set_xticklabels([MONTHS[m - 1] for m in order])
ax.set_ylabel("Median turbidity (FNU)")
ax.set_title("Seasonal cycle before restrictions (2017–Oct 2024; months with ≥10 observations)")
ax.legend(ncol=3, fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig2_seasonal_cycle.png")); plt.close(fig)

# Fig 3: time series T_north vs control reference
tn = ok[ok.zone == "T_north"].set_index("date").sort_index()
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].semilogy(tn.index, tn.turbidity_median, ".", color="#d7301f", ms=4, label="T_north")
axes[0].semilogy(tn.index, tn.control_ref, ".", color="#2c7bb6", ms=4, label="Control reference")
axes[0].set_ylabel("Turbidity (FNU, log)"); axes[0].legend(fontsize=8)
axes[1].plot(tn.index, np.exp(tn.log_ratio), ".", color="k", ms=4)
axes[1].axhline(1, color="grey", lw=0.8)
axes[1].set_ylabel("T_north / control (same day)")
for ax in axes:
    ax.axvspan(*COVID, color="orange", alpha=0.2)
    for y0, y1 in [("2025-02-01", "2025-11-01"), ("2026-02-01", "2026-10-01")]:
        ax.axvspan(pd.Timestamp(y0), pd.Timestamp(y1), color="green", alpha=0.15)
    ax.axvline(RESTRICTION_START, color="green", lw=1, ls="--")
    ax.grid(alpha=0.3)
axes[0].set_title("T_north and offshore control (orange = COVID closure, green = tourism ban; dashed = restrictions begin)")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig3_timeseries_T_north.png")); plt.close(fig)

# Fig 4: same-day ratio by period (exploratory only — NOT the causal estimate)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (s, post) in zip(axes, [("Feb-Apr", "closed"), ("Nov-Jan", "capped")]):
    d = ok[(ok.season == s) & ok.zone.isin(T_ZONES) & ok.period.isin(["pre", post])]
    pos, labels, vals = [], [], []
    for i, z in enumerate(T_ZONES):
        for j, p in enumerate(["pre", post]):
            v = np.exp(d[(d.zone == z) & (d.period == p)].log_ratio.dropna())
            vals.append(v); pos.append(i * 3 + j); labels.append(f"{z.replace('T_', '')}\n{p}")
    bp = ax.boxplot(vals, positions=pos, widths=0.8, patch_artist=True, showfliers=False)
    for k, b in enumerate(bp["boxes"]):
        b.set_facecolor("#bdbdbd" if k % 2 == 0 else "#74c476")
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(1, color="grey", lw=0.8)
    ax.set_title(f"{s}: pre (grey) vs {post} (green)")
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("Treatment / control turbidity (same day)")
fig.suptitle("Exploratory comparison — not adjusted for weather", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig4_ratio_by_period.png")); plt.close(fig)

print("Done. Files saved in:", OUT_DIR)

# Show the four figures here in the notebook
for f in ["fig1_availability.png", "fig2_seasonal_cycle.png",
          "fig3_timeseries_T_north.png", "fig4_ratio_by_period.png"]:
    display(Image(filename=os.path.join(OUT_DIR, f)))
