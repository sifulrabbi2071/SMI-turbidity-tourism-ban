"""
Saint Martin's Island — Fig. 2 (methodological workflow)

Draws the workflow diagram of the manuscript. The numbers shown in the boxes
(acquisition dates, study period, cross-validated R² of ridge regression) are
read from your own analysis files; only the scene count (615), printed by the
Earth Engine script of Step 2.3, is typed below.

Inputs  : SMI_project/outputs/SMI_analysis_dataset_v1.csv   (Step 3.1)
          SMI_project/outputs/step3_2/cv_metrics.csv         (Step 3.2)
Outputs : SMI_project/outputs/step3_5/Fig2_workflow.png (600 dpi) and .pdf

HOW TO RUN IN COLAB
    1. from google.colab import drive; drive.mount('/content/drive')
    2. paste this script into a new cell and run it.
Version 1.0 (October 2026)
"""
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.environ.get("SMI_BASE", "/content/drive/MyDrive/SMI_project/outputs")
OUT = os.path.join(BASE, "step3_5")
os.makedirs(OUT, exist_ok=True)

N_SCENES = 615        # from the Earth Engine console, Step 2.3 ("Number of Sentinel-2 scenes")

# ------------------------------------------------------------------
# Numbers from the analysis files
# ------------------------------------------------------------------
d = pd.read_csv(os.path.join(BASE, "SMI_analysis_dataset_v1.csv"), parse_dates=["date"])
n_dates = d["date"].nunique()
first, last = d["date"].min(), d["date"].max()
cv = pd.read_csv(os.path.join(BASE, "step3_2", "cv_metrics.csv"))
r2 = cv[(cv.model == "M1_Ridge") & cv.zone.isin(["T_north", "T_east", "T_west", "T_south"])]["R2"]
print(f"Scenes {N_SCENES}, dates {n_dates}, {first:%b %Y} – {last:%b %Y}, ridge R² {r2.min():.2f}–{r2.max():.2f}")

# ------------------------------------------------------------------
# Drawing (coordinates in layout units; y increases downwards)
# ------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans"})
W, H = 760, 612
fig = plt.figure(figsize=(7.0, 7.0 * (H - 30) / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(20, W - 12); ax.set_ylim(H, 40); ax.axis("off")
EDGE, INK, QUIET, ACC = "#555555", "#111111", "#333333", "#0072B2"


def box(x, y, w, h, name, lines, accent=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                                fc="#e6f1f8" if accent else "white", ec=ACC if accent else EDGE,
                                lw=1.4 if accent else 0.8))
    ax.text(x + w / 2, y + 22, name, ha="center", va="baseline", fontsize=8.6, fontweight="bold", color=INK)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + 38 + 16 * i, line, ha="center", va="baseline", fontsize=7.4, color=QUIET)


def path(pts, arrow=True):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.plot(xs, ys, color=EDGE, lw=0.8, solid_capstyle="butt")
    if arrow:
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=8,
                                     color=EDGE, lw=0.8, shrinkA=0, shrinkB=0))


# Row 1: data sources
box(36, 56, 160, 72, "Sentinel-2 MSI L2A", [f"{N_SCENES} scenes, {n_dates} dates", f"{first:%b %Y} – {last:%b %Y}"])
box(212, 56, 160, 72, "ERA5 and CHIRPS", ["wind, waves, SST", "Naf catchment rain"])
box(388, 56, 160, 72, "GOT4.10 tides", ["height at overpass", "24-h range, trend"])
box(564, 56, 160, 72, "VIIRS night lights", ["monthly composites", "island and Teknaf"])
# Row 2: zones, turbidity, predictors
box(36, 172, 216, 88, "Study zones", ["MNDWI shoreline (dry season)", "0–1 km: N, E, W, S sectors", "3–6 km control: E, W"])
box(272, 172, 216, 88, "Zonal turbidity", ["Cloud Score+ ≥ 0.60, MNDWI > 0", "Dogliotti et al. (2015)", "median; valid ≥ 30% of zone"])
box(508, 172, 216, 88, "Predictors (17)", ["same-day control turbidity", "ERA5 wind, waves, SST, rain", "tides, season, sun zenith"])
# Row 3: modelling
box(36, 300, 216, 88, "Train: pre-restriction", ["Jan 2017 – Oct 2024", "COVID-2020 window excluded", "4 models per treatment sector"])
box(272, 300, 216, 88, "Leave-one-year-out CV", ["May–Apr years held out", "ridge regression selected", f"R² {r2.min():.2f}–{r2.max():.2f} (log scale)"])
box(508, 300, 216, 88, "Predict counterfactual", [f"Nov 2024 – {last:%b %Y}", "effect = observed − predicted", "log scale, reported as %"])
# Row 4: evaluation
box(36, 428, 216, 104, "Uncertainty", ["bootstrap 95% CI (dates)", "placebo: same months in", "each pre-restriction year", "minimum detectable effect"])
box(272, 428, 216, 104, "Robustness checks", ["pixel QC at 10% and 50%", "Nov–Apr training window", "XGBoost; 300–1000 m zones", "red reflectance; injection"])
box(508, 428, 216, 104, "Manipulation check", ["observed island lights", "vs counterfactual from", "Teknaf, trend and month"])
# Result
box(36, 556, 688, 48, "Restriction effects on nearshore turbidity",
    ["capped season (Nov–Jan) and closure (Feb–Apr), each judged against placebo years"], accent=True)

# Connectors
path([(116, 128), (116, 172)])
path([(176, 128), (176, 146), (330, 146), (330, 172)])
path([(352, 128), (352, 150), (540, 150)], arrow=False)
path([(540, 128), (540, 172)])
path([(252, 216), (272, 216)]); path([(488, 216), (508, 216)])
path([(380, 260), (380, 280)], arrow=False)
path([(616, 260), (616, 280), (144, 280), (144, 300)])
path([(252, 344), (272, 344)]); path([(488, 344), (508, 344)])
path([(616, 388), (616, 408), (144, 408), (144, 428)])
path([(380, 408), (380, 428)])
path([(724, 92), (742, 92), (742, 480), (724, 480)])
for x in (144, 380, 616):
    path([(x, 532), (x, 556)])

fig.savefig(os.path.join(OUT, "Fig2_workflow.png"), dpi=600)
fig.savefig(os.path.join(OUT, "Fig2_workflow.pdf"))
plt.close(fig)
print("Saved:", os.path.join(OUT, "Fig2_workflow.png"))

try:
    from IPython.display import Image, display
    display(Image(filename=os.path.join(OUT, "Fig2_workflow.png"), width=800))
except Exception:
    pass
