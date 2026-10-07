"""
Saint Martin's Island — Graphical abstract (Marine Pollution Bulletin)

Draws the four-panel graphical abstract directly from the analysis outputs,
so every number in the picture matches the manuscript.

Inputs (Google Drive, folder SMI_project):
    SMI_zones_v2_geojson.geojson                      (Step 2.2 export)
    outputs/SMI_analysis_dataset_v1.csv               (Step 3.1)
    outputs/step3_3/robustness_effects.csv            (Step 3.3)
    outputs/step3_4/night_lights_effects.csv          (Step 3.4)
Outputs (Google Drive, folder SMI_project/outputs/figures):
    Graphical_abstract.png / .tif / .pdf   (3984 x 1593 px, 300 dpi)

HOW TO RUN IN COLAB
    1. Run the first cell:  from google.colab import drive; drive.mount('/content/drive')
    2. Paste this whole script into a second cell and run it.
    3. The finished files appear in SMI_project/outputs/figures.
Version 1.0 (October 2026)
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

# ------------------------------------------------------------------
# 0. PATHS — change only if your folders are named differently
# ------------------------------------------------------------------
PROJECT = os.environ.get("SMI_DATA", "/content/drive/MyDrive/SMI_project")
BASE = os.environ.get("SMI_BASE", os.path.join(PROJECT, "outputs"))
ZONES = os.path.join(PROJECT, "SMI_zones_v2_geojson.geojson")
DATASET = os.path.join(BASE, "SMI_analysis_dataset_v1.csv")
ROBUST = os.path.join(BASE, "step3_3", "robustness_effects.csv")
LIGHTS = os.path.join(BASE, "step3_4", "night_lights_effects.csv")
OUT = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)

for f in [ZONES, DATASET, ROBUST, LIGHTS]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Not found: {f}\nCheck the file name and folder in Google Drive.")

# ------------------------------------------------------------------
# 1. NUMBERS FROM THE ANALYSIS (nothing typed by hand)
# ------------------------------------------------------------------
SECTORS = ["T_north", "T_east", "T_west", "T_south"]
LABELS = ["North", "East", "West", "South"]

d = pd.read_csv(DATASET, parse_dates=["date"])
n_dates = d["date"].nunique()
y0, y1 = d["date"].dt.year.min(), d["date"].dt.year.max()
pre = d[d["ok"] & (d["period"] == "pre") & d["zone"].isin(SECTORS)]
ratio = pre.groupby("zone")["log_ratio"].median().apply(np.exp)        # nearshore / offshore
excess_lo, excess_hi = 100 * (ratio.min() - 1), 100 * (ratio.max() - 1)

rb = pd.read_csv(ROBUST)
main = rb[rb["spec"].str.startswith("Main") & (rb["group"] == "Closed (Feb–Apr)")]

nl = pd.read_csv(LIGHTS)
lights_pct = float(nl.loc[nl["group"] == "Closed (Feb–Apr)", "effect_pct"].iloc[0])

print(f"Dates: {n_dates} ({y0}–{y1}) | night lights Feb–Apr: {lights_pct:.1f}% | "
      f"nearshore excess: {excess_lo:.0f}–{excess_hi:.0f}%")
print(main[["zone", "effect_pct", "plac_lo", "plac_hi"]].round(1).to_string(index=False))

# ------------------------------------------------------------------
# 2. STYLE
# ------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans"})
W, H = 13.28, 5.31                      # inches -> 3984 x 1593 px at 300 dpi
INK, Q, BLUE, ORANGE = "#1a1a1a", "#444444", "#0072B2", "#D55E00"
COL = {"T_north": "#D55E00", "T_east": "#E69F00", "T_west": "#CC79A7", "T_south": "#009E73"}

fig = plt.figure(figsize=(W, H))
bg = fig.add_axes([0, 0, 1, 1])
bg.axis("off"); bg.set_xlim(0, W); bg.set_ylim(0, H)

bg.text(W / 2, H - 0.38, "Did a tourism ban clear the water at Saint Martin's Island?",
        ha="center", va="center", fontsize=22, fontweight="bold", color=INK)

xs = [0.25, 3.55, 6.85, 10.15]          # left edge of each panel
pw, py, ph = 2.95, 0.95, 3.55           # panel width, bottom, height
titles = ["1  Tourism closure", "2  Human activity", "3  Nearshore turbidity", "4  Why no change?"]
for x, t in zip(xs, titles):
    bg.add_patch(FancyBboxPatch((x, py), pw, ph, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="white", ec="#9a9a9a", lw=1.2))
    bg.text(x + 0.15, py + ph - 0.3, t, fontsize=15, fontweight="bold", color=INK, va="center")
for x in xs[:-1]:
    bg.add_patch(FancyArrowPatch((x + pw + 0.04, py + ph / 2), (x + pw + 0.31, py + ph / 2),
                                 arrowstyle="-|>", mutation_scale=22, color="#555555", lw=2))

bg.text(W / 2, 0.45, f"{n_dates} Sentinel-2 dates ({y0}–{y1})  ·  machine-learning counterfactual  ·  "
        "placebo tests and detection limits", ha="center", fontsize=14, color=Q)

# ------------------------------------------------------------------
# 3. PANEL 1 — island and sectors
# ------------------------------------------------------------------
g = gpd.read_file(ZONES).set_crs(4326, allow_override=True)
a1 = fig.add_axes([(xs[0] + 0.05) / W, (py + 0.75) / H, 1.3 / W, 2.25 / H]); a1.axis("off")
for z in SECTORS:
    g[g["zone"] == z].plot(ax=a1, color=COL[z], alpha=0.85, lw=0)
g[g["zone"] == "island_land"].plot(ax=a1, color="#cfcfcf", edgecolor="#555555", lw=0.4)
a1.set_aspect(1 / np.cos(np.deg2rad(20.6)))

bx = xs[0] + 1.42
bg.text(bx, py + 2.55, "Bangladesh's", fontsize=12.5, color=Q)
bg.text(bx, py + 2.25, "only coral island", fontsize=12.5, color=Q)
bg.text(bx, py + 1.70, "Closed to", fontsize=14, color=INK, fontweight="bold")
bg.text(bx, py + 1.40, "tourists", fontsize=14, color=INK, fontweight="bold")
bg.text(bx, py + 1.10, "Feb–Oct", fontsize=14, color=BLUE, fontweight="bold")
bg.text(bx, py + 0.80, "2025 & 2026", fontsize=13, color=Q)
bg.text(xs[0] + 0.2, py + 0.30, "4 nearshore sectors (0–1 km)", fontsize=12, color=Q)

# ------------------------------------------------------------------
# 4. PANEL 2 — night lights
# ------------------------------------------------------------------
a2 = fig.add_axes([(xs[1] + 0.55) / W, (py + 0.95) / H, 2.1 / W, 1.95 / H])
a2.bar([0, 1], [100, 100 + lights_pct], color=["#bdbdbd", BLUE], width=0.62)
a2.set_xticks([0, 1]); a2.set_xticklabels(["Expected", "Closure"], fontsize=13); a2.set_yticks([])
for s in ["top", "right", "left"]:
    a2.spines[s].set_visible(False)
a2.set_ylim(0, 125)
a2.text(1, 100 + lights_pct + 4, f"{lights_pct:+.0f}%".replace("-", "−"),
        ha="center", va="bottom", fontsize=20, fontweight="bold", color=BLUE)
bg.text(xs[1] + 0.15, py + 0.42, "Satellite night lights vs expected,", fontsize=12, color=Q)
bg.text(xs[1] + 0.15, py + 0.15, "Feb–Apr closures", fontsize=12, color=Q)

# ------------------------------------------------------------------
# 5. PANEL 3 — turbidity effects with placebo ranges
# ------------------------------------------------------------------
a3 = fig.add_axes([(xs[2] + 0.75) / W, (py + 1.0) / H, 2.05 / W, 1.9 / H])
for i, z in enumerate(SECTORS):
    r = main[main["zone"] == z].iloc[0]
    a3.plot([r["plac_lo"], r["plac_hi"]], [-i, -i], color="#dcdcdc", lw=9, solid_capstyle="butt")
    a3.plot(r["effect_pct"], -i, "o", ms=11, color=COL[z], mec="white", mew=1)
a3.axvline(0, color="#555555", lw=1)
a3.set_xlim(-30, 30)
a3.set_yticks([0, -1, -2, -3]); a3.set_yticklabels(LABELS, fontsize=12)
a3.set_xticks([-20, 0, 20]); a3.set_xticklabels(["−20%", "0", "+20%"], fontsize=11)
for s in ["top", "right", "left"]:
    a3.spines[s].set_visible(False)
a3.tick_params(axis="y", length=0)
bg.text(xs[2] + 0.15, py + 0.42, "No detectable change (Feb–Apr)", fontsize=12, color=INK)
bg.text(xs[2] + 0.15, py + 0.15, "grey bars: normal year-to-year range", fontsize=10.5, color=Q)

# ------------------------------------------------------------------
# 6. PANEL 4 — interpretation
# ------------------------------------------------------------------
x4, y = xs[3] + 0.2, py + 2.75
lines = [("Nearshore water tracks", 13, Q, "normal"),
         ("offshore water", 13, Q, "normal"),
         (f"(only {excess_lo:.0f}–{excess_hi:.0f}% more turbid)", 12, Q, "normal"),
         ("", 8, Q, "normal"),
         ("Regional sediment", 15, ORANGE, "bold"),
         ("(Naf River, monsoon waves)", 12, Q, "normal"),
         ("controls water clarity", 15, ORANGE, "bold")]
for text, fs, c, fw in lines:
    bg.text(x4, y, text, fontsize=fs, color=c, fontweight=fw)
    y -= 0.33 if fs > 9 else 0.12
bg.text(x4, py + 0.50, "Closures alone are unlikely", fontsize=12.5, color=INK, fontweight="bold")
bg.text(x4, py + 0.20, "to clear the water", fontsize=12.5, color=INK, fontweight="bold")

# ------------------------------------------------------------------
# 7. SAVE
# ------------------------------------------------------------------
png = os.path.join(OUT, "Graphical_abstract.png")
fig.savefig(png, dpi=300)
fig.savefig(os.path.join(OUT, "Graphical_abstract.pdf"))
plt.close(fig)
Image.open(png).convert("RGB").save(os.path.join(OUT, "Graphical_abstract.tif"),
                                     dpi=(300, 300), compression="tiff_lzw")
print("Saved in:", OUT, "| size (px):", Image.open(png).size)

try:
    from IPython.display import Image as Show, display
    display(Show(filename=png, width=900))
except Exception:
    pass
