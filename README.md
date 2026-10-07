# Did a tourism ban clear the water?

Code and data for the article *Did a tourism ban clear the water? A satellite-based counterfactual assessment of nearshore turbidity at Saint Martin's Island, Bay of Bengal*.

[First author full name], [Supervisor full name]
Department of Fisheries and Marine Science, Noakhali Science and Technology University, Noakhali 3814, Bangladesh

Archived version: https://doi.org/10.5281/zenodo.[number]

## About the study

From November 2024, Bangladesh capped visitor numbers at Saint Martin's Island, its only coral-bearing island, and closed it to tourists from February to October in 2025 and again from February 2026. This study tested whether these restrictions reduced nearshore turbidity. Turbidity was retrieved from 609 Sentinel-2 acquisition dates (2017–2026) for four nearshore and two offshore control sectors. Ridge regression models trained on pre-restriction data predicted the turbidity expected without the restrictions (the counterfactual), and VIIRS night-time lights served as a check that human activity had in fact declined.

This repository contains everything needed to reproduce the numbers, tables and figures of the article: the Google Earth Engine scripts, the Python scripts for Google Colaboratory, the data exported from Earth Engine and the analysis outputs.

## Repository structure

```
gee/       Google Earth Engine (JavaScript) scripts, Steps 2.1–2.6
python/    Python scripts for Google Colaboratory, Steps 3.1–3.5, Fig. 2 and graphical abstract
data/      inputs exported from Earth Engine (CSV and GeoJSON)
outputs/   analysis outputs of Steps 3.1–3.5 (CSV and text files)
```

Figures are not included, because their copyright is transferred to the publisher on publication. All figures can be regenerated with the scripts.

## Workflow

| Step | Script | What it does | Main output |
|---|---|---|---|
| 2.1 | gee/SMI_Step2_1_StudyArea.js | Dry-season shoreline (MNDWI), 0–1 km and 3–6 km rings | SMI_zones_v1 |
| 2.2 | gee/SMI_Step2_2_Sectors.js | Treatment sectors (N, E, W, S), control sectors (E, W), ferry corridor | data/SMI_zones_v2_geojson.geojson |
| 2.3 | gee/SMI_Step2_3_Turbidity.js | Sentinel-2 turbidity (Dogliotti et al., 2015) and red reflectance per zone and date | data/SMI_turbidity_timeseries_v2.csv |
| 2.4 | gee/SMI_Step2_4_Covariates.js | ERA5 wind, waves, SST and rainfall; CHIRPS rainfall | data/SMI_covariates_v1.csv |
| 2.6 | gee/SMI_Step2_6_NightLights.js | VIIRS monthly night lights, island and Teknaf | data/SMI_night_lights_v1.csv |
| 3.1 | python/SMI_Step3_1_QC_EDA_colab.py | Quality control and merging | outputs/SMI_analysis_dataset_v1.csv |
| 3.2 | python/SMI_Step3_2_Counterfactual_colab.py | GOT4.10 tides, four models, leave-one-year-out cross-validation, effects | outputs/step3_2/ |
| 3.3 | python/SMI_Step3_3_Robustness_colab.py | Placebo intervals, robustness checks (incl. red reflectance), injection test, permutation importance, COVID-19 dates | outputs/step3_3/ |
| 3.4 | python/SMI_Step3_4_NightLights_colab.py | Night-light counterfactual (manipulation check) | outputs/step3_4/ |
| 3.5 | python/SMI_Step3_5_Manuscript_numbers_colab.py | Remaining manuscript numbers, Tables 5–7, Figs 1 and 3–7 | outputs/step3_5/ |
| Fig. 2 | python/SMI_Fig2_workflow_colab.py | Workflow diagram | outputs/step3_5/ |
| GA | python/SMI_Graphical_abstract_colab.py | Graphical abstract | outputs/figures/ |

There is no Step 2.5 in the final workflow.

## How to reproduce

**Earth Engine (Steps 2.1–2.6).** Open each script in the Earth Engine Code Editor and replace the placeholder `YOUR_PROJECT_ID` with your own Google Cloud project ID. The sector polygons of Step 2.2 (north, south, west and ferry corridor) were digitised by hand in the Code Editor; the resulting zones are provided in `data/SMI_zones_v2_geojson.geojson`, so the later steps can be reproduced exactly without redrawing them.

**Python (Steps 3.1–3.5).** The scripts were written for Google Colaboratory with Google Drive.

1. In your Google Drive, create the folder `SMI_project/` and copy the four files of `data/` into it.
2. To start from the archived outputs instead of rerunning everything, also copy the folder `outputs/` into `SMI_project/`.
3. In Colab, mount Drive in a first cell (`from google.colab import drive; drive.mount('/content/drive')`), then paste a script into a second cell and run it.
4. Run the steps in the order of the table above.

Paths are set at the top of each script. Steps 3.2–3.5 and the figure scripts also accept the environment variables `SMI_DATA` (input folder) and `SMI_BASE` (output folder).

## Software

The analyses were run in Google Colaboratory (September–October 2026) with Python 3.13.16, pandas 2.2.3, NumPy 2.1.3, scikit-learn 1.6.1, xgboost 3.4.1 and pyTMD 3.0.9. All random processes use a fixed seed (42). Step 3.2 downloads the GOT4.10 tide model through pyTMD on first use and caches the predictions in `outputs/step3_2/tides_GOT410.csv`. The two field turbidity values compared in Step 3.5 are taken from Rahman et al. (2026) and are typed in the script with a comment.

## Input data sources

All inputs are openly available and were accessed through Google Earth Engine, except the tide model:

- Sentinel-2 MSI Level-2A surface reflectance (`COPERNICUS/S2_SR_HARMONIZED`; European Space Agency, Copernicus programme)
- Cloud Score+ (`GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`)
- ERA5 hourly reanalysis (`ECMWF/ERA5/HOURLY`; European Centre for Medium-Range Weather Forecasts, Copernicus Climate Change Service)
- CHIRPS daily rainfall (`UCSB-CHG/CHIRPS/DAILY`; Climate Hazards Center, University of California, Santa Barbara)
- VIIRS Day/Night Band monthly composites (`NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`; Earth Observation Group, Payne Institute for Public Policy, Colorado School of Mines)
- GOT4.10 ocean tide model (NASA Goddard Space Flight Center), accessed with pyTMD

## Licence

Code and data are released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0); see `LICENSE`.

## How to cite

Please cite the article and this archive:

[First author], [Supervisor], 2026. Code and data for: Did a tourism ban clear the water? A satellite-based counterfactual assessment of nearshore turbidity at Saint Martin's Island, Bay of Bengal (v1.0). Zenodo. https://doi.org/10.5281/zenodo.[number]

## Contact

[Corresponding author name], [email address]
