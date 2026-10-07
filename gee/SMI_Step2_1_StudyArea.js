/**************************************************************
 * Saint Martin's Island — Phase 2, Step 2.1
 * Study area: land mask, nearshore (treatment) ring and
 *             offshore (control) ring
 *
 * Project : ML-based counterfactual assessment of tourism
 *           restriction impacts on coastal and marine
 *           ecosystems of Saint Martin's Island
 * Version : 1.1 (September 2026) — inland water holes filled
 * Platform: Google Earth Engine Code Editor (JavaScript)
 **************************************************************/

// ============================================================
// 0. SETTINGS — change values ONLY in this block
// ============================================================
var PROJECT_ID = 'YOUR_PROJECT_ID';   // your Cloud Project ID

// Box around the island + offshore waters
var AOI = ee.Geometry.Rectangle([92.22, 20.50, 92.42, 20.72]);

// Small box used to keep ONLY the island's land polygons
// (excludes Teknaf / Myanmar mainland if they appear in AOI)
var ISLAND_BOX = ee.Geometry.Rectangle([92.28, 20.55, 92.37, 20.67]);

// Dry season BEFORE the restriction (clearest images)
var MASK_START = '2023-12-01';
var MASK_END   = '2024-03-01';

var CLEAR_THRESHOLD = 0.60;   // Cloud Score+ (0–1). Higher = stricter
var MNDWI_THRESHOLD = 0.0;    // pixels with MNDWI below this = land
var MIN_LAND_AREA   = 5000;   // m². Land patches smaller than this are removed

var NEAR_DIST  = 1000;        // Treatment ring: 0–1 km from shore (m)
var CTRL_INNER = 3000;        // Control ring: 3–6 km from shore (m)
var CTRL_OUTER = 6000;

Map.centerObject(AOI, 12);
Map.setOptions('SATELLITE');

// ============================================================
// 1. Sentinel-2 surface reflectance + Cloud Score+ masking
// ============================================================
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(AOI)
  .filterDate(MASK_START, MASK_END)
  .linkCollection(csPlus, ['cs_cdf'])          // attach cloud score band
  .map(function (img) {
    var clear = img.select('cs_cdf').gte(CLEAR_THRESHOLD);
    return ee.Image(
      img.select(['B2', 'B3', 'B4', 'B8', 'B11'])  // blue, green, red, NIR, SWIR1
         .divide(10000)                            // to reflectance (0–1)
         .updateMask(clear)                        // remove cloudy pixels
         .copyProperties(img, ['system:time_start'])
    );
  });

print('Sentinel-2 images used for the land mask:', s2.size());

// Median of all clear pixels = one cloud-free image
var composite = s2.median().clip(AOI);
Map.addLayer(composite, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.25},
             '1. True colour (Dec 2023 – Feb 2024)');

// ============================================================
// 2. Land mask with MNDWI = (Green − SWIR1) / (Green + SWIR1)
//    Water > 0, land < 0
// ============================================================
var mndwi = composite.normalizedDifference(['B3', 'B11']).rename('MNDWI');
Map.addLayer(mndwi, {min: -0.5, max: 0.5, palette: ['8c510a', 'f5f5f5', '01665e']},
             '2. MNDWI (brown = land, green = water)', false);

var land = mndwi.lt(MNDWI_THRESHOLD).selfMask();

// Raster land pixels -> polygons
var landPolys = land.reduceToVectors({
  geometry: AOI,
  scale: 10,
  geometryType: 'polygon',
  eightConnected: false,
  maxPixels: 1e9
});

// Keep only island polygons larger than MIN_LAND_AREA.
// v1.1: keep only the OUTER boundary of each polygon, so inland
// ponds / wetlands (holes) become part of the island and are NOT
// counted as nearshore sea water.
var islandPolys = landPolys
  .filterBounds(ISLAND_BOX)
  .map(function (f) {
    var outer = ee.Geometry.Polygon(ee.List(f.geometry().coordinates()).get(0));
    return ee.Feature(outer).set('area_m2', outer.area(1));
  })
  .filter(ee.Filter.gte('area_m2', MIN_LAND_AREA));

var island = islandPolys.union(1).geometry();

print('Island polygons kept:', islandPolys.size());
print('Island land area (km², median tide):', island.area(1).divide(1e6));

// ============================================================
// 3. Treatment and control zones (distance from shoreline)
// ============================================================
var nearRing = island.buffer(NEAR_DIST, 10).difference(island, 10);
var ctrlRing = island.buffer(CTRL_OUTER, 50)
                     .difference(island.buffer(CTRL_INNER, 50), 50);

print('Nearshore ring 0–1 km area (km²):', nearRing.area(10).divide(1e6));
print('Control ring 3–6 km area (km²):', ctrlRing.area(50).divide(1e6));

// Draw outlines on the map
function outline(geom) {
  return ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(geom)]), 1, 2);
}
Map.addLayer(outline(island),   {palette: ['FFFF00']}, '3. Island outline (yellow)');
Map.addLayer(outline(nearRing), {palette: ['FF0000']}, '4. Treatment: nearshore 0–1 km (red)');
Map.addLayer(outline(ctrlRing), {palette: ['00FFFF']}, '5. Control: offshore 3–6 km (cyan)');

// NOTE: any mainland inside the control ring and all cloud/land pixels
// will be removed image-by-image in later steps with a water mask.

// ============================================================
// 4. Save zones (go to the Tasks tab and click RUN for each)
// ============================================================
var zones = ee.FeatureCollection([
  ee.Feature(island,   {zone: 'island_land'}),
  ee.Feature(nearRing, {zone: 'nearshore_0_1km'}),
  ee.Feature(ctrlRing, {zone: 'control_3_6km'})
]);

// (a) To your GEE Assets — all later scripts will load this
Export.table.toAsset({
  collection: zones,
  description: 'SMI_zones_v1_asset',
  assetId: 'projects/' + PROJECT_ID + '/assets/SMI_zones_v1'
});

// (b) To Google Drive as GeoJSON — for the Study Area map in the paper
Export.table.toDrive({
  collection: zones,
  description: 'SMI_zones_v1_geojson',
  folder: 'SMI_project',
  fileFormat: 'GeoJSON'
});
