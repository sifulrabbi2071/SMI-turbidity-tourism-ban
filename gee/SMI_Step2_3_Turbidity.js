/**************************************************************
 * Saint Martin's Island — Phase 2, Step 2.3
 * Turbidity time series from Sentinel-2 for every zone
 *
 * Output: one CSV row per (acquisition date × zone) with
 *         turbidity statistics, valid-pixel fraction and
 *         image metadata. Tide, rainfall, SST etc. will be
 *         added later in Python.
 *
 * Turbidity algorithm: Dogliotti et al. (2015, Remote Sensing
 *   of Environment) semi-analytical switching algorithm:
 *   red band (B4, 665 nm) for clearer water, NIR (B8A, 865 nm)
 *   for very turbid water, linear blend in between.
 *   NOTE: coefficients below are the published Dogliotti values;
 *   verify against the paper and test band-specific
 *   coefficients / ACOLITE correction in the validation step.
 *
 * Version : 1.1 (September 2026) — finer histograms for median /
 *           percentiles (v1 percentiles were quantised), NIR check
 **************************************************************/

// ============================================================
// 0. SETTINGS
// ============================================================
var PROJECT_ID = 'YOUR_PROJECT_ID';

var START = '2017-01-01';        // earliest available L2A will be used
var END   = '2026-10-01';

var CLEAR_THRESHOLD = 0.60;      // Cloud Score+ (same as Step 2.1)
var SWIR_MAX        = 0.03;      // water pixels must have low SWIR (removes glint, foam, haze)
var RED_MAX         = 0.15;      // avoid algorithm blow-up / residual cloud
var INNER_EXCLUDE   = 300;       // m; '_deep' zones exclude 0–300 m from shore (shallow bottom effect)
var SCALE           = 20;        // m; B8A and B11 are 20 m bands

// Dogliotti et al. (2015) coefficients
var A_RED = 228.1,  C_RED = 0.1641;   // red
var A_NIR = 3078.9, C_NIR = 0.2112;   // NIR
var BLEND_LOW = 0.05, BLEND_HIGH = 0.07;  // red reflectance switching range

// ============================================================
// 1. ZONES
// ============================================================
var zones = ee.FeatureCollection('projects/' + PROJECT_ID + '/assets/SMI_zones_v2');
var island = zones.filter(ee.Filter.eq('zone', 'island_land')).geometry();

// Treatment + control zones
var mainZones = zones.filter(ee.Filter.inList('type', ['treatment', 'control']));

// Extra '_deep' versions of the treatment zones (300–1000 m from shore)
var deepZones = zones.filter(ee.Filter.eq('type', 'treatment')).map(function (f) {
  var g = f.geometry().difference(island.buffer(INNER_EXCLUDE, 10), 10);
  return ee.Feature(g, {
    zone: ee.String(f.get('zone')).cat('_deep'),
    type: 'treatment_deep',
    area_km2: g.area(10).divide(1e6)
  });
});

var allZones = mainZones.merge(deepZones);
var aoi = allZones.geometry().bounds();

print('Zones used (should be 10):', allZones.aggregate_array('zone'));

// ============================================================
// 2. SENTINEL-2 PREPARATION
// ============================================================
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(START, END)
  .linkCollection(csPlus, ['cs_cdf']);

print('First image date:', ee.Date(s2.aggregate_min('system:time_start')).format('YYYY-MM-dd'));
print('Last image date:',  ee.Date(s2.aggregate_max('system:time_start')).format('YYYY-MM-dd'));
print('Number of Sentinel-2 scenes:', s2.size());

function prep(img) {
  var r = img.select(['B3', 'B4', 'B8A', 'B11']).divide(10000);
  var red = r.select('B4');
  var nir = r.select('B8A');

  // Pixel quality: clear sky + open water + low SWIR + sensible red
  var clear = img.select('cs_cdf').gte(CLEAR_THRESHOLD);
  var water = r.normalizedDifference(['B3', 'B11']).gt(0);
  var lowSwir = r.select('B11').lt(SWIR_MAX);
  var okRed = red.gt(0).and(red.lt(RED_MAX));
  var okNir = nir.lt(RED_MAX);                 // avoids NIR-branch blow-up
  var valid = clear.and(water).and(lowSwir).and(okRed).and(okNir);

  // Dogliotti turbidity (FNU)
  var tRed = red.multiply(A_RED).divide(ee.Image(1).subtract(red.divide(C_RED)));
  var tNir = nir.multiply(A_NIR).divide(ee.Image(1).subtract(nir.divide(C_NIR)));
  var w = red.subtract(BLEND_LOW).divide(BLEND_HIGH - BLEND_LOW).clamp(0, 1);
  var turb = tRed.multiply(ee.Image(1).subtract(w)).add(tNir.multiply(w)).rename('turbidity');

  return ee.Image(
    turb.addBands(red.rename('rho_red'))
        .updateMask(valid)
        .copyProperties(img, ['system:time_start', 'SPACECRAFT_NAME', 'MEAN_SOLAR_ZENITH_ANGLE'])
  );
}

var prepped = s2.map(prep);

// ============================================================
// 3. DAILY MOSAICS (the island lies where two tiles overlap)
// ============================================================
var dates = prepped.aggregate_array('system:time_start')
  .map(function (t) { return ee.Date(t).format('YYYY-MM-dd'); })
  .distinct();

var daily = ee.ImageCollection.fromImages(dates.map(function (d) {
  var start = ee.Date.parse('YYYY-MM-dd', d);
  var col = prepped.filterDate(start, start.advance(1, 'day'));
  var first = ee.Image(col.first());
  return col.mosaic().set({
    date: d,
    'system:time_start': first.get('system:time_start'),
    spacecraft: first.get('SPACECRAFT_NAME'),
    sun_zenith: first.get('MEAN_SOLAR_ZENITH_ANGLE'),
    n_tiles: col.size()
  });
}));

print('Number of acquisition dates:', dates.size());

// ============================================================
// 4. ZONAL STATISTICS
// ============================================================
// maxBuckets = 10000: fine histogram so median / percentiles are precise
var MAX_BUCKETS = 10000;
var reducer = ee.Reducer.median(MAX_BUCKETS)
  .combine(ee.Reducer.mean(), '', true)
  .combine(ee.Reducer.percentile([25, 75], null, MAX_BUCKETS), '', true)
  .combine(ee.Reducer.count(), '', true);

var rows = daily.map(function (img) {
  // valid = 1 where turbidity exists, 0 elsewhere -> mean = valid fraction
  var valid = img.select('turbidity').mask().rename('valid');
  var stack = img.select(['turbidity', 'rho_red']).addBands(valid);

  var stats = stack.reduceRegions({
    collection: allZones,
    reducer: reducer,
    scale: SCALE,
    tileScale: 4
  });

  return stats.map(function (f) {
    return ee.Feature(null, f.toDictionary()).set({
      date: img.get('date'),
      time_utc: ee.Date(img.get('system:time_start')).format('HH:mm'),
      spacecraft: img.get('spacecraft'),
      sun_zenith: img.get('sun_zenith'),
      n_tiles: img.get('n_tiles')
    });
  });
}).flatten();

print('Preview of first 3 rows:', rows.limit(3));

// ============================================================
// 5. MAP PREVIEW (one clear winter day)
// ============================================================
var exampleRaw = s2.filterDate('2024-01-01', '2024-02-01')
  .sort('CLOUDY_PIXEL_PERCENTAGE').first();
var exampleTurb = prep(ee.Image(exampleRaw)).select('turbidity');

Map.centerObject(allZones, 11);
Map.setOptions('SATELLITE');
Map.addLayer(exampleTurb, {min: 0, max: 30,
  palette: ['08306b', '2171b5', '6baed6', 'c6dbef', 'fdd49e', 'fc8d59', 'b30000']},
  'Turbidity example, Jan 2024 (FNU)');
Map.addLayer(ee.Image().byte().paint(allZones, 1, 1), {palette: ['ffffff']}, 'Zones');
print('Example image date:', ee.Date(ee.Image(exampleRaw).get('system:time_start')).format('YYYY-MM-dd'));

// ============================================================
// 6. EXPORT (Tasks tab → RUN). May take 20–60 minutes.
// ============================================================
Export.table.toDrive({
  collection: rows,
  description: 'SMI_turbidity_timeseries_v2',
  folder: 'SMI_project',
  fileFormat: 'CSV',
  selectors: ['date', 'time_utc', 'spacecraft', 'sun_zenith', 'n_tiles',
              'zone', 'type', 'area_km2',
              'valid_mean', 'valid_count',
              'turbidity_median', 'turbidity_mean', 'turbidity_p25', 'turbidity_p75', 'turbidity_count',
              'rho_red_median']
});
