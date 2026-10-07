/**************************************************************
 * Saint Martin's Island — Phase 2, Step 2.4
 * Environmental covariates (drivers) for every Sentinel-2
 * acquisition date: wind, waves, sea surface temperature and
 * rainfall (local + Naf catchment), over windows BEFORE the
 * satellite overpass.
 *
 * Output: one CSV row per acquisition date. Tide will be added
 *         later in Python.
 * Units in the CSV (convert in Python):
 *   wind u/v/speed  : m/s
 *   waves (swh)     : m
 *   sst             : Kelvin   (°C = K − 273.15)
 *   ERA5 tp         : metres   (mm = m × 1000)
 *   CHIRPS          : mm
 *
 * Version : 1.1 (September 2026) — empty value when a time window
 *           is incomplete (end of ERA5 / CHIRPS record)
 **************************************************************/

// ============================================================
// 0. SETTINGS
// ============================================================
var PROJECT_ID = 'YOUR_PROJECT_ID';
var START = '2017-01-01';
var END   = '2026-10-01';

// Open-sea point ~18 km west of the island (for wind, waves, SST;
// avoids ERA5 coastal/land pixels)
var SEA_PT = ee.Geometry.Point([92.15, 20.62]);

// Local box around the island (local rainfall)
var LOCAL = ee.Geometry.Rectangle([92.20, 20.50, 92.45, 20.75]);

// Approximate Naf River / Teknaf catchment (rainfall that brings
// sediment). Approximate box — can be refined with HydroSHEDS later.
var CATCH = ee.Geometry.Rectangle([92.05, 20.75, 92.45, 21.35]);

// ERA5 band names
var U = 'u_component_of_wind_10m';
var V = 'v_component_of_wind_10m';
var TP = 'total_precipitation';
var SST = 'sea_surface_temperature';
var SWH = 'significant_height_of_combined_wind_waves_and_swell';
var ERA_SCALE = 27830;

// ============================================================
// 1. DATASETS + availability checks
// ============================================================
var era5   = ee.ImageCollection('ECMWF/ERA5/HOURLY');
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY');

var eraBands = era5.first().bandNames();
print('ERA5: needed bands that exist (should be 5):',
      ee.List([U, V, TP, SST, SWH]).filter(ee.Filter.inList('item', eraBands)));
print('ERA5: all wave-related band names:',
      eraBands.filter(ee.Filter.stringContains('item', 'wave')));
print('ERA5 last available date:',
      ee.Date(era5.filterDate('2026-01-01', '2027-01-01')
                  .aggregate_max('system:time_start')).format('YYYY-MM-dd HH:mm'));
print('CHIRPS last available date:',
      ee.Date(chirps.filterDate('2026-01-01', '2027-01-01')
                    .aggregate_max('system:time_start')).format('YYYY-MM-dd'));

// ============================================================
// 2. SENTINEL-2 ACQUISITION DATES (same filters as Step 2.3)
// ============================================================
var zones = ee.FeatureCollection('projects/' + PROJECT_ID + '/assets/SMI_zones_v2');
var aoi = zones.filter(ee.Filter.inList('type', ['treatment', 'control'])).geometry().bounds();

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(START, END);

var dateFC = ee.FeatureCollection(
  s2.aggregate_array('system:time_start').map(function (t) {
    return ee.Feature(null, {date: ee.Date(t).format('YYYY-MM-dd'), t: t});
  })
).distinct('date');

print('Number of acquisition dates (should be 609):', dateFC.size());

// ============================================================
// 3. HELPER FUNCTIONS
//    v1.1: each function returns an empty value (null) unless the
//    WHOLE time window is available (no partial sums at the end
//    of the record, e.g. CHIRPS ends 2026-08-31).
// ============================================================
// Mean of an ERA5 band over the `hours` before time t, at geometry g
function eraMean(band, t, hours, g) {
  var col = era5.select(band).filterDate(t.advance(-hours, 'hour'), t);
  var val = col.mean()
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: g, scale: ERA_SCALE});
  return ee.Algorithms.If(col.size().eq(hours), ee.Dictionary(val).get(band), null);
}

// Sum of an ERA5 band over the `hours` before time t
function eraSum(band, t, hours, g) {
  var col = era5.select(band).filterDate(t.advance(-hours, 'hour'), t);
  var val = col.sum()
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: g, scale: ERA_SCALE});
  return ee.Algorithms.If(col.size().eq(hours), ee.Dictionary(val).get(band), null);
}

// Mean hourly wind speed over the `hours` before time t
function windSpeed(t, hours, g) {
  var col = era5.select([U, V]).filterDate(t.advance(-hours, 'hour'), t);
  var val = col.map(function (i) {
      return i.expression('sqrt(u*u + v*v)', {u: i.select(U), v: i.select(V)}).rename('ws');
    })
    .mean()
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: g, scale: ERA_SCALE});
  return ee.Algorithms.If(col.size().eq(hours), ee.Dictionary(val).get('ws'), null);
}

// CHIRPS rainfall summed over the `days` BEFORE the overpass day
function chirpsSum(day0, days, g) {
  var col = chirps.filterDate(day0.advance(-days, 'day'), day0);
  var val = col.sum()
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: g, scale: 5566});
  return ee.Algorithms.If(col.size().eq(days), ee.Dictionary(val).get('precipitation'), null);
}

// ============================================================
// 4. COVARIATES FOR EACH DATE
// ============================================================
var rows = dateFC.map(function (f) {
  var t = ee.Date(f.get('t'));                          // overpass time (UTC)
  var day0 = ee.Date(t.format('YYYY-MM-dd'));           // 00:00 UTC of overpass day
  return ee.Feature(null, {
    date: f.get('date'),
    time_utc: t.format('HH:mm'),

    // Wind at sea point
    u10_24h:  eraMean(U, t, 24, SEA_PT),
    v10_24h:  eraMean(V, t, 24, SEA_PT),
    ws10_24h: windSpeed(t, 24, SEA_PT),
    ws10_72h: windSpeed(t, 72, SEA_PT),

    // Waves and SST at sea point
    swh_24h: eraMean(SWH, t, 24, SEA_PT),
    sst_24h: eraMean(SST, t, 24, SEA_PT),

    // ERA5 rainfall (metres)
    tp_local_24h:  eraSum(TP, t, 24, LOCAL),
    tp_catch_24h:  eraSum(TP, t, 24, CATCH),
    tp_catch_72h:  eraSum(TP, t, 72, CATCH),
    tp_catch_168h: eraSum(TP, t, 168, CATCH),

    // CHIRPS rainfall over the catchment (mm)
    chirps_catch_1d:  chirpsSum(day0, 1, CATCH),
    chirps_catch_3d:  chirpsSum(day0, 3, CATCH),
    chirps_catch_7d:  chirpsSum(day0, 7, CATCH),
    chirps_catch_30d: chirpsSum(day0, 30, CATCH)
  });
});

print('Preview of first 3 rows:', rows.limit(3));

// Map check
Map.centerObject(CATCH, 9);
Map.addLayer(CATCH, {color: 'orange'}, 'Naf catchment box (rainfall)');
Map.addLayer(LOCAL, {color: 'cyan'}, 'Local box (rainfall)');
Map.addLayer(SEA_PT, {color: 'red'}, 'Sea point (wind, waves, SST)');
Map.addLayer(zones, {color: 'white'}, 'Zones v2');

// ============================================================
// 5. EXPORT (Tasks tab → RUN)
// ============================================================
Export.table.toDrive({
  collection: rows,
  description: 'SMI_covariates_v1',
  folder: 'SMI_project',
  fileFormat: 'CSV',
  selectors: ['date', 'time_utc',
              'u10_24h', 'v10_24h', 'ws10_24h', 'ws10_72h',
              'swh_24h', 'sst_24h',
              'tp_local_24h', 'tp_catch_24h', 'tp_catch_72h', 'tp_catch_168h',
              'chirps_catch_1d', 'chirps_catch_3d', 'chirps_catch_7d', 'chirps_catch_30d']
});
