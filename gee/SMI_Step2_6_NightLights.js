/**************************************************************
 * Saint Martin's Island — Step 2.6
 * Night lights (VIIRS monthly) — did tourism activity on the
 * island actually stop during the restrictions?
 *
 * Island area = island + 300 m (hotels, jetty, beaches).
 * Reference   = Teknaf town (regional changes, e.g. electricity).
 * Output: one CSV row per month.
 * Version : 1.0 (October 2026)
 **************************************************************/

var PROJECT_ID = 'YOUR_PROJECT_ID';
var START = '2017-01-01';
var END   = '2026-10-01';
var SCALE = 463.83;                     // native VIIRS monthly pixel (m)

var zones  = ee.FeatureCollection('projects/' + PROJECT_ID + '/assets/SMI_zones_v2');
var island = zones.filter(ee.Filter.eq('zone', 'island_land')).geometry().buffer(300, 10);
var teknaf = ee.Geometry.Point([92.30, 20.86]).buffer(3000);   // Teknaf town (approx.)

// Stray-light corrected monthly composites
var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
  .filterDate(START, END);

print('VIIRS months available:', viirs.size());
print('Last month available:',
      ee.Date(viirs.aggregate_max('system:time_start')).format('YYYY-MM'));

var rows = viirs.map(function (img) {
  var isl = img.select(['avg_rad', 'cf_cvg']).reduceRegion({
    reducer: ee.Reducer.sum().combine(ee.Reducer.mean(), '', true),
    geometry: island, scale: SCALE, maxPixels: 1e9});
  var tek = img.select('avg_rad').reduceRegion({
    reducer: ee.Reducer.sum(), geometry: teknaf, scale: SCALE, maxPixels: 1e9});
  return ee.Feature(null, {
    'system:time_start': img.get('system:time_start'),
    month: ee.Date(img.get('system:time_start')).format('YYYY-MM'),
    island_light_sum: isl.get('avg_rad_sum'),     // total radiance, island
    island_cloudfree_nights: isl.get('cf_cvg_mean'),  // nights used in the composite
    teknaf_light_sum: tek.get('avg_rad')          // total radiance, Teknaf
  });
});

// Quick chart
print(ui.Chart.feature.byFeature(rows, 'system:time_start', ['island_light_sum', 'teknaf_light_sum'])
  .setOptions({title: 'Monthly night lights: Saint Martin (island+300 m) vs Teknaf',
               vAxis: {title: 'Sum of radiance (nW/cm²/sr)', scaleType: 'log'},
               lineWidth: 1, pointSize: 2}));

// Map: tourist season vs ban
Map.centerObject(island, 12);
var vis = {min: 0, max: 10, palette: ['000000', '444444', 'ffff66', 'ffffff']};
Map.addLayer(viirs.filterDate('2024-01-01', '2024-02-01').first().select('avg_rad'), vis,
             'Night lights Jan 2024 (tourist season)');
Map.addLayer(viirs.filterDate('2025-03-01', '2025-04-01').first().select('avg_rad'), vis,
             'Night lights Mar 2025 (ban)');
Map.addLayer(ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(island)]), 1, 2),
             {palette: ['00ffff']}, 'Island + 300 m');

Export.table.toDrive({
  collection: rows,
  description: 'SMI_night_lights_v1',
  folder: 'SMI_project',
  fileFormat: 'CSV',
  selectors: ['month', 'island_light_sum', 'island_cloudfree_nights', 'teknaf_light_sum']
});
