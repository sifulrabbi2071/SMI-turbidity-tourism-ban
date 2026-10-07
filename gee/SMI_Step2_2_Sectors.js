/**************************************************************
 * Saint Martin's Island — Phase 2, Step 2.2
 * Sectors: split the nearshore (treatment) ring into
 *          North / East / West / South, remove the ferry
 *          corridor from the control ring, and split the
 *          control ring into East / West.
 * Version : 1.1 (September 2026) — shows the rings first and
 *           waits until all four polygons are drawn
 *
 * HOW TO USE
 *  1. Run once: the rings from Step 2.1 appear on the map.
 *  2. Draw 4 polygons (each as a NEW LAYER) and rename the
 *     imports EXACTLY:  north, south, west, corridor
 *       north    – nearshore ring around the northern block
 *                  (jetty, main beaches) down to the narrow neck
 *       south    – nearshore ring around Dakhin Para + Chhera Dwip
 *       west     – WEST half of the middle section (east half is
 *                  computed automatically)
 *       corridor – ferry route: band ~1–1.5 km wide from the jetty
 *                  towards Teknaf, crossing the whole control ring
 *  3. Run again: coloured sectors + areas appear.
 **************************************************************/

// ============================================================
// 0. SETTINGS
// ============================================================
var PROJECT_ID = 'YOUR_PROJECT_ID';
var ME = 10;   // maximum geometric error (m)

var zones1 = ee.FeatureCollection('projects/' + PROJECT_ID + '/assets/SMI_zones_v1');

var island   = zones1.filter(ee.Filter.eq('zone', 'island_land')).geometry();
var nearRing = zones1.filter(ee.Filter.eq('zone', 'nearshore_0_1km')).geometry();
var ctrlRing = zones1.filter(ee.Filter.eq('zone', 'control_3_6km')).geometry();

Map.centerObject(nearRing, 12);
Map.setOptions('SATELLITE');

function outline(geom) {
  return ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(geom)]), 1, 2);
}
Map.addLayer(outline(island),   {palette: ['FFFF00']}, 'Island (yellow)');
Map.addLayer(outline(nearRing), {palette: ['FF0000']}, 'Nearshore 0–1 km (red)');
Map.addLayer(outline(ctrlRing), {palette: ['00FFFF']}, 'Control 3–6 km (cyan)');

// ============================================================
// Check whether all four polygons have been drawn
// ============================================================
var missing = [];
if (typeof north    === 'undefined') missing.push('north');
if (typeof south    === 'undefined') missing.push('south');
if (typeof west     === 'undefined') missing.push('west');
if (typeof corridor === 'undefined') missing.push('corridor');

if (missing.length > 0) {
  print('Still to draw (then Run again): ' + missing.join(', '));
} else {
  runSectors();
}

// ============================================================
// Main analysis — runs only when all polygons exist
// ============================================================
function runSectors() {

  // 1. Treatment sectors (priority: north, south, west; east = rest)
  var tNorth = nearRing.intersection(north, ME);
  var tSouth = nearRing.intersection(south, ME).difference(north, ME);
  var middle = nearRing.difference(north, ME).difference(south, ME);
  var tWest  = middle.intersection(west, ME);
  var tEast  = middle.difference(west, ME);

  // 2. Control sectors: remove ferry corridor, split at island centroid longitude
  var ctrl = ctrlRing.difference(corridor, 50);
  var cx = island.centroid(ME).coordinates().getNumber(0);
  var eastHalf = ee.Geometry.Rectangle(ee.List([cx, 20.0, 93.0, 21.5]), 'EPSG:4326', false);
  var westHalf = ee.Geometry.Rectangle(ee.List([91.5, 20.0, cx, 21.5]), 'EPSG:4326', false);
  var cEast = ctrl.intersection(eastHalf, 50);
  var cWest = ctrl.intersection(westHalf, 50);

  // 3. Combine, compute areas, display
  var zones2 = ee.FeatureCollection([
    ee.Feature(island,   {zone: 'island_land',    type: 'land'}),
    ee.Feature(tNorth,   {zone: 'T_north',        type: 'treatment'}),
    ee.Feature(tEast,    {zone: 'T_east',         type: 'treatment'}),
    ee.Feature(tWest,    {zone: 'T_west',         type: 'treatment'}),
    ee.Feature(tSouth,   {zone: 'T_south',        type: 'treatment'}),
    ee.Feature(cEast,    {zone: 'C_east',         type: 'control'}),
    ee.Feature(cWest,    {zone: 'C_west',         type: 'control'}),
    ee.Feature(corridor, {zone: 'ferry_corridor', type: 'excluded'})
  ]).map(function (f) {
    return f.set('area_km2', f.geometry().area(ME).divide(1e6));
  });

  print('Zone areas (km²):',
        zones2.reduceColumns(ee.Reducer.toList(2), ['zone', 'area_km2']).get('list'));

  var colours = ee.Dictionary({
    island_land: 'ffffff', T_north: 'e41a1c', T_east: 'ff7f00', T_west: '984ea3',
    T_south: '4daf4a', C_east: '377eb8', C_west: '00ffff', ferry_corridor: 'ffff00'
  });
  var styled = zones2.map(function (f) {
    var c = ee.String(colours.get(f.get('zone')));
    return f.set('style', ee.Dictionary({color: c, fillColor: c.cat('66'), width: 1}));
  });
  Map.addLayer(styled.style({styleProperty: 'style'}), {}, 'Zones v2 (coloured)');
  // red = T_north, orange = T_east, purple = T_west, green = T_south,
  // blue = C_east, cyan = C_west, yellow = ferry corridor

  // 4. Save (Tasks tab → RUN each) — only after checking the map!
  Export.table.toAsset({
    collection: zones2,
    description: 'SMI_zones_v2_asset',
    assetId: 'projects/' + PROJECT_ID + '/assets/SMI_zones_v2'
  });
  Export.table.toDrive({
    collection: zones2,
    description: 'SMI_zones_v2_geojson',
    folder: 'SMI_project',
    fileFormat: 'GeoJSON'
  });
}
