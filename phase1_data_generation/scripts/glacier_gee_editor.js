/**
 * Glacier Melting Detection — GEE Code Editor Script
 * ====================================================
 * Project : Final Year Project
 * GEE ID  : final-year-project-485804
 *
 * Paste this file into: https://code.earthengine.google.com
 * Sign in with the Google account linked to project: final-year-project-485804
 *
 * This script:
 *   1. Builds all spectral composites for a chosen year
 *   2. Visualises all layers on the interactive map
 *   3. Runs a stratified sample → CSV (saved to your Google Drive)
 *   4. Exports a sample of image patches
 *   5. Shows a NDSI time-series chart (2018–2025)
 *
 * NOTE: The Code Editor exports go to Google Drive only (GEE limitation).
 *       Use generate_numerical_dataset.py for direct local downloads.
 */

// ═══════════════════════════════════════════════
// SECTION 0 — CONFIG
// ═══════════════════════════════════════════════
var YEAR        = 2023;           // ← change year here (2018–2025)
var START_DATE  = YEAR + '-05-01';
var END_DATE    = YEAR + '-09-30';
var CLOUD_COVER = 20;

var AOI = ee.Geometry.Polygon([
  [66.0, 29.0],[66.0, 40.0],[85.0, 40.0],[85.0, 29.0],[66.0, 29.0]
]);

Map.centerObject(AOI, 6);
Map.setOptions('SATELLITE');


// ═══════════════════════════════════════════════
// SECTION 1 — CLOUD MASKING
// ═══════════════════════════════════════════════
function maskS2(img) {
  var scl = img.select('SCL');
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
  return img.updateMask(mask).divide(10000).copyProperties(img,['system:time_start']);
}

function maskLandsat(img) {
  var qa = img.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1<<3).eq(0).and(qa.bitwiseAnd(1<<4).eq(0));
  var opt = img.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thm = img.select('ST_B10').multiply(0.00341802).add(149.0);
  return img.addBands(opt,null,true).addBands(thm,null,true).updateMask(mask);
}


// ═══════════════════════════════════════════════
// SECTION 2 — SPECTRAL INDICES
// ═══════════════════════════════════════════════
function addIndicesS2(img) {
  var ndsi = img.normalizedDifference(['B3','B11']).rename('NDSI');
  var ndwi = img.normalizedDifference(['B3','B8']).rename('NDWI');
  var ndvi = img.normalizedDifference(['B8','B4']).rename('NDVI');
  var bai  = ee.Image(1).divide(
    img.select('B4').subtract(0.1).pow(2).add(img.select('B8').subtract(0.06).pow(2))
  ).rename('BAI');
  return img.addBands([ndsi,ndwi,ndvi,bai]);
}

function addIndicesLandsat(img) {
  var ndsi = img.normalizedDifference(['SR_B3','SR_B6']).rename('NDSI');
  var ndwi = img.normalizedDifference(['SR_B3','SR_B5']).rename('NDWI');
  var ndvi = img.normalizedDifference(['SR_B5','SR_B4']).rename('NDVI');
  var lst  = img.select('ST_B10').subtract(273.15).rename('LST_Celsius');
  return img.addBands([ndsi,ndwi,ndvi,lst]);
}

function addSAR(img) {
  var vvL = ee.Image(10).pow(img.select('VV').divide(10)).rename('VV_lin');
  var vhL = ee.Image(10).pow(img.select('VH').divide(10)).rename('VH_lin');
  return img.addBands([vvL, vhL, vvL.divide(vhL).rename('VV_VH_ratio')]);
}


// ═══════════════════════════════════════════════
// SECTION 3 — GLACIER MASKS
// ═══════════════════════════════════════════════
function addMasks(img) {
  var ndsi = img.select('NDSI');
  var ndwi = img.select('NDWI');
  var bin  = ndsi.gt(0.40).and(ndwi.lt(0.30)).rename('glacier_mask');
  var mc   = ee.Image(0)
               .where(ndsi.gt(0.40), 1)
               .where(ndwi.gt(0.30).and(ndsi.lte(0.40)), 2)
               .where(ndsi.gt(0.10).and(ndsi.lte(0.40)).and(ndwi.lte(0.30)), 3)
               .rename('multiclass_mask');
  return img.addBands([bin, mc]);
}


// ═══════════════════════════════════════════════
// SECTION 4 — BUILD COMPOSITES
// ═══════════════════════════════════════════════
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(AOI).filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_COVER))
  .map(maskS2).map(addIndicesS2).map(addMasks).median().clip(AOI);

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(AOI).filterDate(START_DATE, END_DATE)
  .filter(ee.Filter.eq('instrumentMode','IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'))
  .map(addSAR).median().clip(AOI);

var ls = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(AOI).filterDate(START_DATE,END_DATE)
  .filter(ee.Filter.lt('CLOUD_COVER',CLOUD_COVER))
  .map(maskLandsat)
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterBounds(AOI).filterDate(START_DATE,END_DATE)
    .filter(ee.Filter.lt('CLOUD_COVER',CLOUD_COVER))
    .map(maskLandsat))
  .map(addIndicesLandsat).median().clip(AOI);

var dem     = ee.Image('USGS/SRTMGL1_003').clip(AOI);
var terrain = ee.Terrain.products(dem);

print('S2 bands:', s2.bandNames());
print('Composite built for year:', YEAR);


// ═══════════════════════════════════════════════
// SECTION 5 — VISUALISE ALL LAYERS
// ═══════════════════════════════════════════════
Map.addLayer(s2, {bands:['B4','B3','B2'],min:0,max:0.3},             'S2 RGB');
Map.addLayer(s2.select('NDSI'), {min:-0.5,max:1,palette:['#00429d','#fff','#0094c6']}, 'NDSI', false);
Map.addLayer(s2.select('NDWI'), {min:-0.5,max:1,palette:['#d73027','#fee090','#4575b4']}, 'NDWI', false);
Map.addLayer(s2.select('glacier_mask'), {min:0,max:1,palette:['#2c2c2a','#5DCAA5']}, 'Glacier mask (binary)');
Map.addLayer(s2.select('multiclass_mask'), {min:0,max:3,palette:['#888780','#B5D4F4','#3B8BD4','#FAC775']}, 'Multi-class', false);
Map.addLayer(s1.select('VV'), {min:-25,max:0,palette:['#2c2c2a','#fff']}, 'SAR VV', false);
Map.addLayer(ls.select('LST_Celsius'), {min:-20,max:15,palette:['#313695','#74add1','#fed976','#f46d43','#a50026']}, 'LST °C', false);
Map.addLayer(terrain.select('elevation'), {min:1000,max:7000,palette:['#006633','#99cc33','#cccc00','#996600','#fff']}, 'DEM', false);

// Legend
var legend = ui.Panel({style:{position:'bottom-left',padding:'8px 12px'}});
legend.add(ui.Label({value:'Multi-class mask',style:{fontWeight:'bold',fontSize:'13px'}}));
[['Land/Rock','#888780'],['Snow/Ice','#B5D4F4'],['Water','#3B8BD4'],['Debris ice','#FAC775']].forEach(function(item){
  var row = ui.Panel({layout:ui.Panel.Layout.flow('horizontal')});
  row.add(ui.Label({style:{backgroundColor:item[1],padding:'8px',margin:'0 4px 4px 0'}}));
  row.add(ui.Label(item[0]));
  legend.add(row);
});
Map.add(legend);


// ═══════════════════════════════════════════════
// SECTION 6 — EXPORT SAMPLE CSV TO GOOGLE DRIVE
// (NOTE: Code Editor can only export to Drive)
// For local export, use generate_numerical_dataset.py instead
// ═══════════════════════════════════════════════
var stack = s2.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
                        'NDSI','NDWI','NDVI','BAI','glacier_mask','multiclass_mask'])
  .addBands(s1.select(['VV','VH','VV_lin','VH_lin','VV_VH_ratio']))
  .addBands(ls.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7',
                        'ST_B10','LST_Celsius','NDSI','NDWI','NDVI'])
              .rename(['LS_B2','LS_B3','LS_B4','LS_B5','LS_B6','LS_B7',
                        'LS_Thermal_K','LST_Celsius','LS_NDSI','LS_NDWI','LS_NDVI']))
  .addBands(terrain.select(['elevation','slope','aspect']))
  .addBands(ee.Image.constant(YEAR).int16().rename('year'))
  .addBands(ee.Image.constant(1).uint8().rename('is_melt_season'));

var samples = stack.stratifiedSample({
  numPoints:1500, classBand:'glacier_mask',
  region:AOI, scale:30, seed:42+YEAR, geometries:true, tileScale:4
}).map(function(f){
  return f.set({
    longitude: f.geometry().coordinates().get(0),
    latitude:  f.geometry().coordinates().get(1),
    year:YEAR, season:'melt'
  });
});

// Export to Drive (Code Editor only)
Export.table.toDrive({
  collection: samples,
  description:'glacier_numerical_'+YEAR+'_melt_GEEeditor',
  folder:'GlacierMeltingDataset_FYP',
  fileFormat:'CSV'
});
print('CSV export submitted to Google Drive/GlacierMeltingDataset_FYP/');


// ═══════════════════════════════════════════════
// SECTION 7 — TIME SERIES CHART (2018–2025)
// ═══════════════════════════════════════════════
var refPoint = ee.Geometry.Point([76.5, 34.5]);  // Karakoram reference
var ts = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(refPoint).filterDate('2018-01-01','2025-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
  .map(maskS2)
  .map(function(img){
    return img.normalizedDifference(['B3','B11']).rename('NDSI')
              .set('system:time_start',img.get('system:time_start'));
  });

print(ui.Chart.image.series({
  imageCollection: ts.select('NDSI'),
  region: refPoint, reducer: ee.Reducer.mean(), scale: 30
}).setOptions({
  title:'NDSI time series — Karakoram reference point (2018–2025)',
  hAxis:{title:'Date'}, vAxis:{title:'NDSI',minValue:-1,maxValue:1},
  series:{0:{color:'#1D9E75'}}, lineWidth:1.5, pointSize:3
}));
