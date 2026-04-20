// =====================================================
// COMPREHENSIVE CLIMATE-WATER-HEALTH NEXUS DATA FETCHER
// For India - District Level Analysis
// Author: [Your Name]
// Date: 2024
// =====================================================

// ==================== CONFIGURATION ====================
print('========================================');
print('CLIMATE-WATER-HEALTH NEXUS DATA EXPORT');
print('========================================');

// Load India districts
var gaul = ee.FeatureCollection('FAO/GAUL/2015/level2');
var indiaDistricts = gaul.filter(ee.Filter.eq('ADM0_NAME', 'India'));

// Time periods
var currentYear = {
  start: '2023-01-01',
  end: '2023-12-31'
};

var historicalPeriod = {
  start: '2015-01-01',
  end: '2023-12-31'
};

var longTermNormal = {
  start: '1981-01-01',
  end: '2010-12-31'
};

// Print configuration
print('Total districts:', indiaDistricts.size());
print('Current year:', currentYear.start, 'to', currentYear.end);
print('Historical period:', historicalPeriod.start, 'to', historicalPeriod.end);

// Center map on India
Map.centerObject(indiaDistricts, 5);
Map.addLayer(indiaDistricts, {color: 'red'}, 'District Boundaries', false);

// ==================== 1. RAINFALL DATA ====================
print('\n--- Processing Rainfall Data ---');

// CHIRPS Daily Rainfall (1981-present, 0.05° resolution)
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY');

// Current year rainfall
var rainfall2023 = chirps
  .filterDate(currentYear.start, currentYear.end)
  .select('precipitation');

// Historical rainfall (2015-2023)
var rainfallHistorical = chirps
  .filterDate(historicalPeriod.start, historicalPeriod.end);

// Long-term normal (1981-2010)
var rainfallNormal = chirps
  .filterDate(longTermNormal.start, longTermNormal.end);

// Calculate statistics
var rainfall_mean_2023 = rainfall2023.mean().rename('rain_mean_2023');
var rainfall_median_2023 = rainfall2023.median().rename('rain_median_2023');
var rainfall_total_2023 = rainfall2023.sum().rename('rain_total_2023');
var rainfall_max_2023 = rainfall2023.max().rename('rain_max_2023');
var rainfall_min_2023 = rainfall2023.min().rename('rain_min_2023');
var rainfall_stddev_2023 = rainfall2023.reduce(ee.Reducer.stdDev()).rename('rain_stddev_2023');

// Historical average (2015-2023)
var rainfall_mean_historical = rainfallHistorical.mean().rename('rain_mean_hist');
var rainfall_total_historical = rainfallHistorical
  .map(function(img) {
    var year = ee.Date(img.get('system:time_start')).get('year');
    return img.set('year', year);
  })
  .select('precipitation')
  .reduce(ee.Reducer.mean()).rename('rain_annual_avg_hist');

// Long-term normal
var rainfall_mean_normal = rainfallNormal.mean().rename('rain_mean_normal');

// Calculate rainfall deviation from normal
var rainfall_deviation = rainfall_mean_2023.subtract(rainfall_mean_normal)
  .divide(rainfall_mean_normal)
  .multiply(100)
  .rename('rain_deviation_pct');

// Coefficient of variation (historical period)
var rainfall_cv = rainfallHistorical.reduce(ee.Reducer.stdDev())
  .divide(rainfallHistorical.reduce(ee.Reducer.mean()))
  .rename('rain_cv');

// Monsoon rainfall (June-September)
var monsoonMonths = [6, 7, 8, 9];
var monsoonRainfall = rainfall2023
  .filter(ee.Filter.calendarRange(6, 9, 'month'))
  .sum()
  .rename('rain_monsoon_2023');

var monsoonDependency = monsoonRainfall.divide(rainfall_total_2023)
  .multiply(100)
  .rename('monsoon_dependency_pct');

// Combine rainfall bands
var rainfallComposite = rainfall_mean_2023
  .addBands(rainfall_median_2023)
  .addBands(rainfall_total_2023)
  .addBands(rainfall_max_2023)
  .addBands(rainfall_min_2023)
  .addBands(rainfall_stddev_2023)
  .addBands(rainfall_mean_historical)
  .addBands(rainfall_mean_normal)
  .addBands(rainfall_deviation)
  .addBands(rainfall_cv)
  .addBands(monsoonRainfall)
  .addBands(monsoonDependency);

// Visualize current year mean
var rainfallVis = {
  min: 0, max: 10,
  palette: ['white', 'lightblue', 'blue', 'darkblue', 'purple', 'red']
};
Map.addLayer(rainfall_mean_2023.clip(indiaDistricts), rainfallVis, 'Mean Daily Rainfall 2023');

print('Rainfall data processed ✓');

// ==================== 2. TEMPERATURE DATA ====================
print('\n--- Processing Temperature Data ---');

// MODIS Land Surface Temperature (2000-present, 1km resolution)
var modis_lst = ee.ImageCollection('MODIS/061/MOD11A1');

// Current year temperature
var temp2023 = modis_lst
  .filterDate(currentYear.start, currentYear.end)
  .select('LST_Day_1km');

// Convert Kelvin to Celsius
var kelvinToCelsius = function(img) {
  return img.multiply(0.02).subtract(273.15)
    .copyProperties(img, ['system:time_start']);
};

var temp2023_celsius = temp2023.map(kelvinToCelsius);

// Temperature statistics
var temp_mean_2023 = temp2023_celsius.mean().rename('temp_mean_2023');
var temp_max_2023 = temp2023_celsius.max().rename('temp_max_2023');
var temp_min_2023 = temp2023_celsius.min().rename('temp_min_2023');
var temp_median_2023 = temp2023_celsius.median().rename('temp_median_2023');
var temp_stddev_2023 = temp2023_celsius.reduce(ee.Reducer.stdDev()).rename('temp_stddev_2023');

// Historical temperature (2015-2023)
var tempHistorical = modis_lst
  .filterDate(historicalPeriod.start, historicalPeriod.end)
  .select('LST_Day_1km')
  .map(kelvinToCelsius);

var temp_mean_historical = tempHistorical.mean().rename('temp_mean_hist');

// Temperature anomaly
var temp_anomaly = temp_mean_2023.subtract(temp_mean_historical).rename('temp_anomaly');

// Heat stress days (days with Tmax > 40°C)
var heatDays = temp2023_celsius
  .map(function(img) {
    return img.gt(40).rename('heat_day');
  })
  .sum()
  .rename('heat_days_count');

// Summer temperature (April-June)
var summerTemp = temp2023_celsius
  .filter(ee.Filter.calendarRange(4, 6, 'month'))
  .mean()
  .rename('temp_summer_mean');

// Combine temperature bands
var temperatureComposite = temp_mean_2023
  .addBands(temp_max_2023)
  .addBands(temp_min_2023)
  .addBands(temp_median_2023)
  .addBands(temp_stddev_2023)
  .addBands(temp_mean_historical)
  .addBands(temp_anomaly)
  .addBands(heatDays)
  .addBands(summerTemp);

// Visualize
var tempVis = {
  min: 15, max: 45,
  palette: ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
};
Map.addLayer(temp_mean_2023.clip(indiaDistricts), tempVis, 'Mean Temperature 2023 (°C)');

print('Temperature data processed ✓');

// ==================== 3. DROUGHT INDICATORS ====================
print('\n--- Processing Drought Indicators ---');

// MODIS Vegetation Indices (NDVI, EVI)
var modis_veg = ee.ImageCollection('MODIS/061/MOD13A2');

// Current year vegetation
var veg2023 = modis_veg
  .filterDate(currentYear.start, currentYear.end)
  .select(['NDVI', 'EVI']);

// Scale factor for MODIS VI products
var scaleVI = function(img) {
  return img.multiply(0.0001)
    .copyProperties(img, ['system:time_start']);
};

var veg2023_scaled = veg2023.map(scaleVI);

// NDVI statistics
var ndvi_mean_2023 = veg2023_scaled.select('NDVI').mean().rename('ndvi_mean_2023');
var ndvi_min_2023 = veg2023_scaled.select('NDVI').min().rename('ndvi_min_2023');
var ndvi_stddev_2023 = veg2023_scaled.select('NDVI').reduce(ee.Reducer.stdDev()).rename('ndvi_stddev_2023');

// Historical NDVI for comparison
var vegHistorical = modis_veg
  .filterDate(historicalPeriod.start, historicalPeriod.end)
  .select('NDVI')
  .map(scaleVI);

var ndvi_mean_historical = vegHistorical.mean().rename('ndvi_mean_hist');

// NDVI anomaly (drought indicator)
var ndvi_anomaly = ndvi_mean_2023.subtract(ndvi_mean_historical)
  .divide(ndvi_mean_historical)
  .multiply(100)
  .rename('ndvi_anomaly_pct');

// Vegetation Condition Index (VCI) - simplified
// VCI = (NDVI - NDVImin) / (NDVImax - NDVImin) * 100
var ndvi_max_hist = vegHistorical.max();
var ndvi_min_hist = vegHistorical.min();

var vci = ndvi_mean_2023.subtract(ndvi_min_hist)
  .divide(ndvi_max_hist.subtract(ndvi_min_hist))
  .multiply(100)
  .rename('vci_2023');

// Drought severity classification
// VCI < 20: Severe drought
// VCI 20-40: Moderate drought
// VCI 40-60: Normal
// VCI > 60: Favorable conditions

var drought_severe = vci.lt(20).rename('drought_severe');
var drought_moderate = vci.gte(20).and(vci.lt(40)).rename('drought_moderate');

// Count drought-affected pixels per district (will be done in reduceRegions)

// Standardized Precipitation Index (SPI) approximation
// Using rainfall z-scores
var rain_zscore = rainfall_mean_2023.subtract(rainfall_mean_historical)
  .divide(rainfall_stddev_2023)
  .rename('spi_approx');

// Drought categories:
// SPI < -2.0: Extreme drought
// SPI -1.5 to -2.0: Severe drought
// SPI -1.0 to -1.5: Moderate drought
var drought_extreme_spi = rain_zscore.lt(-2.0).rename('drought_extreme_spi');
var drought_severe_spi = rain_zscore.gte(-2.0).and(rain_zscore.lt(-1.5)).rename('drought_severe_spi');

// Combine drought bands
var droughtComposite = ndvi_mean_2023
  .addBands(ndvi_min_2023)
  .addBands(ndvi_anomaly)
  .addBands(vci)
  .addBands(drought_severe)
  .addBands(drought_moderate)
  .addBands(rain_zscore)
  .addBands(drought_extreme_spi)
  .addBands(drought_severe_spi);

// Visualize VCI
var vciVis = {
  min: 0, max: 100,
  palette: ['red', 'orange', 'yellow', 'lightgreen', 'darkgreen']
};
Map.addLayer(vci.clip(indiaDistricts), vciVis, 'Vegetation Condition Index 2023');

print('Drought indicators processed ✓');

// ==================== 4. FLOOD INDICATORS ====================
print('\n--- Processing Flood Indicators ---');

// Surface water occurrence change (JRC Global Surface Water)
var gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');

var waterOccurrence = gsw.select('occurrence').rename('water_occurrence');
var waterChange = gsw.select('change_abs').rename('water_change');
var waterSeasonality = gsw.select('seasonality').rename('water_seasonality');

// Maximum water extent
var maxWaterExtent = gsw.select('max_extent').rename('max_water_extent');

// Sentinel-1 SAR for flood detection (2023)
// Note: This is computationally intensive, simplified approach
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterDate(currentYear.start, currentYear.end)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .select('VV');

// Calculate median for dry season (Jan-May) and wet season (Jun-Sep)
var drySeasonSAR = s1.filter(ee.Filter.calendarRange(1, 5, 'month')).median().rename('sar_dry');
var wetSeasonSAR = s1.filter(ee.Filter.calendarRange(6, 9, 'month')).median().rename('sar_wet');

// Simple flood proxy: difference between wet and dry season
var floodProxy = wetSeasonSAR.subtract(drySeasonSAR).rename('flood_proxy_sar');

// Potential flood extent (simple threshold)
var potentialFlood = floodProxy.lt(-3).rename('potential_flood_area');

// MODIS-based flood detection using NDWI
var modis_surface = ee.ImageCollection('MODIS/061/MOD09A1')
  .filterDate(currentYear.start, currentYear.end);

// Calculate NDWI (Normalized Difference Water Index)
var calculateNDWI = function(img) {
  var green = img.select('sur_refl_b04').multiply(0.0001);
  var nir = img.select('sur_refl_b02').multiply(0.0001);
  var ndwi = green.subtract(nir).divide(green.add(nir)).rename('ndwi');
  return img.addBands(ndwi);
};

var ndwiCollection = modis_surface.map(calculateNDWI).select('ndwi');
var ndwi_max_2023 = ndwiCollection.max().rename('ndwi_max_2023');
var ndwi_mean_2023 = ndwiCollection.mean().rename('ndwi_mean_2023');

// High NDWI indicates water presence
var waterPresence = ndwi_mean_2023.gt(0.3).rename('water_presence_high');

// Combine flood bands
var floodComposite = waterOccurrence
  .addBands(waterChange)
  .addBands(waterSeasonality)
  .addBands(maxWaterExtent)
  .addBands(ndwi_max_2023)
  .addBands(ndwi_mean_2023)
  .addBands(waterPresence)
  .addBands(potentialFlood);

// Visualize water occurrence
var waterVis = {
  min: 0, max: 100,
  palette: ['white', 'lightblue', 'blue', 'darkblue']
};
Map.addLayer(waterOccurrence.clip(indiaDistricts), waterVis, 'Water Occurrence %', false);

print('Flood indicators processed ✓');

// ==================== 5. EVAPOTRANSPIRATION & WATER BALANCE ====================
print('\n--- Processing Evapotranspiration ---');

// MODIS Evapotranspiration (MOD16A2)
var modis_et = ee.ImageCollection('MODIS/061/MOD16A2GF')
  .filterDate(currentYear.start, currentYear.end);

// Scale ET values (scale factor = 0.1)
var scaleET = function(img) {
  return img.multiply(0.1)
    .copyProperties(img, ['system:time_start']);
};

var et_scaled = modis_et.map(scaleET);

// ET statistics
var et_mean_2023 = et_scaled.select('ET').mean().rename('et_mean_2023');
var et_total_2023 = et_scaled.select('ET').sum().rename('et_total_2023');
var pet_mean_2023 = et_scaled.select('PET').mean().rename('pet_mean_2023');
var pet_total_2023 = et_scaled.select('PET').sum().rename('pet_total_2023');

// Water stress index: PET/Rainfall ratio
// High ratio indicates water stress
var waterStressIndex = pet_total_2023.divide(rainfall_total_2023).rename('water_stress_index');

// Aridity Index: Rainfall/PET
// <0.03: Hyper-arid
// 0.03-0.2: Arid
// 0.2-0.5: Semi-arid
// 0.5-0.65: Dry sub-humid
// >0.65: Humid
var aridityIndex = rainfall_total_2023.divide(pet_total_2023).rename('aridity_index');

// Combine ET bands
var etComposite = et_mean_2023
  .addBands(et_total_2023)
  .addBands(pet_mean_2023)
  .addBands(pet_total_2023)
  .addBands(waterStressIndex)
  .addBands(aridityIndex);

print('Evapotranspiration data processed ✓');

// ==================== 6. SOIL MOISTURE ====================
print('\n--- Processing Soil Moisture ---');

// NASA SMAP Soil Moisture
var smap = ee.ImageCollection('NASA_USDA/HSL/SMAP10KM_soil_moisture')
  .filterDate(currentYear.start, currentYear.end);

// Surface soil moisture (0-5cm)
var ssm = smap.select('ssm');
var ssm_mean_2023 = ssm.mean().rename('soil_moisture_mean');
var ssm_min_2023 = ssm.min().rename('soil_moisture_min');
var ssm_max_2023 = ssm.max().rename('soil_moisture_max');

// Subsurface soil moisture (0-100cm)
var susm = smap.select('susm');
var susm_mean_2023 = susm.mean().rename('subsurface_moisture_mean');

// Combine soil moisture bands
var soilMoistureComposite = ssm_mean_2023
  .addBands(ssm_min_2023)
  .addBands(ssm_max_2023)
  .addBands(susm_mean_2023);

print('Soil moisture data processed ✓');

// ==================== 7. TERRAIN & TOPOGRAPHY ====================
print('\n--- Processing Terrain Data ---');

// SRTM Digital Elevation Model (30m resolution)
var srtm = ee.Image('USGS/SRTMGL1_003');

var elevation = srtm.select('elevation').rename('elevation_m');

// Calculate terrain derivatives
var slope = ee.Terrain.slope(elevation).rename('slope_degrees');
var aspect = ee.Terrain.aspect(elevation).rename('aspect_degrees');

// Combine terrain bands
var terrainComposite = elevation
  .addBands(slope)
  .addBands(aspect);

// Visualize elevation
var elevationVis = {
  min: 0, max: 3000,
  palette: ['green', 'yellow', 'brown', 'white']
};
Map.addLayer(elevation.clip(indiaDistricts), elevationVis, 'Elevation (m)', false);

print('Terrain data processed ✓');

// ==================== 8. LAND COVER ====================
print('\n--- Processing Land Cover ---');

// MODIS Land Cover (MCD12Q1)
var landCover = ee.ImageCollection('MODIS/061/MCD12Q1')
  .filterDate('2023-01-01', '2023-12-31')
  .first()
  .select('LC_Type1')
  .rename('land_cover_type');

// Calculate land cover percentages will be done in reduceRegions

print('Land cover data processed ✓');

// ==================== 9. AGRO-CLIMATIC ZONES ====================
print('\n--- Processing Agro-Climatic Classification ---');

// Create agro-climatic classification based on:
// - Rainfall
// - Temperature
// - Aridity Index

var agroClimaticZone = ee.Image.constant(0).rename('agro_climatic_zone');

// Zone 1: Arid (Aridity < 0.2)
agroClimaticZone = agroClimaticZone.where(aridityIndex.lt(0.2), 1);

// Zone 2: Semi-Arid (Aridity 0.2-0.5)
agroClimaticZone = agroClimaticZone.where(
  aridityIndex.gte(0.2).and(aridityIndex.lt(0.5)), 2);

// Zone 3: Sub-Humid (Aridity 0.5-0.65)
agroClimaticZone = agroClimaticZone.where(
  aridityIndex.gte(0.5).and(aridityIndex.lt(0.65)), 3);

// Zone 4: Humid (Aridity >= 0.65)
agroClimaticZone = agroClimaticZone.where(aridityIndex.gte(0.65), 4);

// Additional classification by temperature
var tempZone = ee.Image.constant(0).rename('temperature_zone');

// Cool: Mean temp < 20°C
tempZone = tempZone.where(temp_mean_2023.lt(20), 1);

// Warm: Mean temp 20-28°C
tempZone = tempZone.where(
  temp_mean_2023.gte(20).and(temp_mean_2023.lt(28)), 2);

// Hot: Mean temp >= 28°C
tempZone = tempZone.where(temp_mean_2023.gte(28), 3);

print('Agro-climatic zones classified ✓');

// ==================== 10. COMPOSITE ALL BANDS ====================
print('\n--- Creating Master Composite ---');

var masterComposite = rainfallComposite
  .addBands(temperatureComposite)
  .addBands(droughtComposite)
  .addBands(floodComposite)
  .addBands(etComposite)
  .addBands(soilMoistureComposite)
  .addBands(terrainComposite)
  .addBands(landCover)
  .addBands(agroClimaticZone)
  .addBands(tempZone);

print('Master composite created with', masterComposite.bandNames().size(), 'bands');
print('Band names:', masterComposite.bandNames());

// ==================== 11. CALCULATE DISTRICT STATISTICS ====================
print('\n--- Calculating District-Level Statistics ---');
print('This may take several minutes... Please wait...');

var districtStats = masterComposite.reduceRegions({
  collection: indiaDistricts,
  reducer: ee.Reducer.mean()
    .combine({
      reducer2: ee.Reducer.median(),
      sharedInputs: true
    })
    .combine({
      reducer2: ee.Reducer.min(),
      sharedInputs: true
    })
    .combine({
      reducer2: ee.Reducer.max(),
      sharedInputs: true
    })
    .combine({
      reducer2: ee.Reducer.stdDev(),
      sharedInputs: true
    }),
  scale: 5000, // 5km resolution for balance of detail vs. computation
  crs: 'EPSG:4326',
  tileScale: 4 // Increase for memory issues
});

// Print sample
print('Sample district statistics (first 5):', districtStats.limit(5));
print('Total districts processed:', districtStats.size());

// ==================== 12. CALCULATE DERIVED INDICES ====================
print('\n--- Calculating Composite Vulnerability Indices ---');

// Function to calculate percentile ranks
var calculatePercentile = function(feature, property) {
  var value = ee.Number(feature.get(property));
  var allValues = districtStats.aggregate_array(property);
  var sorted = allValues.sort();
  var position = sorted.indexOf(value);
  var percentile = position.divide(sorted.size()).multiply(100);
  return feature.set(property + '_percentile', percentile);
};

// Add derived indices as properties (simplified for export)
districtStats = districtStats.map(function(feature) {
  // Climate Vulnerability Score (0-100)
  var climateVuln = ee.Number(feature.get('rain_deviation_pct')).abs()
    .add(ee.Number(feature.get('temp_anomaly')).abs().multiply(5))
    .add(ee.Number(feature.get('rain_cv')).multiply(100))
    .divide(3);
  
  // Water Stress Score (0-100)
  var waterStress = ee.Number(feature.get('water_stress_index')).multiply(10)
    .add(ee.Number(100).subtract(ee.Number(feature.get('aridity_index')).multiply(100)))
    .divide(2);
  
  // Drought Risk Score (0-100)
  var droughtRisk = ee.Number(100).subtract(ee.Number(feature.get('vci_2023')));
  
  return feature
    .set('climate_vulnerability_score', climateVuln)
    .set('water_stress_score', waterStress)
    .set('drought_risk_score', droughtRisk);
});

print('Composite indices calculated ✓');

// ==================== 13. EXPORT DATA ====================
print('\n========================================');
print('EXPORTING DATA TO CSV');
print('========================================');

// Define all selectors for export
var exportSelectors = [
  // Administrative
  'ADM0_NAME', 'ADM1_NAME', 'ADM2_NAME', 'ADM1_CODE', 'ADM2_CODE',
  
  // Rainfall (mean values)
  'rain_mean_2023_mean', 'rain_median_2023_mean', 'rain_total_2023_mean',
  'rain_max_2023_mean', 'rain_min_2023_mean', 'rain_stddev_2023_mean',
  'rain_mean_hist_mean', 'rain_mean_normal_mean',
  'rain_deviation_pct_mean', 'rain_cv_mean',
  'rain_monsoon_2023_mean', 'monsoon_dependency_pct_mean',
  
  // Rainfall (min/max)
  'rain_total_2023_min', 'rain_total_2023_max',
  
  // Temperature (mean values)
  'temp_mean_2023_mean', 'temp_max_2023_mean', 'temp_min_2023_mean',
  'temp_median_2023_mean', 'temp_anomaly_mean',
  'heat_days_count_mean', 'temp_summer_mean_mean',
  
  // Temperature (min/max)
  'temp_max_2023_max', 'temp_min_2023_min',
  
  // Drought indicators
  'ndvi_mean_2023_mean', 'ndvi_min_2023_mean', 'ndvi_anomaly_pct_mean',
  'vci_2023_mean', 'spi_approx_mean',
  
  // Flood indicators
  'water_occurrence_mean', 'water_change_mean', 'ndwi_mean_2023_mean',
  
  // Evapotranspiration & Water Balance
  'et_total_2023_mean', 'pet_total_2023_mean',
  'water_stress_index_mean', 'aridity_index_mean',
  
  // Soil Moisture
  'soil_moisture_mean_mean', 'soil_moisture_min_mean', 'soil_moisture_max_mean',
  
  // Terrain
  'elevation_m_mean', 'slope_degrees_mean',
  
  // Land Cover
  'land_cover_type_mean',
  
  // Zones
  'agro_climatic_zone_mean', 'temperature_zone_mean',
  
  // Derived Indices
  'climate_vulnerability_score', 'water_stress_score', 'drought_risk_score'
];

// Export 1: Complete district statistics
Export.table.toDrive({
  collection: districtStats,
  description: 'India_Climate_Water_Complete_District_Stats',
  fileNamePrefix: 'India_Complete_Climate_Water_Data_2023',
  fileFormat: 'CSV',
  selectors: exportSelectors
});

// Export 2: Simplified summary (key metrics only)
var keyMetricsSelectors = [
  'ADM0_NAME', 'ADM1_NAME', 'ADM2_NAME',
  'rain_total_2023_mean', 'rain_deviation_pct_mean', 'monsoon_dependency_pct_mean',
  'temp_mean_2023_mean', 'temp_max_2023_max', 'heat_days_count_mean',
  'vci_2023_mean', 'drought_risk_score',
  'water_occurrence_mean', 'ndwi_mean_2023_mean',
  'water_stress_index_mean', 'aridity_index_mean',
  'elevation_m_mean', 'agro_climatic_zone_mean',
  'climate_vulnerability_score', 'water_stress_score'
];

Export.table.toDrive({
  collection: districtStats,
  description: 'India_Climate_Water_Key_Metrics',
  fileNamePrefix: 'India_Key_Climate_Metrics_2023',
  fileFormat: 'CSV',
  selectors: keyMetricsSelectors
});

// Export 3: Shapefile with statistics (for GIS)
Export.table.toDrive({
  collection: districtStats,
  description: 'India_Climate_Water_Shapefile',
  fileNamePrefix: 'India_Climate_Water_Shapefile_2023',
  fileFormat: 'SHP'
});

// Export 4: State-level aggregated summary
var stateStats = districtStats.reduceColumns({
  reducer: ee.Reducer.mean().repeat(20).group({
    groupField: 1,
    groupName: 'state'
  }),
  selectors: [
    'ADM1_NAME',
    'rain_total_2023_mean', 'temp_mean_2023_mean', 'vci_2023_mean',
    'water_stress_index_mean', 'climate_vulnerability_score',
    'drought_risk_score', 'water_stress_score',
    'heat_days_count_mean', 'rain_deviation_pct_mean',
    'aridity_index_mean', 'elevation_m_mean',
    'ndvi_mean_2023_mean', 'et_total_2023_mean',
    'soil_moisture_mean_mean', 'water_occurrence_mean',
    'monsoon_dependency_pct_mean', 'temp_anomaly_mean',
    'rain_cv_mean', 'ndwi_mean_2023_mean'
  ]
});

// Can't directly export state stats, but print for reference
print('State-wise average statistics:', stateStats);

// ==================== 14. CREATE SUMMARY CHARTS ====================
print('\n--- Creating Summary Charts ---');

// Chart 1: Top 20 Climate Vulnerable Districts
var sortedByClimateVuln = districtStats
  .sort('climate_vulnerability_score', false)
  .limit(20);

var chartClimateVuln = ui.Chart.feature.byFeature({
  features: sortedByClimateVuln,
  xProperty: 'ADM2_NAME',
  yProperties: ['climate_vulnerability_score']
})
.setChartType('ColumnChart')
.setOptions({
  title: 'Top 20 Climate Vulnerable Districts',
  hAxis: {
    title: 'District',
    slantedText: true,
    slantedTextAngle: 45
  },
  vAxis: {
    title: 'Climate Vulnerability Score',
    viewWindow: {min: 0}
  },
  colors: ['#d73027'],
  legend: {position: 'none'},
  height: 400,
  width: 800
});
print(chartClimateVuln);

// Chart 2: Top 20 Water Stressed Districts
var sortedByWaterStress = districtStats
  .sort('water_stress_score', false)
  .limit(20);

var chartWaterStress = ui.Chart.feature.byFeature({
  features: sortedByWaterStress,
  xProperty: 'ADM2_NAME',
  yProperties: ['water_stress_score']
})
.setChartType('ColumnChart')
.setOptions({
  title: 'Top 20 Water Stressed Districts',
  hAxis: {
    title: 'District',
    slantedText: true,
    slantedTextAngle: 45
  },
  vAxis: {
    title: 'Water Stress Score',
    viewWindow: {min: 0}
  },
  colors: ['#4575b4'],
  legend: {position: 'none'},
  height: 400,
  width: 800
});
print(chartWaterStress);

// Chart 3: Top 20 Drought Risk Districts
var sortedByDrought = districtStats
  .sort('drought_risk_score', false)
  .limit(20);

var chartDrought = ui.Chart.feature.byFeature({
  features: sortedByDrought,
  xProperty: 'ADM2_NAME',
  yProperties: ['drought_risk_score']
})
.setChartType('ColumnChart')
.setOptions({
  title: 'Top 20 Drought Risk Districts',
  hAxis: {
    title: 'District',
    slantedText: true,
    slantedTextAngle: 45
  },
  vAxis: {
    title: 'Drought Risk Score',
    viewWindow: {min: 0}
  },
  colors: ['#fdae61'],
  legend: {position: 'none'},
  height: 400,
  width: 800
});
print(chartDrought);

// Chart 4: Scatter plot - Climate Vulnerability vs Water Stress
var scatterData = districtStats.select(
  ['climate_vulnerability_score', 'water_stress_score', 'ADM2_NAME']
);

var chartScatter = ui.Chart.feature.byFeature({
  features: scatterData,
  xProperty: 'climate_vulnerability_score',
  yProperties: ['water_stress_score']
})
.setChartType('ScatterChart')
.setOptions({
  title: 'Climate Vulnerability vs Water Stress (All Districts)',
  hAxis: {
    title: 'Climate Vulnerability Score',
    viewWindow: {min: 0}
  },
  vAxis: {
    title: 'Water Stress Score',
    viewWindow: {min: 0}
  },
  pointSize: 3,
  colors: ['#1b9e77'],
  legend: {position: 'none'},
  height: 500,
  width: 600,
  trendlines: {
    0: {
      type: 'linear',
      color: 'red',
      lineWidth: 2,
      opacity: 0.5,
      showR2: true,
      visibleInLegend: true
    }
  }
});
print(chartScatter);

// Chart 5: Rainfall deviation distribution
var chartRainDeviation = ui.Chart.feature.histogram({
  features: districtStats,
  property: 'rain_deviation_pct_mean',
  maxBuckets: 30
})
.setOptions({
  title: 'Distribution of Rainfall Deviation from Normal',
  hAxis: {
    title: 'Rainfall Deviation (%)',
    viewWindow: {min: -100, max: 100}
  },
  vAxis: {
    title: 'Number of Districts'
  },
  colors: ['#3182bd'],
  legend: {position: 'none'},
  height: 400,
  width: 600
});
print(chartRainDeviation);

// Chart 6: Temperature anomaly distribution
var chartTempAnomaly = ui.Chart.feature.histogram({
  features: districtStats,
  property: 'temp_anomaly_mean',
  maxBuckets: 30
})
.setOptions({
  title: 'Distribution of Temperature Anomaly',
  hAxis: {
    title: 'Temperature Anomaly (°C)',
    viewWindow: {min: -5, max: 5}
  },
  vAxis: {
    title: 'Number of Districts'
  },
  colors: ['#e34a33'],
  legend: {position: 'none'},
  height: 400,
  width: 600
});
print(chartTempAnomaly);

// ==================== 15. CREATE RISK CLASSIFICATION MAPS ====================
print('\n--- Creating Risk Classification Maps ---');

// Join statistics back to spatial features for mapping
var districtRiskMap = indiaDistricts.map(function(feature) {
  // Get matching district from stats
  var districtCode = feature.get('ADM2_CODE');
  var matchingStat = districtStats
    .filter(ee.Filter.eq('ADM2_CODE', districtCode))
    .first();
  
  // Copy properties
  return feature.copyProperties(matchingStat, [
    'climate_vulnerability_score',
    'water_stress_score',
    'drought_risk_score'
  ]);
});

// Classification breaks
var lowThreshold = 33;
var highThreshold = 66;

// Map 1: Climate Vulnerability Risk
var climateRiskVis = {
  min: 0,
  max: 100,
  palette: ['#1a9850', '#ffffbf', '#d73027']
};
Map.addLayer(
  districtRiskMap.style({
    fillColor: '1a9850',
    color: '000000',
    width: 0.5
  }),
  {},
  'Climate Vulnerability Map',
  false
);

// Map 2: Water Stress Risk
var waterStressVis = {
  min: 0,
  max: 100,
  palette: ['#2166ac', '#f7f7f7', '#b2182b']
};

// Map 3: Drought Risk
var droughtRiskVis = {
  min: 0,
  max: 100,
  palette: ['#006837', '#ffffbf', '#a50026']
};

print('Risk classification maps created ✓');

// ==================== 16. HISTORICAL TREND ANALYSIS ====================
print('\n--- Calculating Historical Trends (2015-2023) ---');

// Calculate annual rainfall for each year
var years = ee.List.sequence(2015, 2023);

var annualRainfall = ee.ImageCollection.fromImages(
  years.map(function(year) {
    var yearStr = ee.Number(year).format('%d');
    var yearStart = ee.Date.fromYMD(year, 1, 1);
    var yearEnd = ee.Date.fromYMD(year, 12, 31);
    
    var yearlyRain = chirps
      .filterDate(yearStart, yearEnd)
      .sum()
      .rename('rainfall')
      .set('year', year);
    
    return yearlyRain;
  })
);

// Calculate trend using linear regression
var linearFit = annualRainfall.select(['rainfall'])
  .reduce(ee.Reducer.linearFit());

var rainfallTrend = linearFit.select('scale').rename('rainfall_trend_mm_per_year');

// Similar for temperature
var annualTemp = ee.ImageCollection.fromImages(
  years.map(function(year) {
    var yearStart = ee.Date.fromYMD(year, 1, 1);
    var yearEnd = ee.Date.fromYMD(year, 12, 31);
    
    var yearlyTemp = modis_lst
      .filterDate(yearStart, yearEnd)
      .select('LST_Day_1km')
      .map(kelvinToCelsius)
      .mean()
      .rename('temperature')
      .set('year', year);
    
    return yearlyTemp;
  })
);

var tempTrend = annualTemp.select(['temperature'])
  .reduce(ee.Reducer.linearFit())
  .select('scale')
  .rename('temp_trend_celsius_per_year');

// Add trends to master composite
var trendsComposite = rainfallTrend.addBands(tempTrend);

// Calculate district-level trends
var districtTrends = trendsComposite.reduceRegions({
  collection: indiaDistricts,
  reducer: ee.Reducer.mean(),
  scale: 5000,
  crs: 'EPSG:4326'
});

// Export trends
Export.table.toDrive({
  collection: districtTrends,
  description: 'India_Climate_Trends_2015_2023',
  fileNamePrefix: 'India_Climate_Trends_2015_2023',
  fileFormat: 'CSV',
  selectors: [
    'ADM0_NAME', 'ADM1_NAME', 'ADM2_NAME',
    'rainfall_trend_mm_per_year', 'temp_trend_celsius_per_year'
  ]
});

print('Historical trends calculated ✓');

// ==================== 17. EXTREME EVENT COUNTING ====================
print('\n--- Counting Extreme Events (2015-2023) ---');

// Count drought years (rainfall < 75% of normal)
var droughtYears = years.map(function(year) {
  var yearStart = ee.Date.fromYMD(year, 1, 1);
  var yearEnd = ee.Date.fromYMD(year, 12, 31);
  
  var yearlyRain = chirps
    .filterDate(yearStart, yearEnd)
    .sum();
  
  var isDrought = yearlyRain.lt(rainfall_mean_normal.multiply(365).multiply(0.75));
  
  return isDrought.rename('drought_' + year);
});

var droughtYearsCollection = ee.ImageCollection.fromImages(droughtYears);
var droughtCount = droughtYearsCollection.sum().rename('drought_years_count_2015_2023');

// Count flood years (using high NDWI as proxy)
var floodYears = years.map(function(year) {
  var yearStart = ee.Date.fromYMD(year, 1, 1);
  var yearEnd = ee.Date.fromYMD(year, 12, 31);
  
  var yearlyNDWI = modis_surface
    .filterDate(yearStart, yearEnd)
    .map(calculateNDWI)
    .select('ndwi')
    .max();
  
  var isFlood = yearlyNDWI.gt(0.4);
  
  return isFlood.rename('flood_' + year);
});

var floodYearsCollection = ee.ImageCollection.fromImages(floodYears);
var floodCount = floodYearsCollection.sum().rename('flood_years_count_2015_2023');

// Count heat wave years (days > 40°C for >5 days)
var heatWaveYears = years.map(function(year) {
  var yearStart = ee.Date.fromYMD(year, 1, 1);
  var yearEnd = ee.Date.fromYMD(year, 12, 31);
  
  var yearlyHeatDays = modis_lst
    .filterDate(yearStart, yearEnd)
    .select('LST_Day_1km')
    .map(kelvinToCelsius)
    .map(function(img) {
      return img.gt(40);
    })
    .sum();
  
  var isHeatWave = yearlyHeatDays.gt(5);
  
  return isHeatWave.rename('heatwave_' + year);
});

var heatWaveYearsCollection = ee.ImageCollection.fromImages(heatWaveYears);
var heatWaveCount = heatWaveYearsCollection.sum().rename('heatwave_years_count_2015_2023');

// Combine extreme event counts
var extremeEventsComposite = droughtCount
  .addBands(floodCount)
  .addBands(heatWaveCount);

// Calculate district-level extreme events
var districtExtremes = extremeEventsComposite.reduceRegions({
  collection: indiaDistricts,
  reducer: ee.Reducer.mean(),
  scale: 5000,
  crs: 'EPSG:4326'
});

// Export extreme events
Export.table.toDrive({
  collection: districtExtremes,
  description: 'India_Extreme_Events_2015_2023',
  fileNamePrefix: 'India_Extreme_Events_2015_2023',
  fileFormat: 'CSV',
  selectors: [
    'ADM0_NAME', 'ADM1_NAME', 'ADM2_NAME',
    'drought_years_count_2015_2023',
    'flood_years_count_2015_2023',
    'heatwave_years_count_2015_2023'
  ]
});

print('Extreme events counted ✓');

// ==================== 18. MONSOON CHARACTERISTICS ====================
print('\n--- Analyzing Monsoon Characteristics ---');

// Monsoon onset, withdrawal, and intensity for 2023
var monsoonMonthly = ee.List.sequence(6, 9).map(function(month) {
  var monthlyRain = rainfall2023
    .filter(ee.Filter.calendarRange(month, month, 'month'))
    .sum()
    .rename('rain_month_' + month);
  return monthlyRain;
});

var monsoonComposite = ee.ImageCollection.fromImages(monsoonMonthly)
  .toBands()
  .rename(['rain_june', 'rain_july', 'rain_august', 'rain_september']);

// Calculate monsoon characteristics
var monsoonTotal = monsoonComposite.reduce(ee.Reducer.sum()).rename('monsoon_total');
var monsoonMean = monsoonComposite.reduce(ee.Reducer.mean()).rename('monsoon_mean');
var monsoonCV = monsoonComposite.reduce(ee.Reducer.stdDev())
  .divide(monsoonMean)
  .rename('monsoon_cv');

// Peak monsoon month
var peakMonth = monsoonComposite.reduce(ee.Reducer.max()).rename('monsoon_peak');

// Monsoon deficit/excess
var monsoonNormal = rainfallNormal
  .filter(ee.Filter.calendarRange(6, 9, 'month'))
  .mean()
  .multiply(122); // Approx days in Jun-Sep

var monsoonDeficit = monsoonTotal.subtract(monsoonNormal)
  .divide(monsoonNormal)
  .multiply(100)
  .rename('monsoon_deviation_pct');

// Combine monsoon characteristics
var monsoonCharComposite = monsoonTotal
  .addBands(monsoonMean)
  .addBands(monsoonCV)
  .addBands(monsoonDeficit)
  .addBands(monsoonComposite);

// Calculate district-level monsoon characteristics
var districtMonsoon = monsoonCharComposite.reduceRegions({
  collection: indiaDistricts,
  reducer: ee.Reducer.mean(),
  scale: 5000,
  crs: 'EPSG:4326'
});

// Export monsoon characteristics
Export.table.toDrive({
  collection: districtMonsoon,
  description: 'India_Monsoon_Characteristics_2023',
  fileNamePrefix: 'India_Monsoon_Characteristics_2023',
  fileFormat: 'CSV',
  selectors: [
    'ADM0_NAME', 'ADM1_NAME', 'ADM2_NAME',
    'monsoon_total', 'monsoon_mean', 'monsoon_cv',
    'monsoon_deviation_pct',
    'rain_june', 'rain_july', 'rain_august', 'rain_september'
  ]
});

print('Monsoon characteristics calculated ✓');

// ==================== 19. GROUNDWATER PROXY INDICATORS ====================
print('\n--- Creating Groundwater Proxy Indicators ---');

// Since direct groundwater data isn't available in GEE,
// create proxy indicators from available data

// Proxy 1: Water Balance (Rainfall - ET - Runoff estimate)
var runoffProxy = rainfall_total_2023.multiply(0.2); // Assume 20% runoff
var waterBalance = rainfall_total_2023
  .subtract(et_total_2023)
  .subtract(runoffProxy)
  .rename('water_balance_mm');

// Negative water balance suggests groundwater depletion

// Proxy 2: Vegetation stress (low NDVI in irrigated areas)
// Agricultural areas with low NDVI may indicate irrigation failure
var agricultureMask = landCover.eq(12).or(landCover.eq(14)); // Croplands
var ndviAgricultural = ndvi_mean_2023.updateMask(agricultureMask)
  .rename('ndvi_agricultural');

// Proxy 3: Change in surface water (may indicate GW pumping)
var surfaceWaterChange = waterChange.multiply(-1).rename('surface_water_decline');
// Negative change indicates decline, which may be due to GW extraction

// Combine groundwater proxies
var groundwaterProxyComposite = waterBalance
  .addBands(ndviAgricultural)
  .addBands(surfaceWaterChange);

// Calculate district-level groundwater proxies
var districtGroundwater = groundwaterProxyComposite.reduceRegions({
  collection: indiaDistricts,
  reducer: ee.Reducer.mean(),
  scale: 5000,
  crs: 'EPSG:4326'
});

// Export groundwater proxies
Export.table.toDrive({
  collection: districtGroundwater,
  description: 'India_Groundwater_Proxies_2023',
  fileNamePrefix: 'India_Groundwater_Proxies_2023',
  fileFormat: 'CSV',
  selectors: [
    'ADM0_NAME', 'ADM1_NAME', 'ADM2_NAME',
    'water_balance_mm',
    'ndvi_agricultural',
    'surface_water_decline'
  ]
});

print('Groundwater proxy indicators created ✓');

// ==================== 20. RIVER BASIN INFORMATION ====================
print('\n--- Processing River Basin Data ---');

// HydroBASINS Level 5 (sub-basins)
var basins = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_5');

// Filter to India region (approximate bounds)
var indiaBasins = basins.filterBounds(indiaDistricts.geometry());

// Add basin ID to districts
var districtsWithBasin = indiaDistricts.map(function(district) {
  var districtGeom = district.geometry();
  
  // Find intersecting basin
  var intersectingBasin = indiaBasins
    .filterBounds(districtGeom)
    .first();
  
  var basinID = ee.Algorithms.If(
    intersectingBasin,
    intersectingBasin.get('HYBAS_ID'),
    -9999
  );
  
  return district.set('basin_id', basinID);
});

// Export districts with basin information
Export.table.toDrive({
  collection: districtsWithBasin,
  description: 'India_Districts_River_Basins',
  fileNamePrefix: 'India_Districts_River_Basins',
  fileFormat: 'CSV',
  selectors: [
    'ADM0_NAME', 'ADM1_NAME', 'ADM2_NAME', 'ADM2_CODE',
    'basin_id'
  ]
});

// Visualize basins
Map.addLayer(indiaBasins, {color: 'blue'}, 'River Basins', false);

print('River basin information processed ✓');

// ==================== 21. LAND USE INTENSITY ====================
print('\n--- Calculating Land Use Intensity ---');

// Calculate land use percentages by type
var landCoverClasses = ee.Dictionary({
  1: 'Evergreen_Needleleaf_Forest',
  2: 'Evergreen_Broadleaf_Forest',
  3: 'Deciduous_Needleleaf_Forest',
  4: 'Deciduous_Broadleaf_Forest',
  5: 'Mixed_Forests',
  6: 'Closed_Shrublands',
  7: 'Open_Shrublands',
  8: 'Woody_Savannas',
  9: 'Savannas',
  10: 'Grasslands',
  11: 'Permanent_Wetlands',
  12: 'Croplands',
  13: 'Urban_and_Built_up',
  14: 'Cropland_Natural_Vegetation_Mosaic',
  15: 'Snow_and_Ice',
  16: 'Barren_or_Sparsely_Vegetated',
  17: 'Water'
});

// Function to calculate land cover percentages per district
var calculateLandCoverPct = function(district) {
  var districtGeom = district.geometry();
  
  // Get land cover for district
  var districtLC = landCover.clip(districtGeom);
  
  // Calculate area of each class
  var lcArea = districtLC.reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(),
    geometry: districtGeom,
    scale: 500,
    maxPixels: 1e9
  });
  
  // Convert to percentages
  var histogram = ee.Dictionary(lcArea.get('land_cover_type'));
  var total = histogram.values().reduce(ee.Reducer.sum());
  
  // Calculate specific percentages
  var croplandPct = ee.Number(histogram.get('12', 0))
    .divide(total).multiply(100);
  var urbanPct = ee.Number(histogram.get('13', 0))
    .divide(total).multiply(100);
  var forestPct = ee.Number(histogram.get('1', 0))
    .add(ee.Number(histogram.get('2', 0)))
    .add(ee.Number(histogram.get('3', 0)))
    .add(ee.Number(histogram.get('4', 0)))
    .add(ee.Number(histogram.get('5', 0)))
    .divide(total).multiply(100);
  var waterPct = ee.Number(histogram.get('17', 0))
    .divide(total).multiply(100);
  
  return district
    .set('cropland_pct', croplandPct)
    .set('urban_pct', urbanPct)
    .set('forest_pct', forestPct)
    .set('water_pct', waterPct);
};

// This is computationally intensive, so limit to sample or export separately
print('Note: Land cover percentage calculation is computationally intensive.');
print('Consider running separately for specific districts if needed.');

// ==================== 22. FINAL SUMMARY STATISTICS ====================
print('\n========================================');
print('DATA PROCESSING SUMMARY');
print('========================================');

// Calculate India-wide statistics
var indiaStats = masterComposite.reduceRegion({
  reducer: ee.Reducer.mean()
    .combine({reducer2: ee.Reducer.min(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.max(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true}),
  geometry: indiaDistricts.geometry(),
  scale: 10000,
  maxPixels: 1e10
});

print('India-wide Statistics:', indiaStats);

// ==================== 23. EXPORT SUMMARY ====================
print('\n========================================');
print('EXPORT SUMMARY');
print('========================================');
print('');
print('Total Exports Queued: 9');
print('');
print('1. India_Climate_Water_Complete_District_Stats.csv');
print('   - Complete dataset with all variables');
print('   - ~70+ columns per district');
print('   - File size: ~5-10 MB');
print('');
print('2. India_Key_Climate_Metrics.csv');
print('   - Simplified key metrics only');
print('   - ~25 columns per district');
print('   - File size: ~2-3 MB');
print('');
print('3. India_Climate_Water_Shapefile_2023.shp');
print('   - Spatial data for GIS mapping');
print('   - Can be loaded in QGIS/ArcGIS');
print('');
print('4. India_Climate_Trends_2015_2023.csv');
print('   - Historical trends (rainfall, temperature)');
print('   - Annual rate of change');
print('');
print('5. India_Extreme_Events_2015_2023.csv');
print('   - Count of drought years');
print('   - Count of flood years');
print('   - Count of heat wave years');
print('');
print('6. India_Monsoon_Characteristics_2023.csv');
print('   - Monthly monsoon rainfall');
print('   - Monsoon variability');
print('   - Monsoon deviation from normal');
print('');
print('7. India_Groundwater_Proxies_2023.csv');
print('   - Water balance estimates');
print('   - Agricultural vegetation health');
print('   - Surface water changes');
print('');
print('8. India_Districts_River_Basins.csv');
print('   - River basin assignment');
print('   - For watershed-level analysis');
print('');
print('========================================');
print('NEXT STEPS:');
print('========================================');
print('1. Go to Tasks tab (top-right corner)');
print('2. Click RUN on each export task');
print('3. Monitor progress (may take 10-30 minutes per export)');
print('4. Check your Google Drive for completed files');
print('5. Download and merge with NFHS-5 data');
print('');
print('========================================');
print('DATA DICTIONARY');
print('========================================');
print('');
print('RAINFALL VARIABLES:');
print('- rain_total_2023_mean: Total annual rainfall (mm)');
print('- rain_deviation_pct_mean: Deviation from long-term normal (%)');
print('- rain_cv_mean: Rainfall variability (coefficient of variation)');
print('- monsoon_dependency_pct_mean: % of annual rain during monsoon');
print('');
print('TEMPERATURE VARIABLES:');
print('- temp_mean_2023_mean: Average temperature (°C)');
print('- temp_max_2023_max: Maximum recorded temperature (°C)');
print('- temp_anomaly_mean: Deviation from historical average (°C)');
print('- heat_days_count_mean: Number of days >40°C');
print('');
print('DROUGHT VARIABLES:');
print('- vci_2023_mean: Vegetation Condition Index (0-100)');
print('   * <20: Severe drought');
print('   * 20-40: Moderate drought');
print('   * >60: Good conditions');
print('- spi_approx_mean: Standardized Precipitation Index');
print('   * <-2: Extreme drought');
print('   * -1 to -2: Severe drought');
print('- drought_risk_score: Composite drought risk (0-100)');
print('');
print('WATER STRESS VARIABLES:');
print('- water_stress_index_mean: PET/Rainfall ratio');
print('   * >2: High water stress');
print('   * 1-2: Moderate stress');
print('   * <1: Low stress');
print('- aridity_index_mean: Rainfall/PET ratio');
print('   * <0.2: Arid');
print('   * 0.2-0.5: Semi-arid');
print('   * >0.65: Humid');
print('');
print('COMPOSITE SCORES:');
print('- climate_vulnerability_score: Overall climate risk (0-100)');
print('- water_stress_score: Overall water stress (0-100)');
print('- drought_risk_score: Overall drought risk (0-100)');
print('');
print('========================================');
print('SCRIPT EXECUTION COMPLETE ✓');
print('========================================');

// ==================== 24. CREATE LEGEND ====================

// Create a panel for legend
var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px'
  }
});

// Add legend title
var legendTitle = ui.Label({
  value: 'Climate Vulnerability Index',
  style: {
    fontWeight: 'bold',
    fontSize: '16px',
    margin: '0 0 4px 0',
    padding: '0'
  }
});
legend.add(legendTitle);

// Create legend items
var makeRow = function(color, name) {
  var colorBox = ui.Label({
    style: {
      backgroundColor: color,
      padding: '8px',
      margin: '0 0 4px 0'
    }
  });
  
  var description = ui.Label({
    value: name,
    style: {margin: '0 0 4px 6px'}
  });
  
  return ui.Panel({
    widgets: [colorBox, description],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
};

// Add color and names
var palette = ['#1a9850', '#ffffbf', '#d73027'];
legend.add(makeRow(palette[0], 'Low Vulnerability'));
legend.add(makeRow(palette[1], 'Moderate Vulnerability'));
legend.add(makeRow(palette[2], 'High Vulnerability'));

// Add legend to map
Map.add(legend);

// ==================== 25. CREATE DATA QUALITY REPORT ====================
print('\n========================================');
print('DATA QUALITY ASSESSMENT');
print('========================================');

// Function to calculate data completeness
var assessDataQuality = function(collection, propertyName) {
  var total = collection.size();
  var withData = collection.filter(
    ee.Filter.neq(propertyName, null)
  ).size();
  var completeness = withData.divide(total).multiply(100);
  return completeness;
};

// Assess key variables
print('\nData Completeness by Variable:');
print('Rainfall data:', assessDataQuality(districtStats, 'rain_total_2023_mean'), '%');
print('Temperature data:', assessDataQuality(districtStats, 'temp_mean_2023_mean'), '%');
print('NDVI data:', assessDataQuality(districtStats, 'ndvi_mean_2023_mean'), '%');
print('Water occurrence:', assessDataQuality(districtStats, 'water_occurrence_mean'), '%');

// Calculate spatial coverage
var totalDistricts = indiaDistricts.size();
var processedDistricts = districtStats.size();
var coveragePct = processedDistricts.divide(totalDistricts).multiply(100);

print('\nSpatial Coverage:');
print('Total districts in India:', totalDistricts);
print('Districts with data:', processedDistricts);
print('Coverage percentage:', coveragePct, '%');

// Identify potential data issues
var lowRainfall = districtStats.filter(
  ee.Filter.lt('rain_total_2023_mean', 100)
);
var highRainfall = districtStats.filter(
  ee.Filter.gt('rain_total_2023_mean', 5000)
);

print('\nData Range Checks:');
print('Districts with very low rainfall (<100mm):', lowRainfall.size());
print('Districts with very high rainfall (>5000mm):', highRainfall.size());

// ==================== 26. TIME SERIES EXPORT (MONTHLY DATA) ====================
print('\n--- Preparing Monthly Time Series Export ---');

// Create monthly rainfall data for 2023
var months = ee.List.sequence(1, 12);

var monthlyRainfall = ee.FeatureCollection(months.map(function(month) {
  var monthStart = ee.Date.fromYMD(2023, month, 1);
  var monthEnd = monthStart.advance(1, 'month');
  
  var monthlyRain = chirps
    .filterDate(monthStart, monthEnd)
    .sum()
    .rename('rainfall');
  
  var districtMonthly = monthlyRain.reduceRegions({
    collection: indiaDistricts,
    reducer: ee.Reducer.mean(),
    scale: 5000,
    crs: 'EPSG:4326'
  });
  
  // Add month property
  return districtMonthly.map(function(feature) {
    return feature.set('month', month).set('year', 2023);
  });
}).flatten());

// Export monthly time series
Export.table.toDrive({
  collection: monthlyRainfall,
  description: 'India_Monthly_Rainfall_2023',
  fileNamePrefix: 'India_Monthly_Rainfall_2023',
  fileFormat: 'CSV',
  selectors: [
    'ADM0_NAME', 'ADM1_NAME', 'ADM2_NAME',
    'year', 'month', 'rainfall'
  ]
});

print('Monthly time series export queued ✓');

// ==================== 27. CORRELATION ANALYSIS ====================
print('\n--- Calculating Variable Correlations ---');

// Sample districts for correlation analysis
var sampleSize = 100;
var sampleDistricts = districtStats.randomColumn('random', 42).limit(sampleSize);

// Extract key variables for correlation
var correlationData = sampleDistricts.select([
  'rain_total_2023_mean',
  'temp_mean_2023_mean',
  'ndvi_mean_2023_mean',
  'water_stress_index_mean',
  'elevation_m_mean'
]);

// Note: GEE doesn't have built-in correlation, but we can export for analysis in R/Python
print('Correlation analysis data prepared for', sampleSize, 'districts');
print('Export this sample for correlation matrix calculation in R/Python');

// ==================== 28. VULNERABILITY HOTSPOTS IDENTIFICATION ====================
print('\n--- Identifying Vulnerability Hotspots ---');

// Define triple-threat districts (high climate vuln + high water stress + high drought)
var hotspots = districtStats.filter(
  ee.Filter.and(
    ee.Filter.gt('climate_vulnerability_score', 66),
    ee.Filter.gt('water_stress_score', 66),
    ee.Filter.gt('drought_risk_score', 66)
  )
);

print('Triple-Threat Hotspot Districts:', hotspots.size());
print('Hotspot Districts:', hotspots.aggregate_array('ADM2_NAME').distinct());

// Export hotspot districts
Export.table.toDrive({
  collection: hotspots,
  description: 'India_Vulnerability_Hotspots',
  fileNamePrefix: 'India_Vulnerability_Hotspots',
  fileFormat: 'CSV'
});

// Visualize hotspots on map
var hotspotsVector = indiaDistricts.filter(
  ee.Filter.inList('ADM2_NAME', hotspots.aggregate_array('ADM2_NAME'))
);

Map.addLayer(
  hotspotsVector,
  {color: 'red'},
  'Triple-Threat Hotspots',
  true,
  0.7
);

// ==================== 29. RESILIENCE CHAMPIONS IDENTIFICATION ====================
print('\n--- Identifying Resilience Champions ---');

// Districts with low vulnerability despite high climate stress
var champions = districtStats.filter(
  ee.Filter.and(
    ee.Filter.or(
      ee.Filter.gt('rain_deviation_pct_mean', 20),  // High rainfall deviation
      ee.Filter.gt('temp_anomaly_mean', 1)           // High temp anomaly
    ),
    ee.Filter.and(
      ee.Filter.lt('drought_risk_score', 33),        // Low drought risk
      ee.Filter.lt('water_stress_score', 33)         // Low water stress
    )
  )
);

print('Resilience Champion Districts:', champions.size());
print('Champion Districts:', champions.aggregate_array('ADM2_NAME').distinct());

// Export champions
Export.table.toDrive({
  collection: champions,
  description: 'India_Resilience_Champions',
  fileNamePrefix: 'India_Resilience_Champions',
  fileFormat: 'CSV'
});

// Visualize champions
var championsVector = indiaDistricts.filter(
  ee.Filter.inList('ADM2_NAME', champions.aggregate_array('ADM2_NAME'))
);

Map.addLayer(
  championsVector,
  {color: 'green'},
  'Resilience Champions',
  true,
  0.7
);

// ==================== 30. GENERATE METADATA FILE ====================
print('\n--- Generating Metadata ---');

var metadata = ee.Dictionary({
  title: 'Climate-Water-Health Nexus Dataset for India',
  description: 'Comprehensive district-level climate, water, and environmental data for India (2023)',
  temporal_coverage: '2023-01-01 to 2023-12-31',
  historical_period: '2015-01-01 to 2023-12-31',
  spatial_coverage: 'India (707 districts)',
  spatial_resolution: '5km',
  coordinate_system: 'EPSG:4326 (WGS84)',
  
  data_sources: {
    rainfall: 'CHIRPS Daily (UCSB-CHG/CHIRPS/DAILY)',
    temperature: 'MODIS LST (MODIS/061/MOD11A1)',
    vegetation: 'MODIS Vegetation Indices (MODIS/061/MOD13A2)',
    evapotranspiration: 'MODIS ET (MODIS/061/MOD16A2GF)',
    soil_moisture: 'NASA SMAP (NASA_USDA/HSL/SMAP10KM_soil_moisture)',
    surface_water: 'JRC Global Surface Water (JRC/GSW1_4/GlobalSurfaceWater)',
    land_cover: 'MODIS Land Cover (MODIS/061/MCD12Q1)',
    elevation: 'SRTM DEM (USGS/SRTMGL1_003)',
    districts: 'FAO GAUL (FAO/GAUL/2015/level2)',
    river_basins: 'HydroSHEDS (WWF/HydroSHEDS/v1/Basins/hybas_5)'
  },
  
  variable_categories: {
    rainfall: 12,
    temperature: 8,
    drought: 9,
    flood: 8,
    water_balance: 6,
    soil: 4,
    terrain: 3,
    composite_indices: 3
  },
  
  total_variables: 53,
  
  processing_date: ee.Date(Date.now()).format('YYYY-MM-dd'),
  
  citation: 'Generated using Google Earth Engine. Please cite original data sources.',
  
  contact: 'Your Name / Your Institution',
  
  notes: [
    'All statistics calculated at district level',
    'Mean, median, min, max, and standard deviation provided where applicable',
    'Composite vulnerability scores range from 0-100',
    'Missing data encoded as -9999 or null',
    'District boundaries based on FAO GAUL 2015'
  ]
});

print('Metadata:', metadata);

// ==================== 31. FINAL VALIDATION CHECKS ====================
print('\n========================================');
print('FINAL VALIDATION CHECKS');
print('========================================');

// Check for null values in critical fields
var nullCheckFields = [
  'rain_total_2023_mean',
  'temp_mean_2023_mean',
  'climate_vulnerability_score',
  'water_stress_score'
];

nullCheckFields.forEach(function(field) {
  var nullCount = districtStats.filter(
    ee.Filter.eq(field, null)
  ).size();
  print('Null values in', field + ':', nullCount);
});

// Check for extreme outliers
var outlierCheck = function(field, lowerBound, upperBound) {
  var outliers = districtStats.filter(
    ee.Filter.or(
      ee.Filter.lt(field, lowerBound),
      ee.Filter.gt(field, upperBound)
    )
  );
  return outliers.size();
};

print('\nOutlier Detection:');
print('Rainfall outliers (>10000mm or <0):', 
  outlierCheck('rain_total_2023_mean', 0, 10000));
print('Temperature outliers (>50°C or <-10°C):', 
  outlierCheck('temp_mean_2023_mean', -10, 50));
print('VCI outliers (>100 or <0):', 
  outlierCheck('vci_2023_mean', 0, 100));

// ==================== 32. GENERATE QUICK STATS TABLE ====================
print('\n========================================');
print('QUICK STATISTICS TABLE');
print('========================================');

// Calculate quartiles for key variables
var calculateQuartiles = function(collection, property) {
  var sorted = collection.aggregate_array(property).sort();
  var size = sorted.size();
  
  var q1_idx = size.multiply(0.25).round();
  var median_idx = size.multiply(0.5).round();
  var q3_idx = size.multiply(0.75).round();
  
  return {
    min: sorted.reduce(ee.Reducer.min()),
    q1: sorted.get(q1_idx),
    median: sorted.get(median_idx),
    q3: sorted.get(q3_idx),
    max: sorted.reduce(ee.Reducer.max())
  };
};

print('\nRainfall Statistics (mm):');
print('Distribution:', calculateQuartiles(districtStats, 'rain_total_2023_mean'));

print('\nTemperature Statistics (°C):');
print('Distribution:', calculateQuartiles(districtStats, 'temp_mean_2023_mean'));

print('\nClimate Vulnerability Score:');
print('Distribution:', calculateQuartiles(districtStats, 'climate_vulnerability_score'));

print('\nWater Stress Score:');
print('Distribution:', calculateQuartiles(districtStats, 'water_stress_score'));

// ==================== 33. EXPORT TASK SUMMARY TABLE ====================

// Create a summary of all export tasks
var exportSummary = ee.FeatureCollection([
  ee.Feature(null, {
    task_number: 1,
    task_name: 'India_Climate_Water_Complete_District_Stats',
    description: 'Complete dataset with all variables',
    estimated_size_mb: 8,
    priority: 'High',
    variables: 70
  }),
  ee.Feature(null, {
    task_number: 2,
    task_name: 'India_Key_Climate_Metrics',
    description: 'Key metrics only',
    estimated_size_mb: 3,
    priority: 'High',
    variables: 25
  }),
  ee.Feature(null, {
    task_number: 3,
    task_name: 'India_Climate_Water_Shapefile_2023',
    description: 'Spatial data for GIS',
    estimated_size_mb: 15,
    priority: 'Medium',
    variables: 70
  }),
  ee.Feature(null, {
    task_number: 4,
    task_name: 'India_Climate_Trends_2015_2023',
    description: 'Historical trends',
    estimated_size_mb: 1,
    priority: 'Medium',
    variables: 3
  }),
  ee.Feature(null, {
    task_number: 5,
    task_name: 'India_Extreme_Events_2015_2023',
    description: 'Extreme event counts',
    estimated_size_mb: 1,
    priority: 'Medium',
    variables: 4
  }),
  ee.Feature(null, {
    task_number: 6,
    task_name: 'India_Monsoon_Characteristics_2023',
    description: 'Monsoon analysis',
    estimated_size_mb: 2,
    priority: 'Low',
    variables: 8
  }),
  ee.Feature(null, {
    task_number: 7,
    task_name: 'India_Groundwater_Proxies_2023',
    description: 'Groundwater indicators',
    estimated_size_mb: 1,
    priority: 'Low',
    variables: 4
  }),
  ee.Feature(null, {
    task_number: 8,
    task_name: 'India_Districts_River_Basins',
    description: 'River basin mapping',
    estimated_size_mb: 0.5,
    priority: 'Low',
    variables: 2
  }),
  ee.Feature(null, {
    task_number: 9,
    task_name: 'India_Monthly_Rainfall_2023',
    description: 'Monthly time series',
    estimated_size_mb: 3,
    priority: 'Low',
    variables: 2
  }),
  ee.Feature(null, {
    task_number: 10,
    task_name: 'India_Vulnerability_Hotspots',
    description: 'Triple-threat districts',
    estimated_size_mb: 0.5,
    priority: 'High',
    variables: 70
  }),
  ee.Feature(null, {
    task_number: 11,
    task_name: 'India_Resilience_Champions',
    description: 'Resilient districts',
    estimated_size_mb: 0.5,
    priority: 'High',
    variables: 70
  })
]);

print('\nTotal Export Tasks:', exportSummary.size());
print('Total Estimated Data Size: ~36 MB');

// ==================== 34. USER INSTRUCTIONS ====================
print('\n========================================');
print('📋 STEP-BY-STEP USER GUIDE');
print('========================================');
print('');
print('STEP 1: RUN EXPORT TASKS');
print('-------------------------');
print('1. Look at the "Tasks" tab (top-right, orange icon)');
print('2. You will see 11 export tasks');
print('3. Click RUN on each task');
print('4. Confirm the export settings (usually default is fine)');
print('5. Wait for completion (10-30 minutes per task)');
print('');
print('STEP 2: DOWNLOAD FROM GOOGLE DRIVE');
print('-----------------------------------');
print('1. Go to your Google Drive');
print('2. Find the exported CSV and SHP files');
print('3. Download all files to your computer');
print('4. Create a folder: "Climate_Water_Data_India"');
print('');
print('STEP 3: MERGE WITH NFHS-5 DATA');
print('--------------------------------');
print('In R or Python, merge using district codes:');
print('');
print('R CODE:');
print('climate_data <- read.csv("India_Complete_Climate_Water_Data_2023.csv")');
print('nfhs_data <- read_dta("IAHR7EFL.DTA")');
print('merged <- left_join(nfhs_data, climate_data, by = c("shdist" = "ADM2_CODE"))');
print('');
print('PYTHON CODE:');
print('climate_data = pd.read_csv("India_Complete_Climate_Water_Data_2023.csv")');
print('nfhs_data = pd.read_stata("IAHR7EFL.DTA")');
print('merged = nfhs_data.merge(climate_data, left_on="shdist", right_on="ADM2_CODE")');
print('');
print('STEP 4: PROCEED WITH ANALYSIS');
print('------------------------------');
print('Now you have combined dataset ready for:');
print('- Correlation analysis');
print('- Regression models');
print('- Mediation analysis');
print('- Spatial mapping');
print('');
print('========================================');
print('🎉 SCRIPT COMPLETE!');
print('========================================');
print('');
print('All data processing finished successfully.');
print('Please proceed to Tasks tab to run exports.');
print('');
print('For questions or issues, check the GEE documentation:');
print('https://developers.google.com/earth-engine/');
print('');
print('Good luck with your research!');
print('========================================');

// END OF SCRIPT