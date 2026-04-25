# Registry of Open Data on AWS (RODA)

The [Registry of Open Data on AWS](https://registry.opendata.aws/)  contains over 850 datasets with over 300 Petabytes of data available for you to use for this hackathon (some restrictions apply). Explore the Registry to find geospatial, environmental, and healthcare/Life Sciences data.

[

### Description

](https://catalog.us-east-1.prod.workshops.aws/event/dashboard/en-US/workshop/00-setup/12-roda#description)

This registry exists to help people discover and share datasets that are available via AWS resources. See [recent additions](https://registry.opendata.aws/change-log) .

Get started using data quickly using [SageMaker Studio Lab notebooks samples and tutorials](https://registry.opendata.aws/service/amazon-sagemaker-studio-lab/usage-examples) .

See [all usage examples for datasets listed in this registry](https://registry.opendata.aws/usage-examples) .

[

### Resources in the Registry of Open Data

](https://catalog.us-east-1.prod.workshops.aws/event/dashboard/en-US/workshop/00-setup/12-roda#resources-in-the-registry-of-open-data)

Foundation Models:

- [MadeWithClay geospatial foundation model](https://registry.opendata.aws/clay-model-v0-embeddings/) 
- [Clay Model embeddings](https://registry.opendata.aws/?search=managedBy:source%20cooperative) 
- [AlphaEarth embeddings](https://registry.opendata.aws/aef-source/) 

HLS and NAIP datasets:

- [NAIP on AWS](https://registry.opendata.aws/naip/) 
- [Clay v1.5 NAIP-2](https://registry.opendata.aws/clay-v1-5-naip-2/) 
- [OPERA Dynamic Surface Water Extent from Harmonized Landsat Sentinel-2 product (Version 1)](https://registry.opendata.aws/nasa-operal3dswx-hlsv1/) 
- [OPERA Land Surface Disturbance Annual from Harmonized Landsat Sentinel-2 product (Version 1)](https://registry.opendata.aws/nasa-operal3dist-ann-hlsv1/) 
- [OPERA Land Surface Disturbance Alert from Harmonized Landsat Sentinel-2 product (Version 1)](https://registry.opendata.aws/nasa-operal3dist-alert-hlsv1/) 
- [OPERA Land Surface Disturbance Alert from Harmonized Landsat Sentinel-2 provisional product (Version 0)](https://registry.opendata.aws/nasa-operal3dist-alert-hlsprovisionalv0/) 
- [HLS Landsat Operational Land Imager Surface Reflectance and TOA Brightness Daily Global 30m v2.0](https://registry.opendata.aws/nasa-hlsl30/) 
- HLS Sentinel-2 Multi-spectral Instrument Surface Reflectance Daily Global 30m v2.0 

Useful publications on these datasets:

- [HLS Sentinel-2 Multi-spectral Instrument Surface Reflectance Daily Global 30m v2.0 publications](https://www.earthdata.nasa.gov/data/catalog/lpcloud-hlss30-2.0#publications) 
- [Individual Tree Detection in Large-Scale Urban Environments using High-Resolution Multispectral Imagery (NAIP)](https://arxiv.org/abs/2208.10607) 

See other datasets from:

- [NASA Space Act Agreement](https://registry.opendata.aws/collab/nasa/) 
- [NOAA Open Data Dissemination Program](https://registry.opendata.aws/collab/noaa/) 
- [Amazon Sustainability Data Initiative](https://registry.opendata.aws/collab/asdi/) 

[Explore the Registry of Open Data on AWS](https://registry.opendata.aws/) 

# Extra Datasets Available for the hackathon

There are other datasets and models that you may use in this hackathon. Feel free to check out the following:

[

### Model

](https://catalog.us-east-1.prod.workshops.aws/event/dashboard/en-US/workshop/00-setup/15-extra-info#model)

- [Prithvi foundation model](https://github.com/NASA-IMPACT/Prithvi-EO-2.0)  - we recommend the Prithvi-EO-2.0-300M for the resources configured in this workshop. See the next page for information on how to clone this repository.

[

### USDA Crop datasets

](https://catalog.us-east-1.prod.workshops.aws/event/dashboard/en-US/workshop/00-setup/15-extra-info#usda-crop-datasets)

- [Corn Yeild](https://www.nass.usda.gov/Charts_and_Maps/graphics/cornyld.pdf)  - yeilds over time in a chart
- [CroplandCROS service](https://croplandcros.scinet.usda.gov/)  - use the tool to export data
- [Crop Condition and Soil Moisture Analytics Tool Crop-CASMA](https://www.drought.gov/data-maps-tools/crop-condition-and-soil-moisture-analytics-tool-crop-casma)  - use the interactive map to export data
- **National 2025 10-Meter CDL zipped file** - see the next page for how to fetch this file

[

### Weather and Climate Datasets

](https://catalog.us-east-1.prod.workshops.aws/event/dashboard/en-US/workshop/00-setup/15-extra-info#weather-and-climate-datasets)

- [NOAA GEFS](https://registry.opendata.aws/noaa-gefs-reforecast/)  - NOAA has generated a multi-decadal reanalysis and reforecast data set to accompany the next-generation version of its ensemble prediction system, the Global Ensemble Forecast System, version 12 (GEFSv12).
- [NOAA GEFS Icechunk Zarr](https://registry.opendata.aws/dynamical-noaa-gefs/)  - GEFS creates 31 separate forecasts (ensemble members) to describe the range of forecast uncertainty. These datasets have been translated to cloud-optimized Icechunk Zarr format by dynamical.org.
- [NASA POWER](https://registry.opendata.aws/nasa-power/)  - The Prediction Of Worldwide Energy Resources (POWER) Project, funded through the Applied Sciences Program at NASA Langley Research Center, gathers NASA Earth observation data and parameters related to the fields of surface solar irradiance and meteorology to serve the public in several free, easy-to-access and easy-to-use methods. POWER helps communities become resilient amid observed climate variability by improving data accessibility, aiding research in energy development, building energy efficiency, and supporting agriculture projects.
- [NASA MODIS](https://registry.opendata.aws/nasa-mod13q1/)  - The MOD13Q1 product provides two primary vegetation layers. The first is the Normalized Difference Vegetation Index (NDVI) which is referred to as the continuity index to the existing National Oceanic and Atmospheric Administration-Advanced Very High Resolution Radiometer (NOAA-AVHRR) derived NDVI. The second vegetation layer is the Enhanced Vegetation Index (EVI), which has improved sensitivity over high biomass regions.
- [NASA SMAP Project](https://search.earthdata.nasa.gov/search?q=SMAP)  - This dataset contains data obtained by the Passive Active L- and S-band (PALS) microwave aircraft instrument that are matched up with a variety of soil moisture campaign data. You can use the NASA EarthData Search tool to find SMAP data in Amazon S3 buckets.
- [NSF NCAR ERA5](https://registry.opendata.aws/nsf-ncar-era5/)  - NSF NCAR provides a NetCDF-4 structured version of the 0.25 degree atmospheric ECMWF Reanalysis 5 (ERA5) to AWS Open Data.
- [Planette ERA5 Archive](https://registry.opendata.aws/planette_era5_reanalysis/)  - The ERA5 archive provides a comprehensive record of global weather and climate from 1940 to present, with multiple temporal aggregations for flexible analysis. This dataset is derived from the ECMWF/Copernicus ERA5 reanalysis and includes daily means, 7-day rolling means, and monthly/seasonal aggregations at 0.25°×0.25° global resolution. The Planette ERA5 archive stores this data in cloud-native format (Zarr with icechunk) for efficient access and analysis.

More information about these datasets

- [SMAP data products](https://smap.jpl.nasa.gov/data/) 

