import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from rasterio.coords import BoundingBox
from shapely.geometry import box
import geopandas as gpd
import json

# Input and output file paths
input_tif = "/home/benjamin/Downloads/EU_DEM_mosaic_5deg/eudem_dem_4258_europe.tif"  # Input raster in EPSG:4326
output_tif = "/home/benjamin/Downloads/EU_DEM_mosaic_5deg/eudem_dem_4258_europe_utm32T.tif"  # Output raster in EPSG:32632

# Define CRS
src_crs = "EPSG:4258"  # ETRS89 (lat/lon)
dst_crs = "EPSG:32632"  # UTM 32N (Switzerland)

# Define Switzerland bounding box in WGS84/ETRS89 (lat/lon)
swiss_bounds_etrs89 = [5.8, 45.7, 10.7, 48.0]  # (min_lon, min_lat, max_lon, max_lat)

# Open the source dataset
with rasterio.open(input_tif) as src:
    # Convert bounding box to rasterio format (left, bottom, right, top)
    bbox = box(*swiss_bounds_etrs89)
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=src_crs)
    geo_json = json.loads(geo.to_json())['features'][0]['geometry']

    # Crop to Switzerland (ETRS89 lat/lon)
    cropped_img, cropped_transform = mask(src, [geo_json], crop=True)

    # Compute transform for UTM 32T with fixed 25m resolution
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *swiss_bounds_etrs89, resolution=25
    )

    # Update metadata
    kwargs = src.meta.copy()
    kwargs.update({
        "crs": dst_crs,
        "transform": transform,
        "width": width,
        "height": height,
        #"nodata": -9999  # Define NoData value
    })

    # Reproject and save output
    with rasterio.open(output_tif, "w", **kwargs) as dst:
        for i in range(1, src.count + 1):  # Loop over bands
            reproject(
                source=cropped_img[i - 1],  # Cropped DEM data
                destination=rasterio.band(dst, i),
                src_transform=cropped_transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear  # Smooth resampling for elevation
            )

print("Reprojection and cropping complete! Saved as", output_tif)