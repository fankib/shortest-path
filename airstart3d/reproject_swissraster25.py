import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Input and output file paths
input_tif = "/home/benjamin/Downloads/SwissRaster25/swiss-map-raster25_2021_1229_komb_1.25_2056_airstart.tif"
output_tif = "/home/benjamin/Downloads/SwissRaster25/swiss-map-raster25_2021_1229_utm32T.tif"

# Define Destination CRS (UTM 32T)
dst_crs = "EPSG:32632"  # UTM Zone 32N (Switzerland)

# Open the source raster
with rasterio.open(input_tif) as src:
    print(f"Source CRS: {src.crs}")  # Debug: Check source CRS
    print(f"Data Type: {src.dtypes}")  # Debug: Check data type (should be uint8)
    print(f"Band Count: {src.count}")  # Debug: Check if RGB (3 bands) or RGBA (4 bands)
    print(f"Source NoData Value: {src.nodata}")  # Debug: Check NoData value

    # Compute transformation for UTM 32T (keep resolution consistent)
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=25
    )

    # Update metadata while keeping dtype as uint8
    kwargs = src.meta.copy()
    kwargs.update({
        "crs": dst_crs,
        "transform": transform,
        "width": width,
        "height": height,
        "nodata": src.nodata,  # Preserve NoData
        "dtype": "uint8"  # Ensure colors remain uint8
    })

    # Create the output raster
    with rasterio.open(output_tif, "w", **kwargs) as dst:
        for i in range(1, src.count + 1):  # Loop over all bands (RGB or RGBA)
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest  # Use nearest-neighbor for sharpness
            )

print("✅ Reprojection complete! Saved as", output_tif)
