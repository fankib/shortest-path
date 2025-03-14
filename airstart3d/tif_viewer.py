import rasterio
import matplotlib.pyplot as plt

src_tif = "/home/benjamin/Downloads/EU_DEM_mosaic_5deg/eudem_dem_4258_europe.tif"  # Input raster in EPSG:4326
utm32_tif = "/home/benjamin/Downloads/EU_DEM_mosaic_5deg/eudem_dem_32632_switzerland.tif"  # Output raster in EPSG:32632

with rasterio.open(utm32_tif) as src:
    plt.imshow(src.read(1), cmap="terrain")
    plt.colorbar(label="Elevation (m)")
    plt.title("EU-DEM (Switzerland, UTM 32T, 25m)")
    plt.show()