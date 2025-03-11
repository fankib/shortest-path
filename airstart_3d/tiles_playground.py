import requests
from vpython import box, textures, vector
from PIL import Image
from io import BytesIO
import numpy as np
import math


# Function to download tile from OpenStreetMap


def download_tile(x, y, zoom):
    url = create_url(x, y, zoom)
    response = requests.get(url)
    
    if response.status_code == 200:
        # Create texture from the tile image
        img = Image.open(BytesIO(response.content))
        return img
    else:
        print(f"Error: Could not download tile at zoom {zoom}, x {x}, y {y}")
        print(url)
        return None

# Function to apply texture to the box
def apply_texture_to_box(image):
    # Convert the image to a format VPython can use
    texture_img = np.array(image)
    ground_box.texture = textures.texture(data=texture_img)




if __name__ == '__main__':
    # Example location (latitude, longitude) for Switzerland (or any location)
    lat = 46.629433   # Latitude for Switzerland
    lon = 7.855289    # Longitude for Switzerland
    zoom = 12     # Choose your zoom level
    # expected: x=534/662 or 534,361



    x, y, off_x, off_y = latlon_to_tile(lat, lon, zoom)
    print(f"Tile coordinates for zoom {zoom}: x = {x}, y = {y}, using the offset: {off_x}/{off_y}")


    # Download the tile for the given region
    #tile_image = download_tile(tile_x, tile_y, zoom_level)

    #if tile_image:
        # Apply the downloaded texture to the ground box
        #apply_texture_to_box(tile_image)


    # Create a box at the ground level plane (x/z)
    url = create_url(x, y, zoom)
    print('url', url)
    ground_box = box(pos=vector(0, 0.01, 0), size=vector(100, 0.1, 100))  # Box with ground level height
    ground_box.texture = url
