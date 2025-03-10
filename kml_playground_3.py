import utm
import simplekml
import math

# Example UTM coordinates with height, color, size, and transparency
utm_coords = [
    (389000, 5810000, 33, 'U', 1000, "aaff0000", 1.5),  # Transparent Red, small
    (390000, 5811000, 33, 'U', 1200, "aaff00ff", 2.5),  # Transparent Magenta, medium
    (391000, 5812000, 33, 'U', 800,  "aa00ff00", 3.5),  # Transparent Green, bigger
    (392000, 5813000, 33, 'U', 1500, "aaffff00", 4.5)   # Transparent Yellow, large
]

# Create KML object
kml = simplekml.Kml()

# Create a polygon (circle) at each coordinate
for east, north, zone, letter, altitude, color, scale in utm_coords:
    
    # Create a circle by defining a polygon with multiple points
    # Define a small radius to create a circle
    radius = scale * 50  # Scale determines the size of the circle
    num_points = 10  # Number of points to approximate the circle

    # Generate the circle points
    circle_points_utm = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points  # Divide the circle into equal parts
        lat_offset = radius * math.sin(angle)  # Latitude offset for the circle (meters)
        lon_offset = radius * math.cos(angle)  # Longitude offset for the circle (meters)

        # Convert the lat/lon offset in meters to UTM offsets
        # UTM coordinates are in meters, so we need to convert the offsets appropriately
        # 1 degree of latitude is roughly 111,320 meters, and the length of a degree of longitude
        # depends on the latitude (cosine effect).
        #lat_offset_utm = lat_offset / 111320.0  # Convert meters to degrees
        #lon_offset_utm = lon_offset / (111320.0 * math.cos(math.radians(north)))  # Adjust for longitude at the given latitude

        # Append the new UTM coordinates
        #circle_points_utm.append((east + lon_offset_utm, north + lat_offset_utm, altitude))
    
    # Transform to lat long:
    lat, lon = utm.to_latlon(east, north, zone, letter)
    circle_points = []
    for cp_utm in circle_points_utm:
        cp_lat, cp_lon = utm.to_latlon(cp_utm[0], cp_utm[1], zone, letter)
        circle_points.append((cp_lon, cp_lat))

    # Create the polygon (circle)
    pol = kml.newpolygon(name="Transparent Circle", outerboundaryis=circle_points)
    pol.altitudemode = simplekml.AltitudeMode.absolute  # Position the polygon in 3D space

    # Set transparent color: "AABBGGRR" format (A=Alpha/Transparency, BB=Blue, GG=Green, RR=Red)
    pol.style.polystyle.color = color  # Color (with transparency)
    pol.style.linestyle.color = "ff000000"  # Transparent line (no border)
    pol.style.polystyle.fill = 1  # Fill the circle

    # Enable backface rendering to show both sides of the circle (disable backface culling)
    pol.style.polystyle.backfaceculling = False

# Save the KML file
kml.save("utm_transparent_circles_with_backfaces.kml")
