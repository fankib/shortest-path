import math
import simplekml

# Constants
START_LAT = 46.9481  # Latitude of Bern, Switzerland
START_LON = 7.4474   # Longitude of Bern, Switzerland
ALTITUDE = 1000      # Altitude in meters
NUM_SEGMENTS = 10   # Number of segments
SEGMENT_LENGTH = 100  # Each segment is 10 meters long

# Function to move 10 meters in the direction of the current bearing
def move_point(lat, lon, bearing, distance):
    # Earth radius in meters
    R = 6371000
    
    # Convert latitude, longitude, and bearing to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing)
    
    # Calculate new latitude and longitude
    lat_new = math.asin(math.sin(lat_rad) * math.cos(distance / R) +
                        math.cos(lat_rad) * math.sin(distance / R) * math.cos(bearing_rad))
    
    lon_new = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance / R) * math.cos(lat_rad),
                                    math.cos(distance / R) - math.sin(lat_rad) * math.sin(lat_new))
    
    # Convert back to degrees
    lat_new_deg = math.degrees(lat_new)
    lon_new_deg = math.degrees(lon_new)
    
    return lat_new_deg, lon_new_deg

# Generate the line with varying transparency and width
def generate_kml():
    kml = simplekml.Kml()
    coords = [(START_LON, START_LAT)]  # Starting point
    
    # Generate points and modify transparency and width
    for i in range(1, NUM_SEGMENTS + 1):
        # Calculate the bearing for the direction of movement
        bearing = 0  # Heading in degrees, change this if you want a different direction (e.g., North)
        
        # Move the point
        lat_new, lon_new = move_point(coords[-1][1], coords[-1][0], bearing, SEGMENT_LENGTH)
        coords.append((lon_new, lat_new, 1000))
        
        # Create transparency (from 0.1 to 1) and width (from 1 to 5)
        transparency = max(0.1, min(1, 1 - 0.5*i / NUM_SEGMENTS))  # From 1 to 0.1
        width = max(1, min(50, 1 + 50*i / NUM_SEGMENTS))  # From 1 to 5
        
        transparency = 1.0
        print('transparency:', transparency, '\twidth:', width)

        # Add line segment to KML
        line = kml.newlinestring(coords=coords)
        line.altitudemode = simplekml.AltitudeMode.absolute
        line.style.linestyle.width = width
        line.style.linestyle.color = simplekml.Color.changealphaint(int(255 * transparency), simplekml.Color.red)  # Red line with varying transparency
        
    # Save KML to file
    kml.save("line_with_varying_segments.kml")

# Run the function to generate the KML
generate_kml()
