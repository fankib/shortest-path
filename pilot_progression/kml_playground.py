import utm
import simplekml

utm_coords = [
    (389000, 5810000, 33, 'U', 1000, "ff0000ff", 2),  # Rot
    (390000, 5811000, 33, 'U', 1200, "ff00ffff", 4),  # Magenta
    (391000, 5812000, 33, 'U', 800,  "ff00ff00", 6),  # Grün
    (392000, 5813000, 33, 'U', 1500, "ffffff00", 8)   # Gelb
]

kml = simplekml.Kml()
doc = kml.newdocument(name="3D Track")

trk = doc.newgxtrack(name="Benjamin")

for east, north, zone, letter, altitude, color, width in utm_coords:

    lat, lon = utm.to_latlon(east, north, zone, letter)

    trk.newwhen(["2025-03-08T12:00:00Z"])
    trk.newgxcoord([(lon, lat, altitude)])

    # stil:
    linestyle = trk.linestyle
    linestyle.color = color
    linestyle.width = width

kml.save("dump/benjamin.kml")