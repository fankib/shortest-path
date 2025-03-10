import utm
import simplekml

# Beispielhafte UTM-Koordinaten mit Höhe, Farbe und Linienstärke
utm_coords = [
    (389000, 5810000, 33, 'U', 1000, simplekml.Color.red, 20),  
    (390000, 5811000, 33, 'U', 1200, simplekml.Color.red, 40),
    (391000, 5812000, 33, 'U', 800, simplekml.Color.red, 20),
    (392000, 5813000, 33, 'U', 1500, simplekml.Color.red, 8)
]

# KML-Objekt erstellen
kml = simplekml.Kml()

# Liniensegmente erzeugen
for i in range(len(utm_coords) - 1):
    # Start- und Endpunkt für jedes Segment
    e1, n1, zone, letter, alt1, color, width = utm_coords[i]
    e2, n2, _, _, alt2, _, _ = utm_coords[i + 1]
    
    # UTM -> Lat/Lon Umrechnung
    lat1, lon1 = utm.to_latlon(e1, n1, zone, letter)
    lat2, lon2 = utm.to_latlon(e2, n2, zone, letter)
    
    # Neues Liniensegment
    line = kml.newlinestring()
    line.coords = [(lon1, lat1, alt1), (lon2, lat2, alt2)]
    
    # Extrusion zum Boden
    #line.extrude = 1
    line.altitudemode = simplekml.AltitudeMode.absolute

    # Individuelle Farbe und Linienstärke setzen
    #line.style.linestyle.color = color  # Farbe aus Liste
    line.style.linestyle.color = simplekml.Color.changealphaint(int(255 * width/40), simplekml.Color.red)
    line.style.linestyle.width = width  # Dicke aus Liste

# KML speichern
kml.save("utm_3d_spur_variabel.kml")