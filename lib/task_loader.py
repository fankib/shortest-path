import json
import utm
from datetime import datetime

import numpy as np

from shortest_path import Task, Turnpoint, Point2f

'''
Loads a Task from xctsk files
'''

def load_from_xctsk(filename):
    f = open(filename)
    j = json.load(f)

    first_gate = j['sss']['timeGates'][0]
    airstart = datetime.strptime(first_gate, "%H:%M:%SZ").time()
    

    tps = j['turnpoints']
    lats = np.array([float(tp['waypoint']['lat']) for tp in tps])
    lons = np.array([float(tp['waypoint']['lon']) for tp in tps])
    print(lats, lons)
    xs, ys, _, _ = utm.from_latlon(lats, lons)
    radiuses = [float(tp['radius']) for tp in tps]
    tps = [Turnpoint(center=Point2f(x, y), radius=r) for x, y, r in zip(xs, ys, radiuses)]
    return Task(tps, airstart)
