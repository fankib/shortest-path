import datetime
import utm
import pandas as pd

from aerofiles import igc
from shortest_path import Path, GridSearchShortestPath, Point2f, ShortestPathOptimizer

from lib.task_loader import load_from_xctsk

'''
This script computs the pilot prorgress towards goal. It is based on the shorts path algorihtm and gradually deletes turnpoints
if it matches them.
'''

task = load_from_xctsk('tasks/task_2025-03-01.xctsk')
task.turnpoints[-1].radius = 100 # simulate line goal
airstart = datetime.time(11, 35) # use utc

# Create initial shortest path
center_path = Path.from_center_points(task)
optimizer = GridSearchShortestPath(task)
path, distances, config = optimizer.run_fast()

pilots = {'benjamin': 'igc/task_2025-03-01-Regio/2025-03-01-XCT-BFA-11.igc',
           'roger': 'igc/task_2025-03-01-Regio/2025-03-01-XCT-RAE-01.igc',
           'patrick': 'igc/task_2025-03-01-Regio/2025-03-01-XCT-PMO-01.igc'
           }

# Select a pilot:
pilot = 'roger'

# Compute the pilots progress at each timestep
with open(pilots[pilot], 'r') as f:
    pilot_igc = igc.Reader().read(f)


# build pandas cols:
col_time = []
col_lat = []
col_lon = []
col_pressure_alt = []
col_gps_alt = []
col_airstart = []
col_next_tp = []
col_goal = []
col_distance = []

turnpoint_counter = 1
in_goal = False
for record in pilot_igc['fix_records'][1]:
    # {'time': datetime.time(11, 16, 31), 'lat': 46.7004, 'lon': 7.820883333333334, 'validity': 'A', 'pressure_alt': 1181, 'gps_alt': 1275, 'LAD': 1, 'LOD': 3, 'datetime': datetime.datetime(2025, 3, 1, 11, 16, 31, tzinfo=<aerofiles.util.timezone.TimeZoneFix object at 0x762567367e00>)}
    
    # for each record: 
    #  - check if airstart is on
    #  - check if pilot hits first turnpoint => remove turnpoint from path
    #  - if there is no turnpoint left, pilot is in goal / (or ess, depending on modeling)
    #  - compute the progress to goal
    #  - save progress

    task_started = airstart <= record['time']

    x, y, _, _ = utm.from_latlon(record['lat'], record['lon'])
    pilot_position = Point2f(x,y)

    # missuse the first turnpoint as pilot position:
    task.turnpoints[0].center = pilot_position
    task.turnpoints[0].radius = 10 # small radius

    # test for turnpoint 1
    if not in_goal and task_started and task.turnpoints[1].intersect(pilot_position):
        print(f'HIT TP{turnpoint_counter} at', record['time'])
        if len(task.turnpoints) >= 3:
            del task.turnpoints[1]
            del path.points[1] # remove coresponding point
            turnpoint_counter += 1            
        else:
            print('In Goal!')
            in_goal = True
    
    # optimize distance given the updated path and the best config
    optimizer = ShortestPathOptimizer(task, config['lr'], config['itr'], config['crit'], config['weight'], config['back'])
    path = optimizer.shortest_path(path)

    # store values
    col_time.append(record['time'])
    col_lat.append(record['lat'])
    col_lon.append(record['lon'])
    col_pressure_alt.append(record['pressure_alt'])
    col_gps_alt.append(record['gps_alt'])
    col_airstart.append(task_started)
    col_next_tp.append(f'TP{turnpoint_counter+1}')
    col_goal.append(in_goal)
    col_distance.append(path.distance())

    print('Date: ', record['time'], '\tGPS: ', record['gps_alt'], '\tAirstart: ', airstart <= record['time'], 'NextTP:', f'TP{turnpoint_counter+1}', 'Goal:', in_goal, '\tDistance: ', path.distance() / 1000.)

df = pd.DataFrame({
    'time': col_time,
    'lat': col_lat,
    'lon': col_lon,
    'pressure_alt': col_pressure_alt,
    'gps_alt': col_gps_alt,
    'airstart': col_airstart,
    'next_tp': col_next_tp,
    'goal': col_goal,
    'distance': col_distance
})
#df.to_csv(f'dump/{pilot}.csv.gz', index=False, compression="gzip")
df.to_csv(f'dump/{pilot}.csv', index=False)


