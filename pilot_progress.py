import datetime
import utm
import os
import re
import pandas as pd
from tqdm import tqdm

from aerofiles import igc
from shortest_path import Path, GridSearchShortestPath, Point2f, ShortestPathOptimizer

from lib.task_loader import load_from_xctsk

'''
This script computs the pilot prorgress towards goal. It is based on the shorts path algorihtm and gradually deletes turnpoints
if it matches them.
'''

def clean_name(name):
    name = name.lower()
    name = name.replace("ä", "a").replace("ö", "o").replace("ü", "u").replace('é', 'e').replace('è', 'e').replace('à', 'a')
    name = re.sub('[^a-zA-Z0-9]+', '-', name)
    name = name.strip('-')
    return name

# contains .xctsk and .igc
#ROOT_DIRECTORY = 'igc/task_2025-03-01' 

# Swiss Leage Cups
ROOT_DIRECTORY = 'igc/swissleague/march/igc6494_2025-03-08'

# scan directory
task_file = None
igc_files = []
for file in os.listdir(ROOT_DIRECTORY):
    if file.lower().endswith('.xctsk'):
        assert task_file is None, 'two task files found!'
        task_name = file.split('.')[0]
        task_file = os.path.join(ROOT_DIRECTORY, file)
    if file.lower().endswith('.igc'):
        igc_files.append(os.path.join(ROOT_DIRECTORY, file))

# Scan igc files for names:
print("Pilots found:")
pilots = {}
for igc_file in igc_files:
    with open(igc_file, 'r') as f:
        pilot_igc = igc.Reader().read(f)
        pilot_name = pilot_igc['header'][1]['pilot']
        if pilot_name is not None:
            pilot_name = clean_name(pilot_igc['header'][1]['pilot'])
        else:
            match = re.search(r"LiveTrack (.+?)\.\d", igc_file)
            pilot_name = match.group(1)
        assert pilot_name not in pilots, 'Dublicated file found! ' + igc_file
        pilots[pilot_name] = igc_file




original_task = load_from_xctsk(task_file)
#task.turnpoints[-1].radius = 100 # simulate line goal
#airstart = datetime.time(11, 35) # use utc

# Create initial shortest path
center_path = Path.from_center_points(original_task)
optimizer = GridSearchShortestPath(original_task)
original_path, distances, config = optimizer.run_fast()

#pilots = {'benjamin': 'igc/task_2025-03-01-Regio/2025-03-01-XCT-BFA-11.igc',
#           'roger': 'igc/task_2025-03-01-Regio/2025-03-01-XCT-RAE-01.igc',
#           'patrick': 'igc/task_2025-03-01-Regio/2025-03-01-XCT-PMO-01.igc'
#           }

# Select a pilot:
#pilot = 'roger'

for pilot in pilots.keys():
    print('Process pilot: ', pilot)

    # restore original task and path (hackedihackhack)
    task = original_task.copy()
    path = original_path.copy()

    if pilot == 'adrian-muller' or pilot == 'muhlemann-ren':
        print('skip', pilot)
        continue

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
    for record in tqdm(pilot_igc['fix_records'][1]):
        # {'time': datetime.time(11, 16, 31), 'lat': 46.7004, 'lon': 7.820883333333334, 'validity': 'A', 'pressure_alt': 1181, 'gps_alt': 1275, 'LAD': 1, 'LOD': 3, 'datetime': datetime.datetime(2025, 3, 1, 11, 16, 31, tzinfo=<aerofiles.util.timezone.TimeZoneFix object at 0x762567367e00>)}
        
        # for each record: 
        #  - check if airstart is on
        #  - check if pilot hits first turnpoint => remove turnpoint from path
        #  - if there is no turnpoint left, pilot is in goal / (or ess, depending on modeling)
        #  - compute the progress to goal
        #  - save progress

        task_started = task.airstart <= record['time']

        x, y, _, _ = utm.from_latlon(record['lat'], record['lon'])
        pilot_position = Point2f(x,y)

        # missuse the first turnpoint as pilot position:
        task.turnpoints[0].center = pilot_position
        task.turnpoints[0].radius = 1 # small radius

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

        #print('Date: ', record['time'], '\tGPS: ', record['gps_alt'], '\tAirstart: ', task.airstart <= record['time'], 'NextTP:', f'TP{turnpoint_counter+1}', 'Goal:', in_goal, '\tDistance: ', path.distance() / 1000.)

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
    #try:
    #    os.mkdir(f'dump/{task_name}/')
    #except OSError:
    #    pass
    os.makedirs(f'dump/{task_name}', exist_ok=True)
    df.to_csv(f'dump/{task_name}/{pilot}.csv', index=False)


