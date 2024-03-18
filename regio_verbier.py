import matplotlib.pyplot as plt
import numpy as np
import json
import utm
from shortest_path import Turnpoint, Task, Path, Visualizer, ShortestPathOptimizer, Point2f, GridSearchShortestPath, reject_outliers

from airstart import angle, is_dominant, compute_normal
from igc_reader import IgcReader


# Regio Verbier 2024
f = open('tasks/regio_verbier/task_2024-03-16_1.xctsk')

# create Task from file
tps = json.load(f)['turnpoints']
lats = np.array([float(tp['waypoint']['lat']) for tp in tps])
lons = np.array([float(tp['waypoint']['lon']) for tp in tps])
xs, ys, a, b = utm.from_latlon(lats, lons, 32, 'T')
print('a', a)
print('b', b)
radiuses = [float(tp['radius']) for tp in tps]
tps = [Turnpoint(center=Point2f(x, y), radius=r) for x, y, r in zip(xs, ys, radiuses)]
task = Task(tps)

# draw task
visualizer = Visualizer()
visualizer.equal_aspect()
visualizer.draw_task(task)

center_path = Path.from_center_points(task)
visualizer.draw_path(center_path, alpha=0.2)

# run a grid search to test different settings
optimizer = GridSearchShortestPath(task)
path, distances = optimizer.run_fast()
#path, distances = optimizer.run_slow()
print(f'distance of shortest path: {path.distance()/1000:.1f}')
visualizer.draw_path(path)

#igc_reader = IgcReader('tasks/regio_verbier/igc', '12:45') # 2. Start
#igc_reader = IgcReader('tasks/regio_verbier/igc', '13:05') # 2. Start
igc_reader = IgcReader('tasks/regio_verbier/igc', '13:25') # sss

#igc_reader = IgcReader('tasks/regio_verbier/igc', '14:27') # time in utc
#igc_reader = IgcReader('tasks/regio_verbier/igc', '14:35') # time in utc

for pilot in igc_reader.pilots:
    a = pilot.position
    plt.scatter(a[0], a[1])
    plt.text(a[0], a[1], pilot.name)


## Do the 3D Domination thingy ##

# cut sss to wp1 in 3D:
sss_to_wp = Path(path.points[2:4]) # 2:4 as the start is special
visualizer.draw_path(sss_to_wp)
sss_2d = path.points[2]
wp1_2d = path.points[3]
sss = np.array([sss_2d.x, sss_2d.y, 2000])
wp1 = np.array([wp1_2d.x, wp1_2d.y, 2000])

# show sss, wp1:
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(sss[0], sss[1], sss[2], label='sss')
ax.scatter(wp1[0], wp1[1], wp1[2], label='wp1')
ax.plot([sss[0], wp1[0]], [sss[1], wp1[1]], [sss[2], wp1[2]])

# compute normal of domination plan:
ratio = 1/3 # domination glide angle
#ratio = 1/2 # domination glide angle
n = compute_normal(sss, wp1, ratio)


# compute domination
for pilot in igc_reader.pilots:
    a = pilot.position
    sss_pilot = np.array([a[0], a[1], 2000])
    n = compute_normal(sss_pilot, wp1, ratio)
    domination_counter = 0
    # use default n
    for p in igc_reader.pilots:
        if p.name == pilot.name:
            continue
        b = p.position
        dominated = is_dominant(a, b, n)
        if dominated:
            domination_counter += 1
    pilot.dominations = domination_counter

# sort pilots:
pilots_by_domination = sorted(igc_reader.pilots, key=lambda p: p.dominations, reverse=True)
for p in pilots_by_domination:
    print(f'{p.name:<25}{p.dominations}')
#print([(p.name,p.dominations) for p in pilots_by_domination])

# process for pilot 1:
pilot = igc_reader.pilot_by_name('Fankhauser')
print('view of ', pilot.name)
a = pilot.position
sss_pilot = np.array([a[0], a[1], 2000])
n = compute_normal(sss_pilot, wp1, ratio)
ax.scatter(a[0], a[1], a[2], c='g', s=50)
ax.text(a[0], a[1], a[2], pilot.name)
for p in igc_reader.pilots:
    if p.name == pilot.name:
        continue
    b = p.position
    dominated = is_dominant(a, b, n)
    color = 'r' if dominated else 'b'
    alpha = 0.7 if dominated else 0.4
    ax.scatter(b[0], b[1], b[2], c=color, alpha=alpha)
    ax.text(b[0], b[1], b[2], p.name, alpha=alpha)



ax.set_aspect('equalxy')
ax.legend()

plt.show()




plt.show()



