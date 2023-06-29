import matplotlib.pyplot as plt
import numpy as np
import json
import utm
from shortest_path import Turnpoint, Task, Path, Visualizer, ShortestPathOptimizer, Point2f, GridSearchShortestPath, reject_outliers

''' This demo shows how to turn a XCTSK task file into the respective UTM coordinates and compute the optimal distance as
shortest path from start to end, touching all turnpoints. It uses a small grid search to test different parameters and matplotlib
to present the results.
'''

# define task file
#f = open('tasks/67km-Napf-Walalp.xctsk')
f = open('tasks/sc_biel_t1.xctsk')
#f = open('tasks/task_2022-02-28.xctsk')
#f = open('tasks/task_2022-04-12.xctsk')
#f = open('tasks/task_2023-03-04.xctsk')

# create Task from file
tps = json.load(f)['turnpoints']
lats = np.array([float(tp['waypoint']['lat']) for tp in tps])
lons = np.array([float(tp['waypoint']['lon']) for tp in tps])
xs, ys, _, _ = utm.from_latlon(lats, lons)
radiuses = [float(tp['radius']) for tp in tps]
tps = [Turnpoint(center=Point2f(x, y), radius=r) for x, y, r in zip(xs, ys, radiuses)]
task = Task(tps)

# visualize Task
visualizer = Visualizer()
visualizer.equal_aspect()
visualizer.draw_task(task)

# create center path
center_path = Path.from_center_points(task)
print(f'distance of center path: {center_path.distance()/1000:.1f}')
visualizer.draw_path(center_path, alpha=0.2)

# run a grid search to test different settings
optimizer = GridSearchShortestPath(task)
path, distances = optimizer.run_fast()
#path, distances = optimizer.run_slow()
print(f'distance of shortest path: {path.distance()/1000:.1f}')
visualizer.draw_path(path)

# run the shortest path optimizer manually (path loss):
#distances = None
#optimizer = ShortestPathOptimizer(task, lr=5.0, iterations=10000, stop_criteria=2000, tp_weight=0.000, backproject=True)
#path = optimizer.shortest_path(path)
#print(f'distance of shortest path: {path.distance()/1000:.1f}')
#visualizer.draw_path(path)

# segment statistics:
#for i in range(len(path.points)-1):
#    a = path.points[i].x - path.points[i+1].x
#    b = path.points[i].y - path.points[i+1].y
#    d = np.sqrt(a**2+b**2)
#    print(f'segment {i}: {d:.3f}m')

# manually:
#path.points[5].x = 374208
#path.points[5].y = 5225738
#print(f'distance of xctrack path: {path.distance()/1000:.1f}')
#visualizer.draw_path(path)

# show histogram of distances (when using grid search)
SHOW_DISTANCES = False
if distances and SHOW_DISTANCES:
    plt.figure()
    plt.hist(reject_outliers(distances)/1000)

plt.show()
