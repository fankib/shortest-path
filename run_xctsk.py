import matplotlib.pyplot as plt
import numpy as np
import json
import utm
from shortest_path import Turnpoint, Task, Path, Visualizer, ShortestPathOptimizer, Point2f, GridSearchShortestPath, reject_outliers

from lib.task_loader import load_from_xctsk

''' This demo shows how to turn a XCTSK task file into the respective UTM coordinates and compute the optimal distance as
shortest path from start to end, touching all turnpoints. It uses a small grid search to test different parameters and matplotlib
to present the results.
'''

# define task file
#f = open('tasks/67km-Napf-Walalp.xctsk')
#f = open('tasks/sc_biel_t1.xctsk')
#f = open('tasks/task_2022-02-28.xctsk')
#f = open('tasks/task_2022-04-12.xctsk')
#f = open('tasks/task_2023-03-04.xctsk')
#f = open('tasks/task_2025-03-01.xctsk')

# Regio Verbier 2024
#f = open('tasks/regio_verbier/task_2024-03-16_1.xctsk')

# create Task from file

task = load_from_xctsk('tasks/task_2025-03-01.xctsk')
task.turnpoints[-1].radius = 10 # simulate line goal
#task = Task(tps[1:-1]) # sss ess task


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
path, distances, config = optimizer.run_fast()
#path, distances, config = optimizer.run_slow()
path_sss_ess = Path(path.points[1:-1])
print(f'distance of shortest path: {path.distance()/1000:.1f}')
print(f'distance of shortest path sss-ess: {path_sss_ess.distance()/1000:.1f}')
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
