import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from tqdm import tqdm

### Little helpers

# we do not want to be informed about crashed optimizations
np.seterr(all='ignore')

def reject_outliers(data, m=2.):
    data = np.array(data)
    data = data[~np.isnan(data)]
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]

class EarlyStopp(Exception): pass

### Data classes

class Point2f:

    def __init__(self, x=0., y=0.):
        self.x = x
        self.y = y
    
    def copy(self):
        return Point2f(self.x, self.y)

class Task:
    
    def __init__(self, turnpoints):
        self.turnpoints = turnpoints

class Turnpoint:
    
    def __init__(self, center=Point2f(), radius=0.0):
        self.center = center
        self.radius = radius

class Path:

    @staticmethod
    def from_center_points(task):
        return Path([tp.center.copy() for tp in task.turnpoints])        

    def __init__(self, points):
        self.points = points
    
    def distance(self):
        d = 0
        for i in range(len(self.points)-1):
            d += np.sqrt((self.points[i].x-self.points[i+1].x)**2 + (self.points[i].y-self.points[i+1].y)**2)
        return d


### Visualizer

class Visualizer:

    def equal_aspect(self):
        plt.gca().set_aspect('equal')

    def draw_task(self, task):
        for tp in task.turnpoints:
            circle = plt.Circle((tp.center.x, tp.center.y), tp.radius, color='b', fill=False)
            plt.gca().add_patch(circle)

    def draw_path(self, path, **kwargs):
        xs = [p.x for p in path.points]
        ys = [p.y for p in path.points]
        plt.plot(xs, ys, **kwargs)
        plt.scatter(xs, ys, **kwargs)

### Define Loss functions

class TurnpointSquareLoss():
    ''' This loss is a quadratic function (with a weight) in the form of L=w*x^2 applied only outside of the turnpoints radius.
    L=0 if inside. Backproject moves a point outside of the turnpoint back to to the turnpoints radius.
    '''
    
    def __init__(self, turnpoint, point, weight):
        self.turnpoint = turnpoint
        self.p = point
        self.weight = weight

    def eval(self):
        d = np.sqrt((self.p.x-self.turnpoint.center.x)**2 + (self.p.y-self.turnpoint.center.y)**2)
        if d < self.turnpoint.radius:
            return 0        
        return self.weight*(d-self.turnpoint.radius)**2
    
    def gradient(self):
        d = np.sqrt((self.p.x-self.turnpoint.center.x)**2 + (self.p.y-self.turnpoint.center.y)**2)
        if d < self.turnpoint.radius:
            return 0, 0        
        dx = 2*self.weight*(self.p.x-self.turnpoint.center.x)*(d-self.turnpoint.radius)/d
        dy = 2*self.weight*(self.p.y-self.turnpoint.center.y)*(d-self.turnpoint.radius)/d
        return (dx, dy)
    
    def backproject(self):
        d = np.sqrt((self.p.x-self.turnpoint.center.x)**2 + (self.p.y-self.turnpoint.center.y)**2)
        if d < self.turnpoint.radius:
            return
        ax = (self.p.x-self.turnpoint.center.x)
        ay = (self.p.y-self.turnpoint.center.y)
        self.p.x = self.turnpoint.center.x + self.turnpoint.radius/d*ax
        self.p.y = self.turnpoint.center.y + self.turnpoint.radius/d*ay

class SpringSquareLoss():
    ''' This loss is half of a quadratic function of the distance of a path segment: d=sqrt(x^2+y^2). L = 0.5*d^2 = 0.5*x^2 + 0.5^y^2 '''

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def eval(self):
        return 0.5*(self.p1.x-self.p2.x)**2 + 0.5*(self.p1.y-self.p2.y)**2

class SpringSquareLossStart(SpringSquareLoss):
    ''' Computes the gradient in respect of the first point of the path segment'''

    def gradient(self):
        dx = (self.p1.x-self.p2.x)
        dy = (self.p1.y-self.p2.y)
        return (dx, dy)
    
class SpringSquareLossEnd(SpringSquareLoss):
    ''' Computes the gradient in respect of the second point of the path segment'''

    def gradient(self):
        dx = -(self.p1.x-self.p2.x)
        dy = -(self.p1.y-self.p2.y)
        return (dx, dy)

class PathSquareLoss():
    '''The path square loss applies a quadratic loss over the distance of the whole path instead of each segment.
    d = \sum_i{\sqrt{x_i^2 + y_i^2}}. L = 0.5*d^2. The idea is to not penalize segments of different lengths.
    But I think there is still a bug.
    '''

    def __init__(self, points, idx):
        self.ps = points
        self.idx = idx
    
    def eval(self):
        return 0.5*(self.summand()**2)
    
    def summand(self):
        res = 0
        for i in range(len(self.ps)-1):
            a = self.ps[i].x - self.ps[i+1].x
            b = self.ps[i].y - self.ps[i+1].y
            res += np.sqrt(a**2+b**2)
        return res
    
class PathSquareLossStart(PathSquareLoss):

    def gradient(self):
        a = self.ps[self.idx].x - self.ps[self.idx+1].x
        b = self.ps[self.idx].y - self.ps[self.idx+1].y
        divisor = np.sqrt(a**2+b**2)
        summand = self.summand()
        dx = a*summand / (divisor+1e8)
        dy = b*summand / (divisor+1e8)
        return (dx, dy)

class PathSquareLossEnd(PathSquareLoss):

    def gradient(self):
        a = self.ps[self.idx].x - self.ps[self.idx+1].x
        b = self.ps[self.idx].y - self.ps[self.idx+1].y
        divisor = np.sqrt(a**2+b**2)
        summand = self.summand()
        dx = -a*summand / (divisor+1e8)
        dy = -b*summand / (divisor+1e8)
        return (dx, dy)


### Optimizer interface

class GridSearchShortestPath:
    ''' This little gridsearch tries out different settings and selects the
    best performing parameters. Slower runs mean higher iteration limits.'''

    def __init__(self, task):
        self.task = task
    
    def run_fast(self):
        return self.run_many(100)
    
    def run_medium(self):
        return self.run_many(1000)
    
    def run_slow(self):
        return self.run_many(10000)

    def run_many(self, max_iterations):
        # define many (semi) useful options
        lrs = [0.1, 0.01, 0.001]
        iterations = [max_iterations]
        stop_criterias = [10, 100]
        tp_weights = [1, 5, 10, 20, 30]
        backprojects = [True]

        # construct optimizers and paths
        paths = []
        distances = []
        configs = []
        grid = list(product(lrs, iterations, stop_criterias, tp_weights, backprojects))
        for lr, itr, crit, weight, back in tqdm(grid):
            configs.append(f'lr={lr}, itr={itr}, crit={crit}, weight={weight}, back={back}')
            optimizer = ShortestPathOptimizer(self.task, lr, itr, crit, weight, back)
            path = optimizer.shortest_path()
            paths.append(path)
            distances.append(path.distance())
        
        # take the minimal of all paths
        idx = np.nanargmin(distances)
        print(f'found path with minimal distance: {distances[idx]} and config: {configs[idx]}')
        return paths[idx], distances

class ShortestPathOptimizer:
    '''
    The shortest path optimizer is not perfect and only for demonstration purposes. It uses quadratic loss functions to have fast and robust convergence.
    The downside is that the (important) boundary conditions are not respected. The general idea is to guess well enough and then backproject
    to a valid solution.

    Turnpoint Loss: If the point on the path moves outside the turnpoint radius a quadratic term (x^2*tp_weight) will push the point back
    towards the turnpoint.

    Spring Loss: Each segment on the path tries to squeeze itself using a quadratic force (0.5*x^2). It has fast convergence even when initialized
    at the center of each turnpoint. The downside is it has bad convergence at almost straight lines as well as it might favour segments
    of similar length. Because of a^2+a^2 < (2a)^2+0. There is no justification to favor that on a shortest path.

    The combination allows for a quick and robust convergence. Due to the limitations of not correctly respecting the boundaries,
    only an optimization scheme with correct boundary constraints will improve the solutions. But in practice it should be not
    less than 100-500 meters. And the quadratic approach might act as an useful initial state.

    There is no support for a Line-Goal or ESS-optimization.
    '''

    def __init__(self, task, lr=0.01, iterations=100, stop_criteria=-1, tp_weight=20, backproject=True):
        self.task = task
        self.lr = lr
        self.iterations=iterations
        self.init_stop_criteria = stop_criteria
        self.tp_weight = tp_weight
        self.backproject = backproject        
    
    def shortest_path(self, path=None):
        # initialize path (if none given)
        if path is None:            
            path = Path.from_center_points(self.task)

        # create loss functions
        tp_functions = []
        for tp,p in zip(self.task.turnpoints, path.points):
            tp_functions.append(TurnpointSquareLoss(tp, p, self.tp_weight))        
        spring_functions = []
        for i in range(len(path.points)-1):
            spring_functions.append(SpringSquareLossStart(path.points[i], path.points[i+1]))
            spring_functions.append(SpringSquareLossEnd(path.points[i], path.points[i+1]))            

        # Path Loss:
        #spring_functions.append(PathSquareLossStart(path.points, 0))
        #for i in range(len(path.points)-2):
        #    spring_functions.append(PathSquareLossEnd(path.points, i))
        #    spring_functions.append(PathSquareLossStart(path.points, i+1))
        #spring_functions.append(PathSquareLossEnd(path.points, len(path.points)-2))

        # run gradient descent based on the stop criteria (path.distance()) and iterations
        distance = path.distance()
        stop_criteria = self.init_stop_criteria # the amount of iterations without progress.
        try:
            for itr in range(self.iterations):
                
                # compute gradients:
                tp_grads = [func.gradient() for func in tp_functions]
                spring_grads = [func.gradient() for func in spring_functions]
            
                for i in range(len(path.points)):
                    # distribute tp_gards like 1-1-1..-1:
                    grads = [tp_grads[i]]

                    # distribute spring grads like 2-3-3..-2:                   
                    idx = 2*i-1
                    if i == 0:
                        grads = grads + [spring_grads[0]]
                    elif i==len(path.points)-1:
                        grads = grads + [spring_grads[idx]]
                    else:                    
                        grads = grads + [spring_grads[idx], spring_grads[idx+1]]

                    # update step:
                    p = path.points[i]
                    p.x = p.x - self.lr*np.sum([g[0] for g in grads])
                    p.y = p.y - self.lr*np.sum([g[1] for g in grads])

                    # iterative backproject                            
                    if self.backproject:
                        for tp_function in tp_functions:
                            tp_function.backproject()

                    # stop criteria (iterations without progress)
                    current_distance = path.distance()
                    if np.isnan(current_distance):
                        #print(f'early stopping at iteration {itr}: path crashed')                    
                        raise EarlyStopp()                        
                    if stop_criteria > 0:
                        if current_distance > distance:
                            stop_criteria -= 1                        
                        if stop_criteria == 0:
                            #print(f'early stopping at iteration {itr}: no more progress')
                            raise EarlyStopp()                            
                        distance = current_distance
        except EarlyStopp:
            pass

        # backproject to a valid solution
        if self.backproject:
            for tp_function in tp_functions:
                tp_function.backproject()

        return path

if __name__ == '__main__':

    ### Define sample task
    tp1 = Turnpoint(center=Point2f(10, 10), radius=2)
    tp2 = Turnpoint(center=Point2f(20, 30), radius=5)
    tp3 = Turnpoint(center=Point2f(20, 60), radius=9)
    tp4 = Turnpoint(center=Point2f(60, 30), radius=1)
    task = Task([tp1, tp2, tp3, tp4])

    ### Visualize Task
    visualizer = Visualizer()    
    visualizer.equal_aspect()
    visualizer.draw_task(task)

    # create center path
    center_path = Path.from_center_points(task)
    visualizer.draw_path(center_path)

    # create shortest path:
    optimizer = ShortestPathOptimizer(task)
    path = optimizer.shortest_path()
    visualizer.draw_path(path)

    plt.show()

