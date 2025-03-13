

import rasterio
import numpy as np
import os

from vpython import *

import matplotlib.pyplot as plt

'''
This script reads elevation data from the EU-DEM project
'''


''' i will borrow that: '''
class plot3D:
    def __init__(self, f, L, xmin, xmax, ymin, ymax, zmin, zmax, texture=None):
        # The x axis is labeled y, the z axis is labeled x, and the y axis is labeled z.
        # This is done to mimic fairly standard practive for plotting
        #     the z value of a function of x and y.
        self.f = f
        self.texture = texture
        if not xmin: self.xmin = 0
        else: self.xmin = xmin
        if not xmax: self.xmax = 1
        else: self.xmax = xmax
        if not ymin: self.ymin = 0
        else: self.ymin = ymin
        if not ymax: self.ymax = 1
        else: self.ymax = ymax
        if not zmin: self.zmin = 0
        else: self.zmin = zmin
        if not zmax: self.zmax = 1
        else: self.zmax = zmax
        
        self.L = L
        #R = L/100
        #d = L-2
        #xaxis = cylinder(pos=vec(0,0,0), axis=vec(0,0,d), radius=R, color=color.yellow)
        #yaxis = cylinder(pos=vec(0,0,0), axis=vec(d,0,0), radius=R, color=color.yellow)
        #zaxis = cylinder(pos=vec(0,0,0), axis=vec(0,d,0), radius=R, color=color.yellow)
        #k = 1.02
        #h = 0.05*L
        #text(pos=xaxis.pos+k*xaxis.axis, text='x', height=h, align='center', billboard=True, emissive=True)
        #text(pos=yaxis.pos+k*yaxis.axis, text='y', height=h, align='center', billboard=True, emissive=True)
        #text(pos=zaxis.pos+k*zaxis.axis, text='z', height=h, align='center', billboard=True, emissive=True)
    
        self.vertices = []
        for x in range(L):
            for y in range(L):
                val = self.evaluate(x,y)
                x_rel = (x/L)*(self.xmax-self.xmin)+self.xmin
                y_rel = (y/L)*(self.ymax-self.ymin)+self.ymin
                self.vertices.append(self.make_vertex(x_rel, y_rel, x, y, val ))
        
        self.make_quads()
        self.make_normals()
        
    def evaluate(self, x, y):
        return self.f(x, y) # absolute evaluation
        # 
        #d = self.L-2
        #return (d/(self.zmax-self.zmin)) * (self.f(self.xmin+x*(self.xmax-self.xmin)/d, self.ymin+y*(self.ymax-self.ymin)/d)-self.zmin)
    
    def make_quads(self):
        # Create the quad objects, based on the vertex objects already created.
        for x in range(self.L-2):
            for y in range(self.L-2):
                v0 = self.get_vertex(x,y)
                v1 = self.get_vertex(x+1,y)
                v2 = self.get_vertex(x+1, y+1)
                v3 = self.get_vertex(x, y+1)
                quad(vs=[v0, v1, v2, v3], texture=self.texture)
                #quad(vs=[v2, v1, v0, v3], texture=self.texture)
        
    def make_normals(self):
        # Set the normal for each vertex to be perpendicular to the lower left corner of the quad.
        # The vectors a and b point to the right and up around a vertex in the xy plance.
        for i in range(self.L*self.L):
            x = int(i/self.L)
            y = i % self.L
            if x == self.L-1 or y == self.L-1: continue
            v = self.vertices[i]
            a = self.vertices[i+self.L].pos - v.pos
            b = self.vertices[i+1].pos - v.pos
            v.normal = cross(a,b)
    
    def replot(self):
        for i in range(self.L*self.L):
            x = int(i/self.L)
            y = i % self.L
            v = self.vertices[i]
            v.pos.y = self.evaluate(x,y)
        self.make_normals()
                
    def make_vertex(self,x,y,tex_x, tex_y, value):        
        #texpos = vector(tex_x/self.L, tex_y/self.L, 0)
        texpos = vector(tex_y/self.L, 1.-tex_x/self.L, 0)
        return vertex(pos=vec(y,value,x), texpos=texpos, shininess=0.01, normal=vec(0,1,0))
        
    def get_vertex(self,x,y):
        return self.vertices[x*self.L+y]
        
    def get_pos(self,x,y):
        return self.get_vertex(x,y).pos

''' end of it '''


def get_tile_coordinates(lat, lon, tile_size=5):
    # Determine the bottom-left corner of the 5x5 degree tile
    lat_min = (lat // tile_size) * tile_size
    lon_min = (lon // tile_size) * tile_size

    # Calculate the top-right corner of the tile
    lat_max = lat_min + tile_size
    lon_max = lon_min + tile_size

    return lat_min, lon_min, lat_max, lon_max

def extract_elevation_from_tile(tile_path, lat, lon, grid_size_m=25):
    with rasterio.open(tile_path) as src:
        # Convert lat/lon to pixel coordinates within the tile
        row, col = src.index(lon, lat)
        
        # Calculate the pixel buffer for the 25x25m grid around the point
        # We are working with EU-DEM (25m resolution), so the grid size is 25m
        window = rasterio.windows.Window(col, row, grid_size_m, grid_size_m)
        
        # Read the data for the 25x25m window
        elevation_data = src.read(1, window=window)
        
        return elevation_data

def read_elevation_data(lon, lat, width):
    tile_path = "/home/benjamin/Downloads/EU_DEM_mosaic_5deg/eudem_dem_4258_europe.tif"

    import os.path
    if not os.path.isfile(tile_path):
        return np.load("airstart3d/elevation_data.npy")

    with rasterio.open(tile_path) as src:
        row, col = src.index(lon, lat)
        grid = int(width//25)
        half_grid = grid // 2
        #window = rasterio.windows.Window(col-half_grid, row-half_grid, grid, grid)
        window = rasterio.windows.Window(col, row, grid, grid)
        return src.read(1, window=window)    

if __name__ == '__main__':
        


    # Setup Scene Lights
    for i in range( 0, len(scene.lights) ):
        LL = scene.lights[i]
        LL.color *= 0.2
        print( LL.pos, LL.direction, LL.color )        
    scene.ambient = color.white * 0.5

    # Coordinates for Grindelwald, Switzerland
    #lat = 46.6242
    #lon = 8.0390

    # Swiss Cup Flex:
    #lat=46.66903082499045
    #lon=7.957116401763962
    #width=3356.926548743599

    # in shorter
    lat=46.679594465640186
    lon=7.9541015625    
    width=3356.9786285001073
    #width *= 4

    # 1. Determine the 5x5 degree tile for Grindelwald
    #lat_min, lon_min, lat_max, lon_max = get_tile_coordinates(lat, lon)

    #print(f"Tile coordinates: ({lat_min}, {lon_min}) to ({lat_max}, {lon_max})")

    # 2. Assuming you already have the tile downloaded, specify the tile file path
    #tile_file_path = f"path_to_your_tiles/DEM_{lat_min}_{lon_min}_to_{lat_max}_{lon_max}.tif"

    # 3. Extract elevation data for the 25x25m grid around Grindelwald
    #elevation_grid = extract_elevation_from_tile(tile_file_path, lat, lon)
    #print(f"Elevation grid (25x25m) for Grindelwald:\n{elevation_grid}")

    elevation_data = read_elevation_data(lon, lat, width)


    USE_MATPLOTLIB = False
    if USE_MATPLOTLIB:
        # plot contour
        X = np.arange(grid)
        Y = -np.arange(grid)
        X, Y = np.meshgrid(X, Y)

        fig, ax = plt.subplots()
        # iso lines:
        #ax.contour(X, Y, elevation_data, levels=50)

        img = ax.imshow(elevation_data, cmap="gray", origin="upper")


        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)
        ax.axis('off')
        plt.savefig('airstart_3d/swiss_cup_flex_march.png', dpi=150, bbox_inches='tight', pad_inches=0, transparent=False)
        plt.show()
    
    else:
        # use vpython

        L = elevation_data.shape[0]
        #L = L // 4
        if scene is None:
            scene = canvas(title="3D Elevation Profile", width=800, height=400)
            scene.center = vec(0.05*L,0.2*L,0)
            scene.range = 1.3*L

        t = 0
        dt = 0.02
        def f(x, y):
            # Return the value of the function of x and y:
            #return 0.7+0.2*sin(10*x)*cos(10*y)*sin(5*t)
            #grid = int(width//25)
            #i = np.clip(int(x*grid), 0, grid-1)
            #j = np.clip(int(y*grid), 0, grid-1)
            #return elevation_data[x*4,y*4]-2000
            return elevation_data[x,y]-2000

        np.save("airstart3d/elevation_data.npy", elevation_data)

        #660.643, -1500, -495.034>
        p = plot3D(f, L, 0, width, 0, width, 0, 1000) # function, xmin, xmax, ymin, ymax (defaults 0, 1, 0, 1, 0, 1)



    '''

            # Create the VPython scene
            scene = canvas(title="2D Elevation Profile", width=800, height=400)

            # Generate synthetic elevation data (replace with real data)
            x_vals = np.linspace(0, width, grid)
            y_vals = np.linspace(0, width, grid)
            x,y = np.meshgrid(x_vals, y_vals)

            # Create the curve for elevation
            elevation_curve = curve(color=color.white)

            # Add points to the curve
            for i in range(grid):
                for j in range(grid):
                    elevation_curve.append(vector(x_vals[i], y_vals[j], elevation_data[i, j]))        

            # Add axis lines
            #curve(vector(-10, 0, 0), vector(10, 0, 0), color=color.red)  # X-axis
            #curve(vector(-10, -5, 0), vector(-10, 5, 0), color=color.blue)  # Elevation axis


            # Generate synthetic elevation data (for example purposes)
            #size = 200  # Grid size
            size = grid
            x_vals = np.linspace(0, width, size)
            y_vals = np.linspace(0, width, size)
            X, Y = np.meshgrid(x_vals, y_vals)
            #Z = np.sin(X) * np.cos(Y) * 5  # Sample elevation function
            Z = elevation_data

            # VPython scene setup
            scene = canvas(title="Elevation Plot", width=800, height=600)

            # Create points in 3D space
            for i in range(size):
                for j in range(size):
                    x, y, z = X[i, j], Z[i, j], -Y[i, j]
                    sphere(pos=vector(x, y, z), radius=25, color=vector(0.7, 0.7, 0.7))  # Gray color


    '''



    
