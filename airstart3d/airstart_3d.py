
import datetime
import utm
import os
import re
import math
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
from PIL import Image

from vpython import scene, canvas, sphere, vector, rate, color, cylinder, text, label, box, textures

from sklearn.cluster import KMeans, AgglomerativeClustering

from airstart3d.elevation import plot3D, read_elevation_data

'''
This Script reads the 3d data from the pilots in the start thermal.
It interpolates the data and visualizes movements in 3d.
'''

# Utilites:

def crop_time(df, start, end):
    df = df[df['time'] >= start]
    df = df[df['time'] < end]
    return df

def as_seconds(t):
    return t.hour*3600 + t.minute*60 + t.second

tile_server_url = "https://api.maptiler.com/maps/landscape/{z}/{x}/{y}.png?key=Pnj84XMbfpc5r0Ir1bLb"

def create_url(x, y, zoom):
    return tile_server_url.format(z=zoom, x=x, y=y)

def tile_to_latlon(x_tile, y_tile, zoom):
    n = 2.0 ** zoom
    lon = (x_tile / n) * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y_tile / n))))
    lat = math.degrees(lat_rad)
    return lat, lon

TILE_SIZE = 512
def latlon_to_tile(lat, lon, zoom):
    # Convert latitude to Mercator projection
    lat_rad = math.radians(lat)
    
    # Number of tiles at this zoom level
    n = 2.0 ** zoom

    # Calculate tile coordinates
    x = (lon + 180.0) / 360.0 * n * TILE_SIZE
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE

    # Tile number
    x_tile = int(x // TILE_SIZE)
    y_tile = int(y // TILE_SIZE)

    # Pixel offset inside the tile
    x_offset = int(x % TILE_SIZE)
    y_offset = int(y % TILE_SIZE)

    return x_tile, y_tile, x_offset/TILE_SIZE, y_offset/TILE_SIZE

EARTH_CIRCUMFERENCE = 40075016.6856  # in meters (equatorial)

def webmercator_tile_size(lat, zoom):    
    lat_rad = math.radians(lat)   
    num_tiles = 2 ** zoom    
    tile_width_meters = (EARTH_CIRCUMFERENCE * math.cos(lat_rad)) / num_tiles    
    tile_height_meters = EARTH_CIRCUMFERENCE / num_tiles
    return tile_width_meters, tile_height_meters

# END of COPYCODE


# The Pilot Data

class CsvPilot:

    def __init__(self, directory, filename):
        assert filename.endswith('.csv'), 'PilotReader only deals with csv'
        self.directory = directory
        self.filename = filename
        self.name = filename.split('.')[0]        
            
    def process(self, airstart, start, end, nrows=None):
        df = pd.read_csv(os.path.join(self.directory, self.filename), nrows=nrows)
        # convert time
        df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').time())
        df['seconds'] = df['time'].apply(lambda t: as_seconds(t) - as_seconds(airstart))
        
        # filter before airstart
        #df = df[df['airstart'] == False]
        df = crop_time(df, start, end)

        # check if df contains end:
        if len(df) == 0:
            raise ValueError(f'No data for {self.name}')
        
        # convert x,y coordinates:
        xs, ys, _, _ = utm.from_latlon(df['lat'].values, df['lon'].values)
        df['x'] = xs # (xs - 419806.302- 400)
        df['y'] = ys # (ys - 5168141.307)

        # Resample in Time (1 second):
        range_start = as_seconds(start) - as_seconds(airstart)
        range_end = as_seconds(end) - as_seconds(airstart)
        df.set_index('seconds', inplace=True)
        new_index = np.arange(range_start, range_end)
        df = df.reindex(new_index)
        df.index.name = 'seconds'

        df = df[['x', 'y', 'gps_alt', 'pressure_alt', 'time']]        
        df = df.reset_index()

        # interpolate
        df['x'] = df['x'].interpolate(method='linear')
        df['y'] = df['y'].interpolate(method='linear')
        df['gps_alt'] = df['gps_alt'].interpolate(method='linear')

        # compute distances
        df['utm_distance'] = np.sqrt((df['x'].diff())**2 + (df['y'].diff())**2)

        # use shifts:
        shift = 4//2 # period = 2*shift
        df['delta_seconds'] = df['seconds'].shift(-shift) - df['seconds'].shift(shift)
        df['delta_gps_alt'] = df['gps_alt'].shift(-shift) - df['gps_alt'].shift(shift)  
        df['delta_pressure_alt'] = df['pressure_alt'].shift(-shift) - df['pressure_alt'].shift(shift)  
        df['delta_utm_distance'] = df['utm_distance'].rolling(window=2*shift+1, center=True).sum() #df['utm_distance'].shift(-shift) - df['utm_distance'].shift(shift)  


        # compute segment statistics
        df['gps_climb'] = df['delta_gps_alt'] / df['delta_seconds']        
        df['pressure_climb'] = df['delta_pressure_alt'] / df['delta_seconds']
        df['utm_speed'] = 3.6*df['delta_utm_distance'] / df['delta_seconds']
        #df['utm_acceleration'] = df['delta_utm_speed'] / df['delta_seconds']   

        #print(f'Process Pilot {self.name}')
        self.df = df

class CsvCompetition:

    def __init__(self, directory):
        self.directory = directory # here are the csv stored        
        self.pilots = []
        self.view = None        

    def read_pilots(self, airstart, start, end):
        self.total_time = as_seconds(end) - as_seconds(start) # total time in seconds
        # read and process all CSV files:
        counter = 200
        for filename in tqdm(list(os.listdir(self.directory))):
            if counter == 0:
                return
            counter -= 1
            if filename.lower().endswith('.csv'):
                pilot = CsvPilot(self.directory, filename)                
                try:
                    nrows = 2*60*60 # only first two hours
                    pilot.process(airstart, start, end, nrows)
                    self.pilots.append(pilot)
                except ValueError:
                    print(f'Pilot {pilot.name} error. Skip!')
    
    def compute_thermal_centroids(self):
        # Compute Climb Centroid
        value = 'pressure_climb'
        top20 = 50
        self.n_thermals = 5
        dfs = [p.df for p in self.pilots]
        names = [p.name for p in self.pilots]
        df_all = pd.concat(dfs, keys=names, names=['pilot']).reset_index()
        df_all = df_all[df_all['x'] <=  419806 + 1200] # remove outliers
        df_sorted = df_all.sort_values(by=['seconds', value], ascending=[True, False])
        df_top20 = df_sorted.groupby('seconds').head(top20).reset_index()
        #self.df_thermal = df_top20.groupby('seconds')[['x', 'y', 'pressure_alt', 'gps_alt', 'pressure_climb']].mean().reset_index()

        self.df_thermals = []
        for i in range(self.n_thermals):
            self.df_thermals.append(
                {'seconds':[], 'x':[], 'y':[], 'gps_alt':[], 'pressure_climb':[]}
            )
        
        previous_centroids = None
        for seconds, group in tqdm(df_top20.groupby('seconds')):
            coords = group[['x', 'y', 'gps_alt']].values

            if previous_centroids is None:
                kmeans = KMeans(n_clusters=self.n_thermals, n_init=10)
            else:
                kmeans = KMeans(n_clusters=self.n_thermals, init=np.array(previous_centroids), n_init=1)
            kmeans.fit(coords)            
            
            for i,centroid in enumerate(kmeans.cluster_centers_):
                self.df_thermals[i]['seconds'].append(seconds)                
                self.df_thermals[i]['x'].append(centroid[0])
                self.df_thermals[i]['y'].append(centroid[1])
                self.df_thermals[i]['gps_alt'].append(centroid[2])
                weight = 0.7 # (kmeans.labels_ == i).sum() *3 / top20
                self.df_thermals[i]['pressure_climb'].append(weight * group[kmeans.labels_ == i]['pressure_climb'].mean())
            previous_centroids = kmeans.cluster_centers_

            # Use Agglomerative Clustering:
            #agglo = AgglomerativeClustering(n_clusters=self.n_thermals)
            #labels = agglo.fit_predict(coords)
            #centroids = [coords[labels == i].mean(axis=0) for i in range(self.n_thermals)]
            #for i,centroid in enumerate(centroids):
            #    self.df_thermals[i]['seconds'].append(seconds)                
            #    self.df_thermals[i]['x'].append(centroid[0])
            #    self.df_thermals[i]['y'].append(centroid[1])
            #    self.df_thermals[i]['gps_alt'].append(centroid[2])
            #    weight = 0.7 # (kmeans.labels_ == i).sum() *3 / top20                
            #    self.df_thermals[i]['pressure_climb'].append(weight * group[labels == i]['pressure_climb'].mean())
            
            


        self.df_thermals = [pd.DataFrame(data) for data in self.df_thermals]
            
    
    def plot_integrated_climb(self):

        # collect values:
        #self.plot_column_statistics('gps_climb')
        #self.plot_column_statistics('pressure_climb')
        self.plot_column_statistics('utm_speed')
        for pilot in self.pilots:
            if 'fankhauser-benjamin' == pilot.name:
                plt.plot(pilot.df.index, pilot.df['utm_speed'])
        
        plt.legend()
        plt.grid()
        plt.show()
    
    def plot_column_statistics(self, value):
        dfs = [p.df for p in self.pilots]
        dfs_value = [df[value] for df in dfs]
        df_all = pd.concat(dfs_value, axis=1, join='inner')

        # compute statistics per row:
        #df_all.apply(lambda row: row.mean(), axis=1)
        df_all['mean'] = df_all.mean(axis=1)
        df_all['std'] = df_all.std(axis=1)

        plt.plot(df_all.index, df_all['mean'], label='Mean')
        plt.fill_between(df_all.index, df_all['mean'] - df_all['std'], df_all['mean'] + df_all['std'], alpha=0.3, label='Std')



    def animate_pilots(self, fix_pilot=False):
        
        # Setup Scene        
        scene = canvas(width=1920-50, height=1080-100, resizable=True, background=vector(44/255, 44/255, 45/255))

        # Simulation parameters
        speedup = 5
        dt = 0.5/speedup      # time step for the simulation
        #scene.width = 1920-50
        #scene.height = 1080-100                

        # access data more easily:
        X = [pilot.df['x'].values for pilot in self.pilots]
        Y = [pilot.df['gps_alt'].values for pilot in self.pilots]
        Z = [-pilot.df['y'].values for pilot in self.pilots]        

        # compute origin
        origin = vector(np.array(X)[:, 0].mean(), 2500, np.array(Z)[:, 0].mean())
        print('Origin at: ', origin)
        
        # find top-left positions for map
        top_left = vector(np.array(X).min(), 1000, np.array(Z).min()) # move up and down here!
        top_left_local = top_left - origin # in local coordinates, 
        #center_offset_x = w - top_left_x
        #center_offset_z = center_z - top_left_z

        # Create Ground Box
        UTM_ZONE = 32
        UTM_LETTER = 'T'        
        lat, lon = utm.to_latlon(top_left.x, -top_left.z, UTM_ZONE, UTM_LETTER)        
        zoom = 13
        x, y, off_x, off_y = latlon_to_tile(lat, lon, zoom)
        print(f"Tile coordinates for zoom {zoom}: x = {x}, y = {y}, using the offset: {off_x}/{off_y}")        
        width, height = webmercator_tile_size(lat, zoom)        
        map_offset = vector((off_x-0.5)*width, 0, (off_y-0.5)*width) # offset form top-left to map-top_left
        map_offset_elevation = vector(off_x*width, 0, off_y*width)
        
        # print for elevation data:
        tile_lat, tile_lon = tile_to_latlon(x, y, zoom)
        print(f'get elevation at lat={tile_lat}, lon={tile_lon}, width={width}')        
        
        # create 3x3 grid
        grid_size = 1
        grid = list(itertools.product(range(grid_size), repeat=2))
        for coord in grid:
            i,j = coord
            box_pos = top_left - map_offset # central position

            # move in grid
            box_pos.x += i*width
            box_pos.z += j*width            
            
            # create ground box
            ground_box = box(pos=box_pos-origin, size=vector(width, 0.1, width)) 
            ground_box.texture = create_url(x+i, y+j, zoom)
            ground_box.bumpmap = "airstart_3d/swiss_cup_flex_march.png"

            print(f"ground box: pos={ground_box.pos}")

            #image = Image.open("/home/benjamin/Pictures/swiss_cup_flex_march.png")
            #ground_box.texture = textures.texture(data=image)
            #round_box.texture = "airstart_3d/swiss_cup_flex_march.png"



        # Elevation        
        elevation_data = read_elevation_data(tile_lon, tile_lat, width)
        assert elevation_data.shape[0] == elevation_data.shape[1], 'elevation must be square'
        #def f(x, y):            
        #    return elevation_data[x,y]        
        f = lambda x,y: elevation_data[x,y] - origin.y

        plot_pos = top_left_local - map_offset_elevation
        
        # requires some funky transposition:
        p = plot3D(f, elevation_data.shape[0],plot_pos.z, plot_pos.z+width, plot_pos.x, plot_pos.x+width, 0, 1000, texture=create_url(x, y, zoom))


        # PIlOTS:


        #C = [pilot.df['gps_climb'].values for pilot in self.pilots]
        C = [pilot.df['pressure_climb'].values for pilot in self.pilots]
        #C = [np.abs(pilot.df['gps_climb'].values - pilot.df['pressure_climb'].values) for pilot in self.pilots]

        # statistics:
        print('Value statistics: mean', np.nanmean(C), 'std', np.nanstd(C), 'min', np.nanmin(C), 'max', np.nanmax(C))

        # create pilots:
        pilots = []
        n_pilots = len(self.pilots)

        for i in range(n_pilots):
            # Starting position: at angle 0 on the circle for each pilot
            start_x = X[i][0]
            start_y = Y[i][0]
            start_z = Z[i][0]
            start_c=color.red
            emissive = False
            retain = 25
            if 'fankhauser-benjamin' == self.pilots[i].name:
                start_c = color.yellow
                emissive = False
                retain = 100
                
            sp = sphere(pos=vector(start_x, start_y, start_z)-origin,
                radius=10,
                make_trail=True, retain=retain, trail_radius=1,
                emissive=emissive, color=start_c)
            pilots.append(sp)
        
        # Add Massive Thermal Sphere
        #if self.df_thermals is not None:
        thermal_X = [df['x'].values for df in self.df_thermals]
        thermal_Y = [df['gps_alt'].values for df in self.df_thermals]
        thermal_Z = [-df['y'].values for df in self.df_thermals]
        thermal_C = [df['pressure_climb'].values for df in self.df_thermals]
        
        # center correction
        #thermal_X = [x-center_x for x in thermal_X]
        #thermal_Z = [z-center_z for z in thermal_Z]

        thermals = [sphere(pos=vector(0, 0, 0), radius=100, opacity=0.1, color=color.yellow) for _ in range(self.n_thermals)]

        # Add axes:
        L = 500
        R = L/100
        d = L-2
        xaxis = cylinder(pos=vector(0,0,0), axis=vector(d,0,0), radius=R, color=color.yellow)
        yaxis = cylinder(pos=vector(0,0,0), axis=vector(0,d,0), radius=R, color=color.yellow)
        zaxis = cylinder(pos=vector(0,0,0), axis=vector(0,0,-d), radius=R, color=color.yellow)        
        k = 1.02
        h = 0.05*L
        text(pos=xaxis.pos+k*xaxis.axis, text='east', height=h, align='center', billboard=True, emissive=True)
        text(pos=yaxis.pos+k*yaxis.axis, text='alt', height=h, align='center', billboard=True, emissive=True)
        text(pos=zaxis.pos+k*zaxis.axis, text='north', height=h, align='center', billboard=True, emissive=True)
        txt_timer = label(pos=yaxis.pos+1.1*yaxis.axis, text='timer', height=h, align='center', box=False, border=0., emissive=True)

        # find pilot to follow:
        follow_pilot = None
        if fix_pilot:
            for i in range(n_pilots):
                if self.pilots[i].name == 'fankhauser-benjamin': # could be faster
                    follow_pilot = i
                    print('follow pilot: ', follow_pilot)
                    print('current camera:', scene.camera.pos)
                    break

        #################
        # Run Animation!#
        #################
        
        t = 0
        # offset for pilot fixation
        prev_pos = None        
        offset_x = 0
        offset_z = 0
        while t < self.total_time-1:
            rate(speedup/dt)  # controls the simulation speed

            t_j = math.floor(t)                        
            tt = t - t_j # interpolation alpha            

            txt_timer.text = f'{t_j-self.total_time}s'            

            for i in range(n_pilots):
                # Calculate the new angle for this pilot

                # Interpolate
                new_x = (1.-tt)*X[i][t_j] + (tt)*X[i][t_j+1]                
                new_y = (1.-tt)*Y[i][t_j] + (tt)*Y[i][t_j+1]
                new_z = (1.-tt)*Z[i][t_j] + (tt)*Z[i][t_j+1]    
                col = (1.-tt)*C[i][t_j] + (tt)*C[i][t_j+1]            

                pilots[i].pos = vector(new_x, new_y, new_z)-origin


                #if i == 0 and tt < 0.1:
                #    print('climb', climb)               
    
                # Change the color based on the altitude:
                # r, g, b values vary with cosine and sine functions to produce a smooth transition.
                vmin = -1
                vmax = 3
                #norm = Normalize(vmin=vmin, vmax=vmax)
                #rgb = plt.cm.viridis((col-vmin)/(vmax-vmin))
                #pilots[i].color = vector(rgb[0], rgb[1], rgb[2])
                r_val = np.clip((col-vmin)/(vmax-vmin), 0, 1)
                g_val = 0.2 
                b_val = 0.1
                pilots[i].color = vector(r_val, g_val, b_val)                
            
            # Update thermal sphere:
            for i in range(self.n_thermals):
                new_x = (1.-tt)*thermal_X[i][t_j] + (tt)*thermal_X[i][t_j+1]                
                new_y = (1.-tt)*thermal_Y[i][t_j] + (tt)*thermal_Y[i][t_j+1]
                new_z = (1.-tt)*thermal_Z[i][t_j] + (tt)*thermal_Z[i][t_j+1]    
                col = (1.-tt)*thermal_C[i][t_j] + (tt)*thermal_C[i][t_j+1]
                thermals[i].pos = vector(new_x, new_y, new_z)-origin

                # update radius and opacity:            
                thermals[i].opacity = np.clip((col-0)/3., 0.1, 1.)
                thermals[i].radius = 100*(col)/3.

            # follow pilot:
            if follow_pilot is not None:
                if prev_pos is not None:
                    #print('follow pilot pos:', pilots[follow_pilot].pos, 'prevpos:', prev_pos)
                    
                    offset = pilots[follow_pilot].pos - prev_pos
                    #print('camera pos', scene.camera.pos, '\toffset:', offset)                
                    #scene.camera.pos.x += offset.x
                    #scene.camera.pos.z += offset.z
                    scene.camera.pos = vector(scene.camera.pos.x + offset.x, scene.camera.pos.y, scene.camera.pos.z+offset.z)
                pos = pilots[follow_pilot].pos
                prev_pos = vector(pos.x, pos.y, pos.z)
            
                
            # update time
            t += dt


# main:
if __name__ == '__main__':

    # Swiss League Cup March
    airstart = datetime.time(12, 30) # UTC
    t_start = datetime.time(12, 15)
    t_end = datetime.time(12, 30)
    competition = CsvCompetition('dump/task_2025-03-08')
    competition.read_pilots(airstart, t_start, t_end)
    competition.compute_thermal_centroids()
    #competition.plot_integrated_climb()
    competition.animate_pilots(fix_pilot=False)



