
import datetime
import utm
import os
import re
import math

import pandas as pd
import numpy as np

from vpython import scene, sphere, vector, rate, color, cylinder, text, label


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

# END of COPYCODE


# The Pilot Data

class CsvPilot:

    def __init__(self, directory, filename):
        assert filename.endswith('.csv'), 'PilotReader only deals with csv'
        self.directory = directory
        self.filename = filename
        self.name = filename.split('.')[0]        
            
    def process(self, airstart, start, end):
        df = pd.read_csv(os.path.join(self.directory, self.filename))
        # convert time
        df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').time())
        df['seconds'] = df['time'].apply(lambda t: as_seconds(t) - as_seconds(airstart))
        
        # filter before airstart
        df = df[df['airstart'] == False]
        df = crop_time(df, start, end)

        # check if df contains end:
        if len(df) == 0:
            raise ValueError(f'No data for {self.name}')
        
        # convert x,y coordinates:
        xs, ys, _, _ = utm.from_latlon(df['lat'].values, df['lon'].values)
        df['x'] = (xs - 419806.302- 400)
        df['y'] = (ys - 5168141.307)

        # Resample in Time (1 second):
        df.set_index('seconds', inplace=True)
        new_index = np.arange(-900, 0)
        df = df.reindex(new_index)
        df.index.name = 'seconds'

        assert len(df) == 900, 'reindexing went wrong'

        df = df[['x', 'y', 'gps_alt', 'time']]        
        df = df.reset_index()

        # interpolate
        df['x'] = df['x'].interpolate(method='linear')
        df['y'] = df['y'].interpolate(method='linear')
        df['gps_alt'] = df['gps_alt'].interpolate(method='linear')

        # use shifts:
        shift = 4//2 # period = 2*shift
        df['delta_seconds'] = df['seconds'].shift(-shift) - df['seconds'].shift(shift)
        df['delta_gps_alt'] = df['gps_alt'].shift(-shift) - df['gps_alt'].shift(shift)        

        # compute segment statistics
        df['climb'] = df['delta_gps_alt'] / df['delta_seconds']        

        print(f'Process Pilot {self.name}')
        self.df = df

class CsvCompetition:

    def __init__(self, directory):
        self.directory = directory # here are the csv stored        
        self.pilots = []
        self.view = None        

    def read_pilots(self, airstart, start, end):
        # read and process all CSV files:
        counter = 200
        for filename in os.listdir(self.directory):
            if counter == 0:
                return
            counter -= 1
            if filename.lower().endswith('.csv'):
                pilot = CsvPilot(self.directory, filename)                
                try:
                    pilot.process(airstart, start, end)
                    self.pilots.append(pilot)
                except ValueError:
                    print(f'Pilot {pilot.name} error. Skip!')
    
    def animate_pilots(self):
        
        # Setup Scene
        # Simulation parameters
        total_time = 900  # total simulation time in seconds (5 minutes)        
        speedup = 5
        dt = 0.5/speedup      # time step for the simulation
        scene.width = 1920-50
        scene.height = 1080-100

        # access data more easily:
        X = [pilot.df['x'].values for pilot in self.pilots]
        Y = [pilot.df['y'].values for pilot in self.pilots]
        Z = [pilot.df['gps_alt'].values - 2500 for pilot in self.pilots]
        C = [pilot.df['climb'].values for pilot in self.pilots]

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
            if 'fankhauser-benjamin' == self.pilots[i].name:
                start_c = color.yellow
                emissive = False
                
            sp = sphere(pos=vector(start_x, start_y, start_z),
                radius=10,
                make_trail=True, retain=25, trail_radius=1,
                emissive=emissive, color=start_c)
            pilots.append(sp)
        
        # Add axes:
        L = 500
        R = L/100
        d = L-2
        xaxis = cylinder(pos=vector(0,0,0), axis=vector(d,0,0), radius=R, color=color.yellow)
        yaxis = cylinder(pos=vector(0,0,0), axis=vector(0,d,0), radius=R, color=color.yellow)
        zaxis = cylinder(pos=vector(0,0,0), axis=vector(0,0,d), radius=R, color=color.yellow)        
        k = 1.02
        h = 0.05*L
        text(pos=xaxis.pos+k*xaxis.axis, text='x', height=h, align='center', billboard=True, emissive=True)
        text(pos=yaxis.pos+k*yaxis.axis, text='y', height=h, align='center', billboard=True, emissive=True)
        text(pos=zaxis.pos+k*zaxis.axis, text='z', height=h, align='center', billboard=True, emissive=True)
        txt_timer = label(pos=zaxis.pos+1.07*zaxis.axis, text='timer', height=h, align='center', box=False, border=0., emissive=True)

        # Run Animation!
        # Animation loop
        t = 0
        while t < total_time-1:
            rate(speedup/dt)  # controls the simulation speed

            t_j = math.floor(t)                        
            tt = t - t_j # interpolation alpha            

            txt_timer.text = f'{t_j-total_time}s'            

            for i in range(n_pilots):
                # Calculate the new angle for this pilot

                # Interpolate
                new_x = (1.-tt)*X[i][t_j] + (tt)*X[i][t_j+1]                
                new_y = (1.-tt)*Y[i][t_j] + (tt)*Y[i][t_j+1]
                new_z = (1.-tt)*Z[i][t_j] + (tt)*Z[i][t_j+1]    
                climb = (1.-tt)*C[i][t_j] + (tt)*C[i][t_j+1]            

                pilots[i].pos = vector(new_x, new_y, new_z)

                #if i == 0 and tt < 0.1:
                #    print('climb', climb)

    
                # Change the color based on the altitude:
                # r, g, b values vary with cosine and sine functions to produce a smooth transition.
                vmin = -1
                vmax = 3
                r_val = np.clip(climb-vmin/(vmax-vmin), 0, 1) #(math.cos(theta) + 1) / 2    # Normalize cosine to [0,1]
                g_val = 0.2 #(math.sin(theta) + 1) / 2      # Normalize sine to [0,1]
                b_val = 0.1 #(math.cos(theta + math.pi/2) + 1) / 2  # Phase-shifted cosine for variation
                
                pilots[i].color = vector(r_val, g_val, b_val)

            t += dt


# main:
if __name__ == '__main__':

    # Swiss League Cup March
    airstart = datetime.time(12, 30) # UTC
    t_start = datetime.time(12, 15)
    t_end = airstart    
    competition = CsvCompetition('dump/task_2025-03-08')
    competition.read_pilots(airstart, t_start, t_end)
    competition.animate_pilots()



