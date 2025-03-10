
import datetime
import utm
import os
import simplekml
import shutil
import re

from tqdm import tqdm
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize



'''

This library capsules functions about the pilot progression visualization.
There are two executables:
  * The first one computes the optimized distance (igc -> csv)
  * The second one analyzes the csv file for google earth visualizations.

Status: The current library only covers the second part
'''


# UTM and Utilities

UTM_ZONE = 32
UTM_LETTER = 'T'

def canonical_name(name):
    name = name.lower()
    #name = name.replace("ä", "a").replace("ö", "o").replace("ü", "u").replace('é', 'e').replace('è', 'e').replace('ë', 'e').replace('à', 'a')
    name = name.translate(str.maketrans("äöüéèëàç", "aoueeeac"))
    name = re.sub('[^a-zA-Z0-9]+', '-', name)
    name = name.strip('-')
    return name

def print_name(name):
    return name.replace("-", " ").title()

def as_seconds(t):
    return t.hour*3600 + t.minute*60 + t.second

def height_compensation(delta_altitude, delta_distance, delta_time, glide2goal):    
    if delta_altitude < 0:
        # check if goal in reach (do not compensate!):
        if glide2goal < 7:            
            extra_time = 0
            compensated_speed = delta_distance / delta_time
            return extra_time, compensated_speed       

        # 2m height compensation:
        mcgreedy_climb = 2.0
        extra_time = -1*delta_altitude / mcgreedy_climb
        compensated_speed = delta_distance / (delta_time+extra_time)
    else:
        # "virtually glide 60km/h@8"
        glide_speed = 60.0/3.6 # in m/s
        glide_ratio = 7
        glide_distance = delta_altitude*glide_ratio
        extra_time = glide_distance/glide_speed
        compensated_speed = (delta_distance+glide_distance)/(delta_time+extra_time)
    return extra_time, compensated_speed

def height_compensation_speed(row, goal_altitude):
    glide2goal = row['distance'] / (row['gps_alt']-goal_altitude)
    _, speed = height_compensation(row['delta_gps_alt'], row['delta_distance'], row['delta_seconds'], glide2goal)
    return speed

def crop_time(df, start, end):
    df = df[df['time'] >= start]
    df = df[df['time'] < end]
    return df

def crop_distance(df, start, end):
    df = df[df['distance'] <= start]
    df = df[df['distance'] > end]
    return df



# The CSV Pilot processes the csv file

class BoosterTeam:

    def __init__(self):
        self.boosters = []
        with open('pilot_progression/booster.txt', 'r') as file:
            self.boosters = file.readlines()
        self.boosters = [line.strip() for line in self.boosters]

        # duplicate reversed names:
        reversed_lines = []
        for line in self.boosters:
            reversed_lines.append('-'.join(reversed(line.split('-'))))            
        self.boosters += reversed_lines

    
    def contains(self, name):
        name = canonical_name(name)        
        return name in self.boosters

class CsvPilot:

    def __init__(self, directory, filename):
        assert filename.endswith('.csv'), 'PilotReader only deals with csv'
        self.directory = directory
        self.filename = filename
        self.name = filename.split('.')[0]
        self.ranking = 86400 # use for ranking
            
    def process(self, airstart, goal_altitude, start, end):
        df = pd.read_csv(os.path.join(self.directory, self.filename))
        # convert time
        df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').time())
        df['seconds'] = df['time'].apply(lambda t: as_seconds(t) - as_seconds(airstart))
        
        # filter by airstart
        df = df[df['airstart'] == True]

        # check if df contains end:
        if len(df) == 0: # process anyways
        #if len(df) == 0 or df['distance'].min() > end:
            raise ValueError(f'No data for {self.name}')
        
        # check if some distance is missing
        distance_to_end = max(0, df['distance'].min() - end)

        #df = crop_time(df, start, end)
        df = crop_distance(df, start, end)

        # convert x,y coordinates:
        xs, ys, _, _ = utm.from_latlon(df['lat'].values, df['lon'].values)
        df['x'] = xs
        df['y'] = ys

        # Resample in 50m of distance    
        df['distance_bin'] = (df['distance'] // 50) * 50 # Round every 10m
        df = df.groupby('distance_bin').first().reset_index()

        # set index distance
        df.set_index('distance_bin', inplace=True)
        df.sort_index(ascending=False, inplace=True)

        # Compute full track statistics    
        first_row = df.iloc[0]    
        last_row = df.iloc[-1]
        delta_time = last_row['seconds'] - first_row['seconds']
        delta_gps_alt = last_row['gps_alt'] - first_row['gps_alt']
        delta_distance = first_row['distance'] - last_row['distance']
        climb = delta_gps_alt / delta_time
        speed = delta_distance / delta_time
        extra_time, compensated_speed = height_compensation(delta_gps_alt, delta_distance, delta_time, 100) # do compensate!
        print(self.name, 'time:', delta_time, 'altitude', delta_gps_alt, 'distance', delta_distance, 'climb', climb, 'speed', speed*3.6, 'extra_time', extra_time, 'compensated_speed', compensated_speed*3.6)
        
        # ranking (lowest time then lowest distance)
        self.ranking = last_row['seconds'] # use the time elapsed for last row as ranking
        if distance_to_end > 0:
            self.ranking = 86400 + distance_to_end
        

        # compute diffs
        #periods=20 # 1=10m
        #df['delta_seconds'] = df['seconds'].diff(periods=periods)    
        #df['delta_gps_alt'] = df['gps_alt'].diff(periods=periods)    
        #df['delta_distance'] = df['distance'].diff(periods=periods).apply(lambda x: -x)

        # use shifts:
        shift = 20//2 # period = 2*shift
        df['delta_seconds'] = df['seconds'].shift(-shift) - df['seconds'].shift(shift)
        df['delta_gps_alt'] = df['gps_alt'].shift(-shift) - df['gps_alt'].shift(shift)
        df['delta_distance'] = (df['distance'].shift(-shift) - df['distance'].shift(shift)).apply(lambda x: -x)

        # compute segment statistics
        df['climb'] = df['delta_gps_alt'] / df['delta_seconds']
        df['speed'] = (df['delta_distance'] / df['delta_seconds'])    
        df['compensated_speed'] = df.apply(lambda row: height_compensation_speed(row, goal_altitude), axis=1)

        self.df = df


# The KmlView handles everything KML specific

class KmlView:

    def __init__(self, filename):
        self.filename = filename # use as directory:
        shutil.rmtree(self.filename, ignore_errors=True)
        os.mkdir(self.filename)
        #self.kml = simplekml.Kml()
        self.current_file = None
        self.current_folder = None # or self.kml if you want to write in root

    def add_folder(self, name):        
        os.mkdir(f'{self.filename}/{name}/')
        #self.current_folder = self.kml.newfolder(name = name)
        self.current_file = f'{self.filename}/{name}/{name}.kmz'
        self.current_folder = simplekml.Kml()

    def save(self):
        #self.kml.save(self.filename)
        #self.current_folder.save(self.current_file)
        self.current_folder.savekmz(self.current_file)

    '''
    Plot the field 'value' of a full pilots dataframe.
    '''
    def plot(self, df, name, norm, invert_alpha):
        assert self.current_folder is not None, 'Current folder not assigned!'
        
        folder = self.current_folder.newfolder(name=print_name(canonical_name(name)))
        
        # do not show by default:
        hide_childs_style = simplekml.ListStyle(listitemtype=simplekml.ListItemType.checkhidechildren)
        folder.style = simplekml.Style(liststyle=hide_childs_style)
        #folder.open = 0
        folder.visibility = 0
        
        # Create Colors by Value
        c = df['value']
        colors = plt.cm.viridis(norm(c))
    
        alphas = 0.01+0.5*norm(c)
        if invert_alpha:
            alphas = 1 - alphas # the bad!
        #alphas = np.clip(np.abs(norm(c)-0.5)*2, 0., 1.)
 
        alphas = np.clip(np.nan_to_num(alphas, nan=0.5), 0., 1.) # how to handle nans? interpolate?
        #colors[:, -1] = alphas # only required for matplotlib

        # Plot Line Segments using alphas and sizes
        for i in range(len(df)-1):
            lon1 = df['lon'].iloc[i]
            lon2 = df['lon'].iloc[i+1]
            lat1 = df['lat'].iloc[i]
            lat2 = df['lat'].iloc[i+1]
            alt1 = df['gps_alt'].iloc[i]
            alt2 = df['gps_alt'].iloc[i+1]

            # create line segment:
            line = folder.newlinestring()
            line.coords = [(lon1, lat1, alt1), (lon2, lat2, alt2)]

            #line.extrude = 1
            line.altitudemode = simplekml.AltitudeMode.absolute

            # Individuelle Farbe und Linienstärke setzen
            kml_color = simplekml.Color.rgb(int(colors[i][0]*255), int(colors[i][1]*255), int(colors[i][2]*255))
            #line.style.linestyle.color = color  # Farbe aus Liste
            
            line.style.linestyle.color = simplekml.Color.changealphaint(int(100 + 155 * alphas[i]), kml_color)
            line.style.linestyle.width = int(3+alphas[i]*7)  # Max 10 pixels

            # always 10 but chaning color:
            #line.style.linestyle.color = simplekml.Color.changealphaint(150, kml_color)
            #line.style.linestyle.width = 10    
            #print(f"Pilot {name}:", np.min(df['compensated_speed'])*3.6, np.max(df['compensated_speed'])*3.6)
                    
# The Competition visualize groups of pilots

class CsvCompetition:

    def __init__(self, directory, output_prefix):
        self.directory = directory # here are the csv stored
        self.output_prefix = output_prefix
        self.pilots = []
        self.view = None
        self.booster = BoosterTeam()

    def read_pilots(self, airstart, goal_altitude, start, end):
        # read and process all CSV files:
        for filename in os.listdir(self.directory):
            if filename.lower().endswith('.csv'):
                pilot = CsvPilot(self.directory, filename)
                #if not self.booster.contains(pilot.name):
                #    continue # speedup
                try:
                    pilot.process(airstart, goal_altitude, start, end)
                    self.pilots.append(pilot)
                except ValueError:
                    print(f'Pilot {pilot.name} did not reach end. Skip!')
        
        # sort by ranking (lower is better)
        self.pilots = sorted(self.pilots, key=lambda p: p.ranking)

        PRINT_RANKING = True
        if PRINT_RANKING:
            print('Ranking:', '\n'.join([f'{p.name} \t{p.ranking}' for p in self.pilots]))

        self.booster_pilots = list(filter(lambda p: self.booster.contains(p.name), self.pilots))
    
    def create_plots_all_pilots(self):
        self.create_plots('all', self.pilots)

    def create_plots_top5(self):
        self.create_plots('top5', self.pilots[:5])

    def create_plots_top20(self):
        self.create_plots('top20', self.pilots[:20])
        
    def create_plots_booster(self):
        self.create_plots('booster', self.booster_pilots)

    def create_plots(self, postfix, pilots):

        # Initlize KML View:
        self.view = KmlView(f'{self.output_prefix}_{postfix}')

        self.view.add_folder('Climb')
        self.create_sub_plot(pilots, 'climb', False, Normalize(-1, 2))
        self.view.save()

        self.view.add_folder('Climb (Standardized)')
        self.create_sub_plot(pilots, 'climb', False, Normalize(-2, 2), standardize=True)
        self.view.save()

        self.view.add_folder('Sink')
        self.create_sub_plot(pilots, 'climb', True, Normalize(-2, -1))
        self.view.save()

        self.view.add_folder('Progress')        
        self.create_sub_plot(pilots, 'speed', False, Normalize(-2, 2), standardize=True)
        #self.create_sub_plot(pilots, 'speed', False, Normalize())
        self.view.save()

        self.view.add_folder('Progress (Energy compensated)')
        self.create_sub_plot(pilots, 'compensated_speed', False, Normalize(-2, 2), standardize=True)
        self.view.save()
    
    def create_sub_plot(self, pilots, value, invert_alpha, norm, normalize=False, standardize=False):

        # Compute Pilot Aggregation:
        dfs = [pilot.df for pilot in pilots]
        names = [pilot.name for pilot in pilots]
        dfs_value = [df[value] for df in dfs]
        df_all = pd.concat(dfs_value, axis=1, join='outer') # use outer?
        # Replace NaN values in each row with the row mean
        #df_all = df_all.apply(lambda row: row.fillna(row.mean()), axis=1) # required? i dont think so!
        df_all.columns = names
        if normalize:
            df_all = df_all.sub(df_all.min(axis=1), axis=0).div(df_all.max(axis=1) - df_all.min(axis=1), axis=0)
        if standardize:
            df_all = df_all.sub(df_all.mean(axis=1), axis=0).div(df_all.std(axis=1), axis=0)

        # join back:
        for df, name in tqdm(list(zip(dfs, names))):
            col = df_all[name].interpolate()
            df = pd.concat([df, col], axis=1, join='inner')
            df = df.rename(columns={name:'value'})            
            self.view.plot(df, name, norm, invert_alpha)

if __name__ == '__main__':

    # Swiss League Cup March
    airstart = datetime.time(12, 30) # UTC
    d_start = 76000
    d_end = 1000
    goal_altitude = 590
    competition = CsvCompetition('dump/task_2025-03-08', 'dump/slc_march')

    # Regio Beo
    #airstart = datetime.time(11, 35) # UTC
    #competition = CsvCompetition('dump/task_2025-03-01', 'dump/regio_march')
    
    # Run
    competition.read_pilots(airstart, goal_altitude, d_start, d_end)
    competition.create_plots_top5()
    competition.create_plots_top20()
    competition.create_plots_booster()
    competition.create_plots_all_pilots()
    



