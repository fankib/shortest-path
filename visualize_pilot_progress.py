import datetime
import utm
import os
import simplekml

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from mpl_toolkits.mplot3d import Axes3D

# Global Setup: Kml
kml = simplekml.Kml()


def plot_pilot_3d(df, name):

    # plot line
    ax.plot(df['x'], df['y'], df['gps_alt'], label = name)

    # plot line in kml:
    folder = kml.newfolder(name=name)
    #lat, lon = utm.to_latlon(df['x'], df['y'], UTM_ZONE, UTM_LETTER)
    coordinates = list(zip(df['lon'],df['lat'],df['gps_alt']))
    color = ax.lines[-1].get_color()
    kml_color = simplekml.Color.rgb(int(color[0]*255), int(color[1]*255), int(color[2]*255))

    # do not plot pilot line:
    '''
    pilot_line = folder.newlinestring(name=f'Track {name}')
    pilot_line.coords = coordinates
    pilot_line.altitudemode = simplekml.AltitudeMode.absolute

    # Individuelle Farbe und Linienstärke setzen
    #line.style.linestyle.color = color  # Farbe aus Liste
    pilot_line.style.linestyle.color = kml_color #simplekml.Color.changealphaint(int(255 * width/40), simplekml.Color.red)
    pilot_line.style.linestyle.width = 2 # Dicke aus Liste
    s'''

    #sc = ax.scatter(df['x'], df['y'], df['gps_alt'], marker='o', label=name)    
    #sc = ax.scatter(df['x'], df['y'], df['gps_alt'], c=df['compensated_speed'], vmin=10/3.6, vmax=45/3.6, cmap='viridis', marker='o', label=name) 
    
    
    # create own colors:
    c = df['value']

    # what to do with nans?

    # standardize norm
    #norm = Normalize(vmin=-2, vmax=2)

    # climb norm:
    norm = Normalize(vmin=-2, vmax=-1)

    #colors = plt.cm.viridis(norm(c))
    
    

def plot_pilot(df, name):    

    # Reduce the points
    #reduction = 10 # aggregate this many rows
    #df = df[df.index % reduction == 0]
    #df['gps_alt'] = df['gps_alt']/1700*30000
    
    value = 'gps_alt'
    #df = df.groupby(df.index // reduction).agg({'time': 'first', 'seconds': 'first', 'distance': 'mean', 'gps_alt': 'mean', 'speed': 'mean', 'climb': 'mean'})
    
    #df['time'] = np.arange(len(df))
    #df = df[df['time'] < 70.]

    #sns.lineplot(data=df, x='seconds', y='distance', marker='o', label=name)
    #sns.lineplot(data=df, x='seconds', y='gps_alt', label=f'{name} GPS Alt')


#sns.scatterplot(data=df, x='time', y='distance', alpha=0.3, s=10)
# load dumped file:
airstart = datetime.time(11, 35)
crop_start = datetime.time(11, 30)
#crop_end = datetime.time(13, 30)
crop_end = datetime.time(11, 50)



DUMP_DIRECTORY = 'dump/task_2025-03-08'

whitelist = ['benjamin', 'roger', 'patrick']
dfs = []
names = []
pilots_in_goal = {}
for file in os.listdir(DUMP_DIRECTORY):
    in_whitelist = False
    for white in whitelist:
        if white in file:
            in_whitelist = True

    #if in_whitelist and file.lower().endswith('.csv'):
    if file.lower().endswith('.csv'):
        


        in_goal = (0 in df.index)
        if in_goal:
            seconds = df.loc[df.index == 0]['seconds'].values[0]
            print('Pilot in goal', pilot_name, seconds)
            pilots_in_goal[pilot_name] = seconds
        
        #ONLY_GOAL = False
        #if ONLY_GOAL and in_goal:
        #    names.append(pilot_name)
        #    dfs.append(df)
        

        names.append(pilot_name)
        dfs.append(df)


# process only pilots in goal:


#df_benjamin = process_pilot('dump/benjamin.csv', airstart, d_start, d_end, 'Benjamin')
#df_patrick = process_pilot('dump/patrick.csv', airstart, d_start, d_end, 'Patrick')
#df_roger = process_pilot('dump/roger.csv', airstart, d_start, d_end, 'Roger')

# 2d plot:
sns.set_theme(style="darkgrid")  # Optional for styling
plt.figure(figsize=(300, 5))
for df,name in zip (dfs, names):
    #sns.lineplot(data=df, x=df.index, y=df['gps_alt'], marker='o', label=f'{name}')
    sns.lineplot(data=df, x=df['seconds'], y=df['distance'], marker='o', label=f'{name}')

#plt.gca().invert_xaxis() # flip towards 0
plt.xlabel("Time")
plt.ylabel("Distance")
plt.title("Distance vs Value")

plt.close()
#plt.show()

# 3d Plot:
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# normalize a value
# we need x, y, gps_alt and the value is the color.
#value = 'speed'
#value = 'compensated_speed'
value = 'climb'
dfs_value = [df[value] for df in dfs]






#df_benjamin = pd.concat([df_benjamin, df_standardized['benjamin']], axis=1, join='outer')
#df_benjamin = df_benjamin.rename(columns={'benjamin':'value'})
#df_patrick = pd.concat([df_patrick, df_standardized['patrick']], axis=1, join='outer')
#df_patrick = df_patrick.rename(columns={'patrick':'value'})
#df_roger = pd.concat([df_roger, df_standardized['roger']], axis=1, join='outer')
#df_roger = df_roger.rename(columns={'roger':'value'})



#plot_pilot_3d(df_benjamin, 'Benjamin')
#plot_pilot_3d(df_patrick, 'Patrick')
#plot_pilot_3d(df_roger, 'Roger')


#plt.xticks(ticks=range(0, len(df_benjamin), 10), rotation=45)

# Finish 3d Plot
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Altitude')
ax.set_title('3D Scatter Plot Colored by Distance')
ax.view_init(elev=25, azim=-50)

#cbar = plt.colorbar(sc, ax=ax, shrink=0.6, aspect=10, pad=0.1)
#cbar.set_label('Distance (m)')

# Show the plot (Seaborn)
plt.legend()

plt.show()





