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

# UTM
UTM_ZONE = 32
UTM_LETTER = 'T'

def as_seconds(t):
    return t.hour*3600 + t.minute*60 + t.second

def height_compensation(delta_altitude, delta_distance, delta_time):
    if delta_altitude < 0:
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

def height_compensation_speed(delta_altitude, delta_distance, delta_time):
    _, speed = height_compensation(delta_altitude, delta_distance, delta_time)
    return speed

def process_pilot(filename, airstart, start, end, name):
    df = pd.read_csv(filename)
    # convert time
    df['time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').time())
    df['seconds'] = df['time'].apply(lambda t: as_seconds(t) - as_seconds(airstart))
    
    # filter by airstart
    df = df[df['airstart'] == True]

    # check if df contains end:
    if len(df) == 0 or df['distance'].min() > end:
        return None

    #df = crop_time(df, start, end)
    df = crop_distance(df, start, end)

    # convert x,y coordinates:
    xs, ys, _, _ = utm.from_latlon(df['lat'].values, df['lon'].values)
    df['x'] = xs
    df['y'] = ys

    # Resample in 100m of distance    
    df['distance_bin'] = (df['distance'] // 50) * 50 # Round every 10m
    df = df.groupby('distance_bin').first().reset_index()

    # set index distance
    df.set_index('distance_bin', inplace=True)
    df.sort_index(ascending=False, inplace=True)

    # Compute statistics    
    first_row = df.iloc[0]    
    last_row = df.iloc[-1]
    delta_time = last_row['seconds'] - first_row['seconds']
    delta_gps_alt = last_row['gps_alt'] - first_row['gps_alt']
    delta_distance = first_row['distance'] - last_row['distance']

    climb = delta_gps_alt / delta_time
    speed = delta_distance / delta_time

    extra_time, compensated_speed = height_compensation(delta_gps_alt, delta_distance, delta_time)


    print(name, 'time:', delta_time, 'altitude', delta_gps_alt, 'distance', delta_distance, 'climb', climb, 'speed', speed*3.6, 'extra_time', extra_time, 'compensated_speed', compensated_speed*3.6)


    # compute diffs
    #periods=20 # 1=10m
    #df['delta_seconds'] = df['seconds'].diff(periods=periods)    
    #df['delta_gps_alt'] = df['gps_alt'].diff(periods=periods)    
    #df['delta_distance'] = df['distance'].diff(periods=periods).apply(lambda x: -x)

    # use shifts:
    shift = 20//2 # period = 2*shift
    df['delta_seconds'] = df['seconds'].shift(-shift) - df['seconds'].shift(shift)
    df['delta_gps_alt'] = df['gps_alt'].shift(-shift) - df['gps_alt'].shift(shift)
    df['delta_distance'] = df['distance'].shift(-shift) - df['distance'].shift(shift)

    
    df['climb'] = df['delta_gps_alt'] / df['delta_seconds']
    df['speed'] = (df['delta_distance'] / df['delta_seconds'])    
    df['compensated_speed'] = df.apply(lambda row: height_compensation_speed(row['delta_gps_alt'], row['delta_distance'], row['delta_seconds']), axis=1)
    
    #print(df)

    #df[df['speed'] > 25.] = 25.

    #plot_pilot(df, name)
    #plot_pilot_3d(df, name)

    return df

def crop_time(df, start, end):
    df = df[df['time'] >= start]
    df = df[df['time'] < end]
    return df

def crop_distance(df, start, end):
    df = df[df['distance'] <= start]
    df = df[df['distance'] > end]
    return df



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
    colors = plt.cm.viridis(norm(c))
    
    
    alphas = 0.01+0.9*norm(c)
    alphas = 1 - alphas # the bad!
    #alphas = np.clip(np.abs(norm(c)-0.5)*2, 0., 1.)
 
    alphas = np.clip(np.nan_to_num(alphas, nan=0.5), 0., 1.) # how to handle nans? interpolate?
    colors[:, -1] = alphas
    sizes = alphas*50
    ax.scatter(df['x'], df['y'], df['gps_alt'], c=colors, marker='o', s=sizes)

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

d_start = 76000
d_end = 5000

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
        pilot_name = file.split('.')[0]
        
        df = process_pilot(f'{DUMP_DIRECTORY}/{file}', airstart, d_start, d_end, pilot_name)
        if df is None:
            print(f'Pilot {pilot_name} did not reach end. Skip!')
            continue

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
df_all = pd.concat(dfs_value, axis=1, join='outer') # use outer?
# Replace NaN values in each row with the row mean
#df_all = df_all.apply(lambda row: row.fillna(row.mean()), axis=1) # required? i dont think so!
df_all.columns = names
# normalize:
#df_all = df_all.sub(df_all.min(axis=1), axis=0).div(df_all.max(axis=1) - df_all.min(axis=1), axis=0)
# standardize:
#df_all = df_all.sub(df_all.mean(axis=1), axis=0).div(df_all.std(axis=1), axis=0)


# join back:
for df, name in zip(dfs, names):
    col = df_all[name].interpolate()
    df = pd.concat([df, col], axis=1, join='inner')
    df = df.rename(columns={name:'value'})
    plot_pilot_3d(df, name)

kml.save('dump/earth.kml')

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





