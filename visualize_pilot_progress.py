import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot_pilot(df, name):
    # filter by airstart
    df = df[df['airstart'] == True]

    # Reduce the points
    reduction = 50 # aggregate this many rows
    #df = df[df.index % reduction == 0]
    df['gps_alt'] = df['gps_alt']/1700*30000
    df = df.groupby(df.index // reduction).agg({'time': 'first', 'distance': 'mean', 'gps_alt': 'mean'})
    
    df['time'] = np.arange(len(df))
    df = df[df['time'] < 70.]

    sns.lineplot(data=df, x='time', y='distance', marker='o', label=name)
    sns.lineplot(data=df, x='time', y='gps_alt', label=f'{name} GPS Alt')

# Plot using Seaborn
sns.set_theme(style="darkgrid")  # Optional for styling
plt.figure(figsize=(8, 5))

#sns.scatterplot(data=df, x='time', y='distance', alpha=0.3, s=10)
# load dumped file:
df_benjamin = pd.read_csv('dump/benjamin.csv')
plot_pilot(df_benjamin, 'Benjamin')
df_roger = pd.read_csv('dump/roger.csv')
plot_pilot(df_roger, 'Roger')
df_patrick = pd.read_csv('dump/patrick.csv')
plot_pilot(df_patrick, 'Patrick')

#plt.xticks(ticks=range(0, len(df_benjamin), 10), rotation=45)

# Show the plot
plt.legend()
plt.xlabel("Time")
plt.ylabel("Distance")
plt.title("Time vs Distance")
plt.show()





