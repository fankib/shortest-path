import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# globals
YEARS = ['2021', '2022', '2023', '2024']

# define some functions
def read_pilot(name):
    results = []
    for year in YEARS:
        results.append(read_pilot_year(name, year))
    return results

def read_pilot_year(name, year):
    filename = f'{name}_{year}.csv'
    df = read_as_pandas(filename)
    df = clean(df)
    df = filter_distance(df, 100.)
    df = filter_top_k(df, 3)
    return df

def read_as_pandas(filename):
    return pd.read_csv(f'xcontest/{filename}')

def clean(df):
    # convert Startplatz to country
    df['Country'] = df['Startplatz'].apply(lambda s: s[:2])
    df['Takeoff'] = df['Startplatz'].apply(lambda s: s[2:])
    
    # convert km into float
    df['Distance'] = df['Länge'].apply(lambda s: float(s[:-3]))

    # convert speed into float
    df['Speed'] = df['km/h']

    # rename and drop columns
    df = df[['Country', 'Takeoff', 'Distance', 'Speed']]

    return df

def filter_distance(df, minimal_distance):
    return df[df['Distance'] >= minimal_distance]

def filter_top_k(df, k):
    values = list(sorted(df['Speed'].values, reverse=True))
    if len(values) <= k:
        return df
    threshold = values[k-1]
    return df[df['Speed'] >= threshold]

def average_speed(results):
    return [r['Speed'].mean() for r in results]

def plot_pilot(name, values, color=None):
    plt.plot(YEARS, values, c=color, label=name) 

if __name__ == '__main__':

    #Forward test case:
    # bembem 2024:
    #result = read_pilot_year('bembem', '2024')
    #print(result)

    # Test bembem
    results = read_pilot('oliverk')
    results = [filter_top_k(r, 2) for r in results]
    for r in results:
        print(r)
        print('~~~')
    #exit() 

    # Plot Pilots:
    plt.figure()
    for pilot in ['bembem', 'oliverk']:
        results = read_pilot(pilot)
        values = average_speed(results)
        plot_pilot(pilot, values)
        print(f'~~~ {pilot} ~~~')
        for r in results:
            print(r)
            print('~~~')
    plt.legend()
    plt.show()

















