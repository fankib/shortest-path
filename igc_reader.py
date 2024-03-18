from os import listdir
from os.path import isfile, join
from datetime import datetime
import numpy as np
import utm


class Pilot:

    def __init__(self, name, position):
        self.name = name
        self.position = position # 3D Position at Start

class IgcReader:

    def __init__(self, folder, time):
        self.folder = folder
        self.time = datetime.strptime(time, '%H:%M').time()
        print(self.time)
        self.pilots = []
        self.read_folder()       

    def pilot_by_name(self, name):
        for p in self.pilots:
            if name in p.name:
                return p
        raise ValueError('Pilot not found: ' + name) 

    def read_folder(self):
        files = self.read_files()
        for f in files:
            pilot = self.parse_pilot(f)
            self.pilots.append(pilot)
    
    def read_files(self):
        return [f for f in listdir(self.folder) if isfile(join(self.folder, f))]
    
    def parse_pilot(self, file):
        #print(file)
        name = self.parse_pilot_name(file)
        position = self.parse_position(file)
        return Pilot(name, position)        
    
    def parse_pilot_name(self, file):
        pos_point = file.find('.')
        return file[10:pos_point]
    
    def parse_position(self, file):
        with open(join(self.folder, file), 'r') as f:
            for line in f.readlines():
                if line[0] != 'B':
                    continue
                b_time = datetime.strptime(line[1:7], '%H%M%S').time()
                if b_time < self.time:
                    continue
                # start time reached:
                #lat = int(line[7:7+7])/1e5
                lat_deg = int(line[7:7+2])
                lat_min = int(line[9:9+2])
                lat_sec = int(line[11:11+3])/10
                lat = lat_deg + (lat_min/60) + (lat_sec/3600)
                lon_deg = int(line[15:15+3])
                lon_min = int(line[18:18+2])
                lon_sec = int(line[20:20+3])/10                
                lon = lon_deg + (lon_min/60) + (lon_sec/3600)
                alt = int(line[-5:])                
                x, y, a, b = utm.from_latlon(lat, lon, 32, 'T')                
                return np.array([x, y, alt])            
        return None


# Test as execution:
if __name__ == '__main__':

    reader = IgcReader('tasks/regio_verbier/igc', '13:25') # Use UTC times
    for p in reader.pilots:
        print(f'{p.name}: {p.position}')



    