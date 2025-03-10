import os
import re


'''
If the pilot name is missing it puts a the parsed name into the file.
Does not check or recreate Signatures!
'''

ROOT_DIR = 'igc/swissleague/march'
TASK_NAME = 'igc6494_2025-03-08'

task_dir = os.path.join(ROOT_DIR, TASK_NAME)
output_dir = os.path.join('dump/', TASK_NAME)
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(task_dir):
    if filename.lower().endswith('.igc'):
        
        # Parse pilot name
        match = re.search(r"LiveTrack (.+?)\.\d", filename)
        pilot_name = match.group(1)

        # get lines:
        lines = []
        with open(os.path.join(task_dir, filename), 'r') as file:
            lines = file.readlines()

        # check and fill:
        updated_lines = []
        for line in lines:
            if line.startswith('HFPLTPILOT:'):
                if line.strip() == 'HFPLTPILOT:':
                    line = f'HFPLTPILOT:{pilot_name}\n'
                    print(f'Updated Record for {pilot_name}')
            updated_lines.append(line)
        
        # write lines
        with open(os.path.join(output_dir, filename), 'w') as file:
            file.writelines(updated_lines)

        
        



