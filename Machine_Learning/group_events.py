#This takes the predicted vertex from a json file and matches that information with the actual event data

#json input format: {"event_name": {"x": #, "y": #, "z": #}}

#json output format: {"index": {"event_id": "", "path": "", "type": "", "pressure": "", "diffusion": "",
#  "x": "", "y": "", "z": ""}}

#===================================================================================================================

#imports

import json
import os
import h5py
import sys
import re
import numpy as np
import pandas as pd

#config

json_file = sys.argv[2]
sim_data_base = sys.argv[1]

grouped_events = {}

#get the percent diffusion from file name

def get_percent(folder):
    match = re.search(r'\d+(\.\d+)?', folder)
    return float(match.group()) if match else None

#sort through the original simulation data and create a dictionary with all the info
grouped_events = {}
idx = 0 

for event_type in os.listdir(sim_data_base):
    type_path = os.path.join(sim_data_base, event_type)

    if os.path.isdir(type_path) and event_type in ["0nubb", "leptoquark"]:

        for pressure in os.listdir(type_path):
            pressure_path = os.path.join(type_path, pressure)

            if os.path.isdir(pressure_path) and "bar" in pressure_path:

                for smear in os.listdir(pressure_path):
                    smear_path = os.path.join(pressure_path, smear)
                    percent = get_percent(smear)

                    for file in os.listdir(smear_path):

                        if file.endswith(".h5"):

                            file_path = os.path.join(smear_path, file)
                            hits_df = pd.read_hdf(file_path, "MC/hits")

                            #Loop through each event in this file
                            for event_id, group in hits_df.groupby("event_id"):


                                grouped_events[idx] = {
                                    "event_id": event_id,
                                    "path": file_path,
                                    "type": event_type,
                                    "pressure": pressure,
                                    "diffusion": percent,
                                }
                                idx += 1 

#event_#_type_pressure_percent

#match the event hits with the event vertex

with open (json_file, "r") as f:
    vertex_data = json.load(f)

    for index in grouped_events:
        event_info = grouped_events[index]
        key = f"event_{event_info['event_id']}_{event_info['type']}_{event_info['pressure']}_{event_info['diffusion']}"
        
        if key in vertex_data:
            vertex = vertex_data[key]
            grouped_events[index]["x"] = float(vertex["x"])
            grouped_events[index]["y"] = float(vertex["y"])
            grouped_events[index]["z"] = float(vertex["z"])

with open ("events_with_vertex.json", "w") as f:
    json.dump(grouped_events, f, indent=2)










