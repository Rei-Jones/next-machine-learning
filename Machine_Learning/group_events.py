#This takes the predicted vertex from a json file and matches that information with the actual event data

#json input format: {"event_name": {"x": #, "y": #, "z": #}}

#json output format: {"index": {"path": "", "type": "", "pressure": "", "diffusion": "", 
# "hits": [[x0, y0, z0],...] "x": "", "y": "", "z": ""}}

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
                                hits_xyz = group[["x", "y", "z"]].to_numpy()

                                grouped_events[str(event_id)] = {
                                    "path": file_path,
                                    "type": event_type,
                                    "pressure": pressure,
                                    "diffusion": percent,
                                    "hits": hits_xyz.tolist()
                                }

#event_#_type_pressure_percent

#match the event hits with the event vertex

with open (json_file, "r") as f:
    vertex_data = json.load(f)
    for event, info in vertex_data.items():
        event_from_vertex = event.split("_")[1]
        type_from_vertex = event.split("_")[2]
        pressure_from_vertex = event.split("_")[3]
        percent_from_vertex = float(event.split("_")[-1].split(".")[0])
        for key, val in grouped_events.items():
            type_from_grouped = val["type"]
            pressure_from_grouped = val["pressure"]
            percent_from_grouped = val["diffusion"]

            if (key == event_from_vertex 
                and type_from_vertex == type_from_grouped 
                and pressure_from_vertex == pressure_from_grouped 
                and percent_from_vertex == percent_from_grouped):
                for dim, loc in info.items():
                    grouped_events[key][dim] = [float(loc)]

with open ("events_with_vertex.json", "w") as f:
    json.dump(grouped_events, f, indent=2)










