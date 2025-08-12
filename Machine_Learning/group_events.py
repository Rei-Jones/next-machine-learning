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
    if os.path.isdir(type_path) and event_type == "0nubb" or event_type == "leptoquark": #only accept those 2 event types
        for pressure in os.listdir(type_path):
            pressure_path = os.path.join(type_path, pressure)
            if os.path.isdir(pressure_path) and "bar" in pressure_path:
                for smear in os.listdir(pressure_path):
                    smear_path = os.path.join(pressure_path, smear)
                    percent = get_percent(smear)
                    for file in os.listdir(smear_path):
                        if ".h5" in file:
                            file_path = os.path.join(smear_path, file)
                            with h5py.File(file_path, "r") as f:
                                hits_table = f["MC/hits/table"]
                                for event in hits_table["index"]:
                                    hits_raw = hits_table["values_block_0"][event][:10]
                                    hits_xyz = np.array([hit[:3] for hit in hits_raw])
                                    grouped_events[str(event)] = {
                                        "path": file_path,
                                        "type": event_type,
                                        "pressure": pressure,
                                        "diffusion": percent,
                                        "hits": hits_xyz.to_list()
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
                    grouped_events[key][dim] = [loc]

with open ("events_with_vertex.json", "w") as f:
    json.dump(grouped_events, f, indent=2)










