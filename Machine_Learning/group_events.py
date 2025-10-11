#This takes the predicted vertex from a json file and matches that information with the actual event data

#json input format: {"event_key": {"x": #, "y": #, "z": #}}

#json output format: {"index": {"event_id": "", "path": "", "type": "", "pressure": "", "diffusion": "",
#  "true_vertex": "", "predicted_vertex": ""}}

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

json_file = "/home/rei/NEXT/vertex_ML/vertex_txt/train_run_3_predictions.json"
sim_data_base = "/home/rei/NEXT/vertex_ML/diffused_data"

grouped_events = {}

#-----GET VERTEX------
def get_vertex(file_, eid):
    part = pd.read_hdf(file_, 'MC/particles')

    x_vertex = part[(part.event_id == eid) & (part.particle_id == 1)].initial_x.iloc[0]
    y_vertex = part[(part.event_id == eid) & (part.particle_id == 1)].initial_y.iloc[0]
    z_vertex = part[(part.event_id == eid) & (part.particle_id == 1)].initial_z.iloc[0]

    return x_vertex, y_vertex, z_vertex

#get the number from folder name

def get_number(folder):
    match = re.search(r'\d+(\.\d+)?', folder)
    return float(match.group()) if match else None

#load YOLO vertex predictions

with open(json_file, "r") as f:
    vertex_data = json.load(f)

# loop through sim data

grouped_events = []
for event_type in os.listdir(sim_data_base):
    type_path = os.path.join(sim_data_base, event_type)

    if os.path.isdir(type_path) and event_type in ["0nubb", "leptoquark"]:
        print(f"processing {event_type}")
        for pressure in os.listdir(type_path):
            pressure_path = os.path.join(type_path, pressure)

            if os.path.isdir(pressure_path) and "bar" in pressure_path:
                print(f"processing {pressure}")
                pressure_int = get_number(pressure)

                for smear in os.listdir(pressure_path):
                    smear_path = os.path.join(pressure_path, smear)
                    diffusion = get_number(smear)
                    print(f"processing {smear}")

                    for file in os.listdir(smear_path):

                        if file.endswith(".h5"):

                            file_path = os.path.join(smear_path, file)
                            try:
                                part_df = pd.read_hdf(file_path, "MC/particles")
                            except (OSError, KeyError, ValueError) as e:
                                print(f"Skipping invalid or unreadable file: {file_path}")
                                print(f"   → Reason: {e}")
                                continue

                            for event_id in part_df["event_id"].unique():
                                xt, yt, zt = get_vertex(file_path, event_id)
                                event_key = f"event_{event_id}_{event_type}_{pressure_int}_{diffusion}"
                                grouped_events.append({"event_id": event_id,
                                              "path": file_path,
                                              "type": event_type,
                                              "pressure": pressure_int,
                                              "diffusion": diffusion,
                                              "true_vertex": [xt, yt, zt],
                                              "event_key": event_key})

#match the event keys to get the redicted vertex, append to dict
for event in grouped_events:
    key = event["event_key"]
    if key in vertex_data:
        pred = vertex_data[key]
        event["predicted_vertex"] = [pred["x"], pred["y"], pred["z"]]
        event.pop("event_key")
        print(f"found matching key: {pred}")

#save file

output_path = "/home/rei/NEXT/vertex_ML/vertex_txt"
file_name = "train_run_3_data.json"
output_path = os.path.join(output_path, file_name)

with open (output_path, "w") as f:
    json.dump(grouped_events, f, indent=2)

print(f"completed {len(grouped_events)} events, saved to {output_path}")









