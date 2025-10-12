#This takes the predicted vertex from a json file and matches that information with the actual event data

#json input format: {"event_key": {"x": #, "y": #, "z": #}}

#json output format: {"index": {"event_id": "", "path": "", "type": "", "pressure": "", "diffusion": "",
#  "true_vertex": "", "predicted_vertex": ""}}

#===================================================================================================================

#imports

import json
import os
import re
import pandas as pd
import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

#config

json_file = "/home/rei/NEXT/vertex_ML/vertex_txt/train_run_3_predictions.json"
sim_data_base = "/home/rei/NEXT/vertex_ML/diffused_data"
output_path = "/home/rei/NEXT/vertex_ML/vertex_txt/train_run_3_data.json"

#get the number from folder name

def get_number(folder):
    match = re.search(r'\d+(\.\d+)?', folder)
    return float(match.group()) if match else None

#load YOLO vertex predictions

with open(json_file, "r") as f:
    vertex_data = json.load(f)

vertex_keys = set(vertex_data.keys())
# loop through sim data

def process_h5(file_path):
    events_list = []
    try:
        part_df = pd.read_hdf(file_path, "MC/particles")
    except (OSError, KeyError, ValueError) as e:
        print(f"Skipping invalid event file: {file_path} ({e})")
        return events_list
    
    part_df = part_df[part_df.particle_id == 1]

    path_parts = file_path.split(os.sep)
    event_type = next((p for p in path_parts if p in ["0nubb", "leptoquark"]), None)
    pressure = next((p for p in path_parts if "bar" in p), None)
    smear = path_parts[-2]

    if not event_type or not pressure or not smear:
        return events_list
    
    pressure_int = get_number(pressure)
    diffusion = get_number(smear)

    for _, row in part_df.iterrows():
        event_key = f"event_{row.event_id}_{event_type}_{pressure_int}_{diffusion}"
        if event_key in vertex_keys:
            pred = vertex_data[event_key]
            events_list.append({
                "event_id": row.event_id,
                "path": file_path,
                "type": event_type,
                "pressure": pressure_int,
                "diffusion": diffusion,
                "true_vertex": [row.initial_x, row.initial_y, row.initial_z],
                "predicted_vertex": [pred["x"], pred["y"], pred["z"]]
            })
    return events_list
    
if __name__ == "__main__":
    h5_files = glob.glob(f"{sim_data_base}/**/*.h5", recursive=True)
    print(f"found {len(h5_files)} .h5 files")

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_h5, h5_files),
                            total=len(h5_files),
                            desc="processing files",
                            dynamic_ncols=True))

    grouped_events = [event for sublist in results for event in sublist]

    with open(output_path, "w") as f:
        json.dump(grouped_events, f, indent=2)

    print(f"completed {len(grouped_events)} events, saved to {output_path}.")
