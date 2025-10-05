import numpy as np
import os
import re
import pandas as pd
import json
import tables

sim_data_base = "/home/rei/NEXT/vertex_ML/diffused_data/"
events_with_vertex = {}
event_count = 0
max_events = 1000

def get_float(folder):
    match = re.search(r'\d+(\.\d+)?', folder)
    return float(match.group()) if match else None

for event_type in os.listdir(sim_data_base):
    type_path = os.path.join(sim_data_base, event_type)
    if os.path.isdir(type_path) and event_type in ["0nubb"]: #only accept those 2 event types
        for pressure in os.listdir(type_path):
            pressure_path = os.path.join(type_path, pressure)
            if os.path.isdir(pressure_path) and "bar" in pressure_path:
                for smear in os.listdir(pressure_path):
                    smear_path = os.path.join(pressure_path, smear)
                    percent = get_float(smear)
                    for file in os.listdir(smear_path):
                        if file.endswith(".h5"):
                            file_path = os.path.join(smear_path, file)

                            # Load both tables once
                            try:
                                hits_df = pd.read_hdf(file_path, "MC/hits")
                            except (OSError, tables.exceptions.HDF5ExtError, ValueError) as e:
                                print(f"Skipping corrupted file: {file_path}, error: {e}")
                                continue
                            part_df = pd.read_hdf(file_path, "MC/particles")

                            # Group particles by event for vertex lookup
                            part_primary = part_df[part_df["primary"] == 1]

                            for event_id, group in hits_df.groupby("event_id"):
                                if event_count >= max_events:
                                    break
                                print(f"processing event {event_id}")

                                # Get vertex from primary particle for this event
                                primary_row = part_primary[part_primary["event_id"] == event_id]
                                if not primary_row.empty:
                                    x, y, z = primary_row.iloc[0][["initial_x", "initial_y", "initial_z"]]
                                else:
                                    x, y, z = None, None, None

                                # Convert vertex coordinates to float as well
                                x = float(x) if x is not None else None
                                y = float(y) if y is not None else None
                                z = float(z) if z is not None else None

                                pressure_float = get_float(pressure)
                                if type(percent) == str:
                                    diffusion = get_float(percent)
                                else: 
                                    diffusion = percent
                                diffusion = str(diffusion) + "percent"

                                events_with_vertex = {
                                    "event_id": event_id,
                                    "path": file_path,
                                    "type": event_type,
                                    "pressure": pressure_float,
                                    "diffusion": diffusion,
                                    "x": x,
                                    "y": y,
                                    "z": z
                                }
                                event_count +=1
                            if event_count >= max_events:
                                break
                    if event_count >= max_events:
                        break
            if event_count >= max_events:
                break
    if event_count >= max_events:
        break


output_json = "/home/rei/NEXT/vertex_ML/events_with_true_vertex_0nubb.json"

with open(output_json, "w") as f:
    json.dump(events_with_vertex, f, indent=2)

if os.path.exists(output_json):
    print(f"Successfully saved to {output_json}")
else:
    print(f"Script failed.")