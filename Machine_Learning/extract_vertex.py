
#This script takes the vertex info from yolov5 and stores it in a json file,
# after running like so:

#yolo task=detect mode=predict model=your_model.pt source=your_data/images --save-txt

#YOLOv5 filename format: event_#_model_pressure_%smear_projection.txt
    #ex: event_100_0nubb_1_0.1_xy.txt
#YOLOv5 format: class x_center _y_center width height [confidence]

#json format: {"event_classification": {"x:" #, "y": #, "z": #},...}

#=====================================================================================

#imports

import json
import os
from collections import defaultdict
import numpy as np

#config

label_dir = "/home/rei/NEXT/vertex_ML/vertex_txt/train_run_3"
image_dims = (512, 512)

vertex_predictions = defaultdict(lambda: defaultdict(list))

#vertex_predictions["event"]["x"/"y"/"z"] = [value]

for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"): #exclude any non .txt files
        continue
    event_name = label_file.replace(".txt", "")
    base_event = event_name[:-3] #trim _xy etc...
    #get the projection, ex: xy, yz..
    dim_1 = label_file[-6] 
    dim_2 = label_file[-5]
    total_dim = dim_1 + dim_2

    #get events with best confidence
    with open(os.path.join(label_dir, label_file)) as f:
        lines = f.readlines()

    best_line = None
    best_conf = -1

    for line in lines:
        parts = line.strip().split()
        if parts[0] != "0":
            continue
        #handle both files with confidence and without
        if len(parts) == 6:
            confidence = float(parts[-1])
        else:
            confidence = 1.0

        if confidence > best_conf:
            best_conf = confidence
            best_line = parts

        if best_line:
            #format : class x_center y_center box_width box_height
            x_rel, y_rel = float(best_line[1]), float(best_line[2])
            x_px, y_px = x_rel*image_dims[0], y_rel*image_dims[1]

            vertex_predictions[base_event][dim_1].append(x_px)
            vertex_predictions[base_event][dim_2].append(y_px)

            print("completed event")



vertex_predictions_mean = {}

for event, coords in vertex_predictions.items():
    if all(k in coords for k in ["x", "y", "z"]): #Get events with all 3 coordinates
        #Get the mean of the predicted vertex for each event
        vertex_predictions_mean[event] = {
            k: float(np.mean(v))
            for k, v in coords.items()
        }

# Save to JSON

file_name = "train_run_3_predictions.txt"
output_path = "/home/rei/NEXT/vertex_ML/vertex_txt"
output_path = os.path.join(output_path, file_name)

with open(output_path, "w") as f:
    json.dump(vertex_predictions_mean, f, indent=2)

print(f"Saved {len(vertex_predictions)} predictions to {output_path}.")