
# This takes a folder of simulated 0nubb or leptoquark events and gets the XY, XZ, and YZ plots for each event, to be used for machine learning.

# This script can be run by doing project_plots.py --pressure # --input_path /path/to/simulated/events --base_path /path/to/ML/data/ --diffusion # --type ""

#===========================================================================================================================================================

#-----IMPORTS-----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json
import random
import argparse

#----- ARGPARSE -----
parser = argparse.ArgumentParser()
parser.add_argument("--pressure", 
                    type = int, 
                    required = True,
                    choices = [1, 5, 10, 15, 25], 
                    help = "Air pressure of detector in simulated events (bar)")

parser.add_argument("--input_path", 
                    type = str, 
                    required = True, 
                    help = "Path to folder storing events")

parser.add_argument("--base_path", 
                    type = str, 
                    required = True, 
                    help = "Path to ML data folder")

parser.add_argument("--diffusion", 
                    type = float, 
                    required = True,
                    choices = [0.05, 0.25, 0.1, 5, 0.0],
                    help = "% Diffusion in events")

parser.add_argument("--type", 
                    type = str, 
                    required = True,
                    choices = ["leptoquark", "0nubb"],
                    help = "Ex: 0nubb, leptoquark")

args = parser.parse_args()

#----- CONFIG -----
pressure = args.pressure
input_path = args.input_path
base_path = args.base_path
diffusion = args.diffusion
event_type = args.type
file_identify = f"{event_type}_{pressure}_{diffusion}"

#-----GET VERTEX------
def get_vertex(file_, eid):
    part = pd.read_hdf(file_, 'MC/particles')

    x_vertex = part[(part.event_id == eid) & (part.particle_id == 1)].initial_x.iloc[0]
    y_vertex = part[(part.event_id == eid) & (part.particle_id == 1)].initial_y.iloc[0]
    z_vertex = part[(part.event_id == eid) & (part.particle_id == 1)].initial_z.iloc[0]

    return x_vertex, y_vertex, z_vertex

#-----GET Z-SHIFT-----
def get_zshift(pressure):
    density = 5.987*pressure
    M = 1000/0.9
    det_size = 1000*np.cbrt((4 * M) / (np.pi * density))/2.0

    return det_size

z_shift = get_zshift(pressure)

#-----GET TRAIN/TEST/VAL SPLIT-----
def get_split():
    r= random.random()
    if r < 0.7:
        return 'train'
    elif r < 0.9:
        return 'val'
    else:
        return 'test'

### ------ PLOTTING 3D EVENT HITS ------
def PlotEvent3D(axis, file_, title, eid, part, zshift):
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)


    hits = pd.read_hdf(file_, 'MC/hits')
    event_hits = hits[hits.event_id == eid].copy()
    event_hits["z"] = event_hits["z"]-zshift
    part = pd.read_hdf(file_, 'MC/particles')
    part = part[(part.event_id == eid) & (part.primary == 1)]
    x_vertex, y_vertex, z_vertex = get_vertex(file_, eid)
    
    # Create 3D axes
    ax = fig.add_subplot(axis, projection='3d')

    fig.set_facecolor('white')
    ax.set_facecolor('white')

    # Scatter plot in 3D
    sc = ax.scatter(event_hits.x, event_hits.y, event_hits.z, 
                    c=event_hits.energy, cmap='Spectral', s=10, label="Reco hits")
    
    ver = ax.scatter(x_vertex, y_vertex, z_vertex, s=50, color="black")

    ax.set_xlabel("X [mm]", fontsize=15, color='black')
    ax.set_ylabel("Y [mm]", fontsize=15, color='black')
    ax.set_zlabel("Z [mm]", fontsize=15, color='black')

    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.zaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.tick_params(axis='z', colors='black')

    ax.set_title(title, fontsize=15, color='black')

    ax.grid(False)

    # Add colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, pad=0.09)
    cbar.set_label("Energy", fontsize=12, color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

    # Remove background panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # extract raw data arrays
    x = event_hits.x.values
    y = event_hits.y.values
    z = event_hits.z.values
    c = event_hits.energy.values


    plt.close(fig)


    return (x, y, z, c), (x_vertex, y_vertex, z_vertex)

###------- GET DATASET FUNCTION --------

def get_data(XYZC, vertex, eid, file_identify, base_path, split):
    x, y, z, c = XYZC
    x_vertex, y_vertex, z_vertex = vertex

    #initialize paths
    path_img = os.path.join(base_path, "images", split)
    path_label = os.path.join(base_path, "labels", split)
    os.makedirs(path_img, exist_ok=True)
    os.makedirs(path_label, exist_ok=True)


    projections = [("xy", x, y, x_vertex, y_vertex),
                   ("yz", y, z, y_vertex, z_vertex),
                   ("xz", x, z, x_vertex, z_vertex)]
    
    w = h = 0.02 #bounding box size

    for dim, X, Y, vx, vy in projections:

        #plot the event
        fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
        ax.scatter(X, Y, c=c, cmap="Spectral", s=5)

        # plot the vertex as a black circle
        #ax.scatter(vx, vy, c="black", s=50, marker="o", edgecolors="white", linewidths=0.5, zorder=5)

        ax.axis("off") #don't show axis

        # axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        #normalize for yolo (0-1)
        cx = (vx - xlim[0]) / (xlim[1] - xlim[0]) # dist from left / total width
        cy = 1 - ((vy - ylim[0]) / (ylim[1] - ylim[0])) # subtract from 1 since y=0 is at top in yolo

        # check if normalized coordinates are valid
        if not (0 <= cx <= 1) or not (0 <= cy <= 1):
            print(f"Skipping event {eid} due to out-of-bounds normalized coordinates: cx={cx}, cy={cy}")
            return (None, None, None)


        print(f"label: 0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        #save label
        with open(os.path.join(path_label, f"event_{eid}_{file_identify}_{dim}.txt"), "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        #save plot
        fig.savefig(os.path.join(path_img, f"event_{eid}_{file_identify}_{dim}.png"), bbox_inches="tight", pad_inches=0)

        plt.close(fig)
    return x_vertex, y_vertex, z_vertex


#----- GET DATASET -----

#make a txt file of folders
os.makedirs(base_path, exist_ok=True)
input_files_txt = os.path.join(base_path, "input_folders.txt")
event_data = os.path.join(base_path, "event_data.json")


# create the file if it doesn't exist
if not os.path.exists(event_data):
    with open(event_data, "w", encoding="utf-8") as f:
        pass 

if not os.path.exists(input_files_txt):
    with open(input_files_txt, "w", encoding="utf-8") as f:
        pass 

with open(input_files_txt, "r") as f:
    text = f.read()

if input_path not in text:
    with open(input_files_txt, "a", encoding="utf-8") as f:
        f.write(input_path + "\n")

#keep track of completed events and files
completed_events_folder = os.path.join(base_path, "completed_events")
os.makedirs(completed_events_folder, exist_ok=True)

completed_files_log = os.path.join(base_path, "completed_files.txt")

if os.path.exists(completed_files_log):
    with open(completed_files_log, "r") as f:
        completed_files_set = set(line.strip() for line in f)
else:
    completed_files_set = set()

#only get h5 files
if os.path.isdir(input_path):
    h5_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith("h5")]
else:
    h5_files = [input_path]

#filter out already processed files
h5_files = [f for f in h5_files if os.path.basename(f) not in completed_files_set]

completed_files = 0


for file in h5_files:
    completed_log = os.path.join(completed_events_folder, os.path.basename(file) + "_completed.txt")

    hits_df = pd.read_hdf(file, "MC/hits")
    part_df = pd.read_hdf(file, "MC/particles")
    event_ids = hits_df["event_id"].unique()

    #keep track of completed events in file
    if completed_log and os.path.exists(completed_log):
        with open(completed_log, "r") as f:
            completed_events_set = set(int(line.strip()) for line in f)
    else:
        completed_events_set = set()
    
    total_events = len(event_ids)
    already_completed_events = len(completed_events_set)

    print(f"{already_completed_events}/{total_events} events completed in file")

    new_completed_events = 0

    #loop through events
    for i, eid in enumerate(event_ids):
        print(f"Processing event {eid}...")
        split = get_split()
        (XYZC, vertex) = PlotEvent3D(111, file, "", eid, part_df, z_shift)
        x, y, z = get_data(XYZC, vertex, eid, file_identify, base_path, split=split)
        if x is None:
            continue
        h5_path = os.path.join(input_path, file)
        true_data_event = {
                            "event_id": int(eid),  # convert to native int
                            "path": h5_path,
                            "type": event_type,
                            "pressure": int(pressure),
                            "diffusion": str(diffusion) + "percent",
                            "x": float(x),
                            "y": float(y),
                            "z": float(z)
                        }
        with open(event_data, "a", encoding="utf-8") as f:
            json.dump(true_data_event, f)
            f.write("\n")

        print(f"completed event {eid}")

        if completed_log:
            with open(completed_log, 'a') as f:
                f.write(f"{eid}\n")

        new_completed_events += 1
    
    print(f"completed file {file}\n{completed_files}/{len(h5_files)} completed")
    with open(completed_files_log, "a") as f:
        f.write(os.path.basename(file) + "\n")


