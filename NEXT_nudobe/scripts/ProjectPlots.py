
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
parser.add_argument("--input_file", 
                    type = str, 
                    required = True, 
                    help = "Path to h5 file")

parser.add_argument("--pressure", 
                    type = int, 
                    required = True,
                    choices = [1, 5, 10, 15, 25], 
                    help = "Air pressure of detector in simulated events (bar)")

parser.add_argument("--diffusion", 
                    type = str, 
                    required = True,
                    help = "% Diffusion in events")

parser.add_argument("--type", 
                    type = str, 
                    required = True,
                    help = "Ex: 0nubb, leptoquark")

args = parser.parse_args()

#----- CONFIG -----
input_file = args.input_file
pressure = args.pressure
diffusion = args.diffusion
event_type = args.type
file_identify = f"{event_type}_{pressure}_{diffusion}"

#-----GET VERTEX------
def get_vertex(part_df, eid):
    row = part_df[
        (part_df.event_id == eid) & (part_df.particle_id == 1)
    ].iloc[0]
    return row.initial_x, row.initial_y, row.initial_z


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
def PlotEvent3D(hits, part, eid, z_shift):
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)


    event_hits = hits[hits.event_id == eid].copy()
    event_hits["z"] = event_hits["z"]-z_shift
    part = part[(part.event_id == eid) & (part.primary == 1)]
    x_vertex, y_vertex, z_vertex = get_vertex(part, eid)
    
    # Create 3D axes
    ax = fig.add_subplot(111, projection='3d')

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
def get_data(XYZC, vertex, eid, file_identify, split):
    x, y, z, c = XYZC
    x_vertex, y_vertex, z_vertex = vertex

    projections = [("xy", x, y, x_vertex, y_vertex),
                   ("yz", y, z, y_vertex, z_vertex),
                   ("xz", x, z, x_vertex, z_vertex)]
    
    w = h = 0.02 #bounding box size
    axis_limits = {} 

    for dim, X, Y, vx, vy in projections:
        #initialize paths
        img_filename = f"event_{eid}_{file_identify}_{dim}_{split}.png"
        label_filename = f"event_{eid}_{file_identify}_{dim}_{split}.txt"

        #plot the event
        fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
        ax.scatter(X, Y, c=c, cmap="Spectral", s=5)
        #ax.scatter(vx, vy, color="black", s=10)

        # plot the vertex as a black circle
        #ax.scatter(vx, vy, c="black", s=50, marker="o", edgecolors="white", linewidths=0.5, zorder=5)

        ax.axis("off") #don't show axis

        # axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Expand limits if vertex is outside current limits
        if vx < xlim[0]:
            xlim = (vx, xlim[1])
        if vx > xlim[1]:
            xlim = (xlim[0], vx)
        if vy < ylim[0]:
            ylim = (vy, ylim[1])
        if vy > ylim[1]:
            ylim = (ylim[0], vy)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        axis_limits[dim] = {"xlim": list(xlim), "ylim": list(ylim)}

        print(f"Event {eid}: xlim={xlim}, ylim={ylim}, vertex=({vx}, {vy})")



        #normalize for yolo (0-1)
        cx = (vx - xlim[0]) / (xlim[1] - xlim[0]) # dist from left / total width
        cy = 1 - ((vy - ylim[0]) / (ylim[1] - ylim[0])) # subtract from 1 since y=0 is at top in yolo

        # check if normalized coordinates are valid
        if not (0 <= cx <= 1) or not (0 <= cy <= 1):
            print(f"Skipping event {eid} due to out-of-bounds normalized coordinates: cx={cx}, cy={cy}")
            return (None, None, None)


        print(f"label: 0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        #save label
        with open(label_filename, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        #save plot
        fig.savefig(img_filename, bbox_inches="tight", pad_inches=0)

        plt.close(fig)

    return x_vertex, y_vertex, z_vertex, axis_limits


#----- GET DATASET -----

#loop through events
all_limits = []
part_df = pd.read_hdf(input_file, "MC/particles")
hits_df = pd.read_hdf(input_file, "MC/hits")
for eid, data in part_df.groupby("event_id"):
    plt.close('all')
    print(f"Processing event {eid}...")
    split = get_split()
    (XYZC, vertex) = PlotEvent3D(hits_df, part_df, eid, z_shift)
    x, y, z, axis_limits = get_data(XYZC, vertex, eid, file_identify, split=split)
    if x is None:
        continue
    all_limits.append({"event_id": eid, "axis_limits": json.dumps(axis_limits)})
    print(f"completed event {eid}")


limits_df = pd.DataFrame(all_limits)
limits_df.to_json(f"{file_identify}_limits.jsonl", orient="records", lines=True)
