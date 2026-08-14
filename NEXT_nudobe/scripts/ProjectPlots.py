# This takes a folder of simulated 0nubb or leptoquark events and gets the XY, XZ, and YZ plots for each event, to be used for machine learning.

# This script can be run by doing project_plots.py --pressure # --input_path /path/to/simulated/events --base_path /path/to/ML/data/ --diffusion # --type ""

#===========================================================================================================================================================

#-----IMPORTS-----
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import copy
import itertools
import matplotlib.pyplot as plt
from TrackReconstruction_functions import *
import sys
import pickle
import os
import re
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
infile = args.input_file
pressure = args.pressure
diffusion = args.diffusion
event_type = args.type
file_identify = f"{event_type}_{pressure}_{diffusion}"

print("Pressure:", pressure, "bar")
print("diffusion:",diffusion)
cluster = 1
tr_opt = 0
plot = 0

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

def get_data(temp_df, hits_event, Tracks, vertex, eid, file_identify, colors, split):
    x_vertex, y_vertex, z_vertex = vertex

    projections = [("xy", "x", "y", x_vertex, y_vertex),
               ("yz", "y", "z", y_vertex, z_vertex),
               ("xz", "x", "z", x_vertex, z_vertex)]
    
    w = h = 0.02 #bounding box size
    axis_limits = {} 

    for dim, X, Y, vx, vy in projections:
        #initialize paths
        img_filename = f"event_{eid}_{file_identify}_{dim}_{split}.png"
        label_filename = f"event_{eid}_{file_identify}_{dim}_{split}.txt"

        #plot the event
        fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)

        plot_tracks(ax, temp_df[X], temp_df[Y], Tracks)
        ax.scatter(hits_event[X], hits_event[Y], c=colors, marker='o', alpha=0.15, s=3)
        
        #ax.scatter(vx, vy, color="black", s=10)

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
            return (None, None, None, None)


        print(f"label: 0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        #save label
        with open(label_filename, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        #save plot
        fig.savefig(img_filename, bbox_inches="tight", pad_inches=0)

        plt.close(fig)

    return x_vertex, y_vertex, z_vertex, axis_limits


Track_dict = {}
df_list = []
df_meta = []
hits = pd.read_hdf(infile, "MC/hits")
print("Total events to process:", len(hits.event_id.unique()))

for index, event_num in enumerate(hits.event_id.unique()):
    print("On index, Event:", index, event_num)

    hit = hits[hits.event_id == event_num]

    # These different function calls allow for different sorting if there is soeme kind of failure
    df, Tracks, connected_nodes, connection_count, pass_flag, contained = RunTracking(hit, cluster, pressure, diffusion, 0)
    if (not pass_flag):
        print("Error in track reco, try resorting hits\n")
        df, Tracks, connected_nodes, connection_count, pass_flag, contained = RunTracking(hit, cluster, pressure, diffusion, 1)
    if (not pass_flag):
        print("Error in track reco, try resorting hits\n")
        df, Tracks, connected_nodes, connection_count, pass_flag, contained = RunTracking(hit, cluster, pressure, diffusion, 2)

    if (not pass_flag):
        print("Track still failed, skipping,...")
        continue

    Track_dict[event_num] = Tracks
    df_list.append(df)
    
    # Slightly different input params for next1t analysis
    if (diffusion == "next1t"):
        temp_meta = GetTrackdf(df, Tracks, 30, 15, 15, pressure)
    else:

        # Allow scan of various parameters for the track reconstruction
        if (tr_opt == 0):
            temp_meta = GetTrackdf(df, Tracks, 400/pressure, 100/pressure, 200/pressure, pressure) # scale these params inversely with the pressure
        elif (tr_opt == 1):
            temp_meta = GetTrackdf(df, Tracks, 100/pressure, 100/pressure, 100/pressure, pressure)
        elif (tr_opt == 2):
            temp_meta = GetTrackdf(df, Tracks, 200/pressure, 200/pressure, 200/pressure, pressure)
        elif (tr_opt == 3)
            temp_meta = GetTrackdf(df, Tracks, 300/pressure, 300/pressure, 300/pressure, pressure)
        elif (tr_opt == 4):
            temp_meta = GetTrackdf(df, Tracks, 400/pressure, 400/pressure, 400/pressure, pressure)
        elif (tr_opt == 5):
            temp_meta = GetTrackdf(df, Tracks, 500/pressure, 500/pressure, 500/pressure, pressure)
        elif (tr_opt == 6):
            temp_meta = GetTrackdf(df, Tracks, 600/pressure, 600/pressure, 600/pressure, pressure)
    
    
    # temp_meta = UpdateTrackMeta(temp_meta, df, 10/pressure) # Merge deltas and brems that are near the blobs in the metadata
    temp_meta = UpdateTrackMeta2(temp_meta) # ensure variables are organized so that var 1 > var 2 e.g blob1>blob2
    temp_meta["contained"] = contained
    df_meta.append(temp_meta)

    print("Printing Metadata\n", temp_meta[["event_id", "primary", "length", "energy", "blob1", "blob2", "blob1R", "blob2R", "Tortuosity1", "Tortuosity2", "Squiglicity1", "Squiglicity2", "label", "contained"]])
    print(temp_meta[["event_id", "blob1RTD", "blob2RTD"]])
    print("\n\n")


df = pd.concat(df_list)
df_meta = pd.concat(df_meta)

# Print the reconstruction efficiency and any events that failed
Reco_eff = 100*len(df_meta.event_id.unique())/ len(hits.event_id.unique())
print("Track reconstruction efficiency:", Reco_eff)

all_limits = []
part_df = pd.read_hdf(infile, "MC/particles")
print("Plotting Events")
for index, evt in enumerate(df.event_id.unique()):

    print("On index, Event:", index, evt)
    split = get_split()
    temp_df = df[df.event_id == evt]
    # temp_df = temp_df.sort_values(by='id')
    temp_df.index = temp_df.id

    hits_event = hits[hits.event_id == evt].copy()
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(hits_event.energy.min(), hits_event.energy.max())
    colors = cmap(norm(hits_event.energy))
    
    vertex = get_vertex(part_df, evt)

    temp_df["z"] = temp_df["z"] - z_shift
    hits_event["z"] = hits_event["z"] - z_shift

    Tracks = Track_dict[evt]

    vx, vy, vz, axis_limits = get_data(temp_df, hits_event, Tracks, vertex, evt, file_identify, colors, split)
    if vx is None:
        continue
    all_limits.append({"event_id": evt, "axis_limits": json.dumps(axis_limits)})
    print(f"completed event {evt}")


limits_df = pd.DataFrame(all_limits)
limits_df.to_json(f"{file_identify}_limits.jsonl", orient="records", lines=True)

