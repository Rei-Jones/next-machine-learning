
# This takes a folder of simulated 0nubb or leptoquark events and gets the XY, XZ, and YZ plots for each event, to be used for machine learning.

# This script can be run by doing project_plots.py --pressure # --input_path /path/to/simulated/events --base_path /path/to/ML/data/ --diffusion # --type ""

#===========================================================================================================================================================

#-----IMPORTS-----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import h5py
import random
import argparse

#----- ARGPARSE -----
parser = argparse.ArgumentParser()
parser.add_argument("--pressure", 
                    type = int, 
                    required = True,
                    choices = [1, 5, 10, 15, 25], 
                    help = "Air pressure of detector in simulated events (bar)")

parser.add_argument("--inuput_path", 
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
                    choices = [0.05, 0.25, 0.1, 5],
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
file_identify = event_type + "_" + pressure + "_" + diffusion

#-----GET VERTEX------
def get_vertex(file_, eid):
    part = pd.read_hdf(file_, 'MC/particles')
    density = 5.987*pressure
    M = 1000/0.9
    det_size = 1000*np.cbrt((4 * M) / (np.pi * density))/2.0

    x_vertex = part[(part.event_id == eid) & (part.particle_id == 1)].initial_x.iloc[0]
    y_vertex = part[(part.event_id == eid) & (part.particle_id == 1)].initial_y.iloc[0]
    z_vertex = part[(part.event_id == eid) & (part.particle_id == 1)].initial_z.iloc[0]+det_size

    return x_vertex, y_vertex, z_vertex

### ------ PLOTTING 3D EVENT HITS ------
def PlotEvent3D(axis, file_, title, eid, part):
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)

    hits = pd.read_hdf(file_, 'MC/hits')
    event_hits = hits[hits.event_id == eid]
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

    plt.close(fig)

    return sc, x_vertex, y_vertex, z_vertex

###------- GET PROJECTED GRAPHS --------
def graph_train(original_plot):
    points = original_plot._offsets3d
    x= points[0]
    y = points[1]
    z = points[2]
    c = original_plot.get_array()

    #XY projection
    fig1, ax1 = plt.subplots(figsize=(5.12, 5.12), dpi=100)
    sc_xy = ax1.scatter(x, y, c=c, cmap='Spectral', s=5)
    ax1.set_xlabel("X [mm]", fontsize=15, color='black')
    ax1.set_ylabel("Y [mm]", fontsize=15, color='black')
    ax1.set_title("XY Projection", fontsize=15, color='black')
    cbar1 = fig1.colorbar(sc_xy, ax=ax1, shrink=0.5, aspect=10, pad=0.09)
    cbar1.set_label("Energy", fontsize=12, color='black')

    # YZ projection
    fig2, ax2 = plt.subplots(figsize=(5.12, 5.12), dpi=100)
    sc_yz = ax2.scatter(y, z, c=c, cmap='Spectral', s=5)
    ax2.set_xlabel("Y [mm]", fontsize=15, color='black')
    ax2.set_ylabel("Z [mm]", fontsize=15, color='black')
    ax2.set_title("YZ Projection", fontsize=15, color='black')
    cbar2 = fig2.colorbar(sc_yz, ax=ax2, shrink=0.5, aspect=10, pad=0.09)
    cbar2.set_label("Energy", fontsize=12, color='black')

    # XZ projection
    fig3, ax3 = plt.subplots(figsize=(5.12, 5.12), dpi=100)
    sc_xz = ax3.scatter(x, z, c=c, cmap='Spectral', s=5)
    ax3.set_xlabel("X [mm]", fontsize=15, color='black')
    ax3.set_ylabel("Z [mm]", fontsize=15, color='black')
    ax3.set_title("XZ Projection", fontsize=15, color='black')
    cbar3 = fig3.colorbar(sc_xz, ax=ax3, shrink=0.5, aspect=10, pad=0.09)
    cbar3.set_label("Energy", fontsize=12, color='black')

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    return fig1, fig2, fig3

###------- SAVE FIGURES --------

def get_split():
    r= random.random()
    if r < 0.7:
        return 'train'
    elif r < 0.9:
        return 'val'
    else:
        return 'test'

def save_plot(eid, train_xy, train_yz, train_xz, base_path = base_path, split = None):
    image_path = os.path.join(base_path, "images", split)
    os.makedirs(image_path, exist_ok=True)
    # Save the plots
    train_xy.savefig(os.path.join(image_path, f'event_{eid}_{file_identify}_xy.png'))
    train_yz.savefig(os.path.join(image_path, f'event_{eid}_{file_identify}_yz.png'))
    train_xz.savefig(os.path.join(image_path, f'event_{eid}_{file_identify}_xz.png'))

def get_training_data(h5file_path, base_path=base_path, completed_log=None):
    hits = pd.read_hdf(h5file_path, 'MC/hits')
    part = pd.read_hdf(h5file_path, 'MC/particles')
    event_ids = hits['event_id'].unique()

    # Read completed events from log file if exists
    if completed_log and os.path.exists(completed_log):
        with open(completed_log, 'r') as f:
            completed_events_set = set(int(line.strip()) for line in f)
    else:
        completed_events_set = set()

    total_events = len(event_ids)
    already_completed = len(completed_events_set)

    print(f"Total events in file: {total_events}")
    print(f"Already completed events: {already_completed}")

    new_completed_events = 0

    for i, eid in enumerate(event_ids):
        if eid in completed_events_set:
            # Skip already processed
            continue

        print(f"Processing event {eid} ({new_completed_events + 1}/{total_events - already_completed})")
        split = get_split()

        result = PlotEvent3D(111, h5file_path, "", eid, part)

        if result[0] is None:
            print(f"Skipping event {eid} due to missing data.")
            continue

        sc, x_vertex, y_vertex, z_vertex = result

        train_xy, train_yz, train_xz = graph_train(sc)

        #LABELS
        
        # XY projection
        label_path = os.path.join(base_path, "labels", split)
        os.makedirs(label_path, exist_ok=True)
        label_file_xy = os.path.join(label_path, f"event_{eid}_{file_identify}_xy.txt")

        ax_xy = train_xy.gca()
        train_xy.canvas.draw()
        x_disp, y_disp = ax_xy.transData.transform((x_vertex, y_vertex))  # convert to pixels
        
        image_w, image_h = train_xy.canvas.get_width_height()
        cx = x_disp / image_w
        cy = y_disp / image_h
        w = h = 0.01
        print(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        print(f"img width: {image_w}\nimage height: {image_h}")
        with open(label_file_xy, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # YZ projection
        label_file_yz = os.path.join(label_path, f"event_{eid}_{file_identify}_yz.txt")
        ax_yz = train_yz.gca()
        train_yz.canvas.draw()
        y_disp, z_disp = ax_yz.transData.transform((y_vertex, z_vertex))
        
        image_w, image_h = train_yz.canvas.get_width_height()
        cy = y_disp / image_w
        cz = z_disp / image_h
        print(f"0 {cy:.6f} {cz:.6f} {w:.6f} {h:.6f}\n")
        print(f"img width: {image_w}\nimage height: {image_h}")
        with open(label_file_yz, "w") as f:
            f.write(f"0 {cy:.6f} {cz:.6f} {w:.6f} {h:.6f}\n")

        # XZ projection
        label_file_xz = os.path.join(label_path, f"event_{eid}_{file_identify}_xz.txt")
        ax_xz = train_xz.gca()
        train_xz.canvas.draw()
        x_disp, z_disp = ax_xz.transData.transform((x_vertex, z_vertex))
        cx = x_disp / image_w
        cz = z_disp / image_h
        print(f"0 {cy:.6f} {cz:.6f} {w:.6f} {h:.6f}\n")
        print(f"img width: {image_w}\nimage height: {image_h}")
        with open(label_file_xz, "w") as f:
            f.write(f"0 {cx:.6f} {cz:.6f} {w:.6f} {h:.6f}\n")

        print(f"Completed event {eid}\n")

        save_plot(eid, train_xy, train_yz, train_xz, base_path=base_path, split=split)

        # Log the completed event
        if completed_log:
            with open(completed_log, 'a') as f:
                f.write(f"{eid}\n")

        new_completed_events += 1

        print(f"Completed event {eid} ({new_completed_events}/{total_events - already_completed})")

    print(f"New events processed this run: {new_completed_events}")
    print(f"Total events completed (including previous): {already_completed + new_completed_events}")

    return new_completed_events


### -----CREATE DATASET-----

# Ensure completed_events folder exists
completed_events_folder = os.path.join(base_path, "completed_events")
os.makedirs(completed_events_folder, exist_ok=True)

# Completed files log
completed_files_log = os.path.join(base_path, "completed_files.txt")

# Read completed files
if os.path.exists(completed_files_log):
    with open(completed_files_log, "r") as f:
        completed_files_set = set(line.strip() for line in f)
else:
    completed_files_set = set()

if os.path.isdir(input_path):
    h5files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.h5')]
else:
    h5files = [input_path]

# Filter out files already processed
h5files = [f for f in h5files if os.path.basename(f) not in completed_files_set]

completed_files = 0

for h5file in h5files:
    print(f"Processing file: {h5file}")

    completed_events_file = os.path.join(completed_events_folder, os.path.basename(h5file) + ".txt")

    new_completed = get_training_data(h5file, base_path, completed_log=completed_events_file)

    print(f"Finished processing file: {h5file}")
    print(f"Total files processed: {completed_files + 1}/{len(h5files)}")
    print(f"New events processed: {new_completed}\n")

    if os.path.exists(completed_events_file):
        os.remove(completed_events_file)
        print(f"Deleted completed events file: {completed_events_file}")

    with open(completed_files_log, "a") as f:
        f.write(os.path.basename(h5file) + "\n")

    completed_files += 1


print(f"Completed processing {completed_files} files.")
