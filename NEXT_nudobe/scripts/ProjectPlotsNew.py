
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
from TrackReconstruction_functions import *
from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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

#-----SPLIT ELECTRON TRACKS-----
def split_e_tracks(hits_df, vertex, eid, diffusion, z_shift):
    event_hits = hits_df[hits_df["event_id"] == eid].copy()
    if diffusion != "nodiff":
        event_hits["z"] = event_hits["z"] - z_shift
    hits = event_hits[["x", "y", "z"]].values
    hits_from_vertex = hits - vertex
    radii = np.linalg.norm(hits_from_vertex, axis=1)
    directions = hits_from_vertex / radii[:, None]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(directions)
    
    event_hits["cluster"] = kmeans.labels_
    track_0 = event_hits[event_hits["cluster"] == 0].sort_values("time")
    track_1 = event_hits[event_hits["cluster"] == 1].sort_values("time")
    
    return track_0, track_1

#-----CLUSTER HITS-----
def get_node_centers_df(hits_df, eid, pressure, diffusion):
    #print(f"diffusion: {diffusion}")
    #print(f"pressure: {pressure}")
    data = hits_df[hits_df["event_id"] == int(eid)].copy()
    if np.isnan(data.z.mean()) or data.z.empty:
        print(f"Skipping event {eid} because hits are missing or invalid")
        return pd.DataFrame()
    #print(pressure, type(pressure))
    #print(diffusion, type(diffusion))
    Diff_smear, energy_threshold, diff_scale_factor, radius_sf, group_sf, Tortuosity_dist, voxel_size, det_size  = InitializeParams(pressure, diffusion)
    # voxel_sf=1.1
    # energy_threshold=0
    # energy_threshold = 0.0004
    #diff_scale_factor = diff_scale_change
    """
    print("Diffussion smear is: ",        Diff_smear,            "mm/sqrt(cm)")
    print("Energy threshold is: ",        1000*energy_threshold, "keV")
    print("diffision scale factor is: ",  diff_scale_factor)
    print("Radius scale factor is: ",     radius_sf)
    print("Hit grouping factor is: ",     group_sf)
    print("Tortuosity distance scale is:", Tortuosity_dist)
    print("The voxel size is:",           voxel_size)
    print("The det_size is", det_size)
    """
    #print(f"diffusion: {diffusion}")
    if (diffusion == "next1t"):
        mean_sigma=6
    elif (diffusion == "nodiff"):
        #print(f"check diffusion: {diffusion}")
        mean_sigma=10/np.sqrt(pressure)
        #print(f"check mean sigma: {mean_sigma}")
    else:
        if data.z.mean() < 0:
            print("skipping event, invalid data.z.mean()")
            return None, None
        mean_sigma = diff_scale_factor*Diff_smear*np.sqrt(0.1*data.z.mean())

    # The expected diffusion is less than vox size so replace
    if (mean_sigma < 1.5*voxel_size):
        mean_sigma = 1.5*voxel_size

    if diffusion != "nodiff":
        mean_sigma = mean_sigma * 1.3 ### ADJUST!
    
    # Create the bins ---- 
    xbw  = mean_sigma
    xmin = -det_size - mean_sigma/2 
    xmax = det_size  + mean_sigma/2
    
    ybw  = mean_sigma
    ymin = -det_size - mean_sigma/2 
    ymax = det_size  + mean_sigma/2
    
    # This shifts the z pos of the events so 0 is at anode
    # can set this to zero
    #z_shift = det_size
    z_shift = 0
    
    zbw=mean_sigma
    zmin=-det_size + z_shift - mean_sigma/2 
    zmax=det_size + z_shift + mean_sigma/2

    # bins for x, y, z
    xbins = np.arange(xmin, xmax+xbw, xbw)
    ybins = np.arange(ymin, ymax+ybw, ybw)
    zbins = np.arange(zmin, zmax+zbw, zbw)
    
    # center bins for x, y, z
    xbin_c = xbins[:-1] + xbw / 2
    ybin_c = ybins[:-1] + ybw / 2
    zbin_c = zbins[:-1] + zbw / 2

    # If there are overlapping voxels, merge them. Otherwise the energy gets messed up
    data = (data.groupby(["event_id", "x", "y", "z"], as_index=False)["energy"].sum())
    # then sort it based on the x,y,z
    data = data.sort_values(by=['x', "y", "z"]).reset_index(drop=True)
    """
    print(f"xmin: {xmin}, xmax: {xmax}, xbw: {xbw}")
    print(f"ymin: {ymin}, ymax: {ymax}, ybw: {ybw}")
    print(f"zmin: {zmin}, zmax: {zmax}, zbw: {zbw}")
    
    print(f"data x range: {data['x'].min()} to {data['x'].max()}")
    print(f"data y range: {data['y'].min()} to {data['y'].max()}")
    print(f"data z range: {data['z'].min()} to {data['z'].max()}")
    """
    # Apply grouping
    data_copy = data.copy()
    #df_merged = CutandRedistibuteEnergy(data_copy, energy_threshold)
    print("applied grouping")

    
    
    if diffusion == "next1t":
        mean_sigma_group = 10
    elif (diffusion == "nodiff"):
        #print(f"diffusion: {diffusion}")
        mean_sigma_group=15
    else:
        mean_sigma_group = group_sf*Diff_smear*np.sqrt(0.1*data.z.mean())

    if (mean_sigma_group < voxel_size/2.0):
        mean_sigma_group = voxel_size/2.0
        """
    print("Mean sigma group", mean_sigma_group)
    print("Number of hits:", len(data))
    print("Mean z:", data.z.mean(), " diffusion = ", mean_sigma)
    print("Mean Sigma: ", mean_sigma)
    """
    print("Mean sigma group:", mean_sigma_group)

    
    df_merged = GroupHits(data_copy, mean_sigma_group)
    #print(f"df_merged: {df_merged.group_id.unique()}")
    # Apply clustering
    node_centers_df = []
    energy_gid_dict = {}
    for gid in sorted(df_merged.group_id.unique()):
        #print(f"applying clustering for {gid}")
        temp_df = df_merged[df_merged.group_id == gid]
        energy_sum = temp_df["energy"].sum()
        energy_gid_dict[gid] = energy_sum
        temp_df.reset_index(drop=True, inplace=True)
        node_centers_df.append(Cluster(temp_df, mean_sigma))
    max_energy_gid = max(energy_gid_dict, key=energy_gid_dict.get)
    node_centers_df = pd.concat(node_centers_df, ignore_index=True)
    node_centers_df = node_centers_df[node_centers_df.group_id == max_energy_gid]
    #print("COMPLETED NODE CENTERS DF")
    return node_centers_df

#-----NEAREST NEIGHBORS ALGORITHM-----
def nearest_neighbors(XC, YC):
    coords = np.column_stack([XC, YC])
    tree = BallTree(coords) #create spatial index
    centroid = coords.mean(axis=0) #find avg position of all hits
    dists = np.linalg.norm(coords - centroid, axis=1)
    start = np.argmax(dists) #start at furthest dist
    
    route = [start]
    visited = set([start])
    while len(visited) < len(coords): # go until every point is visited
        current = route[-1] # last point added to route is current position
        _, indices = tree.query([coords[current]], k=len(coords)) # queries all points sorted by dist
        # take nearest unvitited point and add to route
        for idx in indices[0]:
            if idx not in visited:
                route.append(idx)
                visited.add(idx)
                break
    return np.array(route)

### ------ PLOTTING 3D EVENT HITS ------
def PlotEvent3D(hits, part, eid, z_shift, pressure, diffusion):
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)


    event_hits = hits[hits.event_id == eid].copy()
    clustered_hits = get_node_centers_df(hits, eid, pressure, diffusion)

    if diffusion != "nodiff":
        event_hits["z"] = event_hits["z"] - z_shift
        clustered_hits["z"] = clustered_hits["z"] - z_shift
        
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

    xc = clustered_hits.x.values
    yc = clustered_hits.y.values
    zc = clustered_hits.z.values

    plt.close(fig)


    return (x, y, z, c), (xc, yc, zc), (x_vertex, y_vertex, z_vertex)

###------- GET DATASET FUNCTION --------
def get_data(XYZC, clustered_xyz, vertex, eid, file_identify, split):
    x, y, z, c = XYZC
    x_vertex, y_vertex, z_vertex = vertex
    xc, yc, zc = clustered_xyz

    projections = [("xy", x, y, xc, yc, x_vertex, y_vertex),
                   ("yz", y, z, yc, zc, y_vertex, z_vertex),
                   ("xz", x, z, xc, zc, x_vertex, z_vertex)]
    
    w = h = 0.02 #bounding box size
    axis_limits = {} 

    
    for dim, X, Y, XC, YC, vx, vy in projections:
        #initialize paths
        img_filename = f"event_{eid}_{file_identify}_{dim}_{split}.png"
        label_filename = f"event_{eid}_{file_identify}_{dim}_{split}.txt"

        #plot the event
        fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
        ax.scatter(X, Y, c=c, cmap="Spectral", s=5, alpha=.1)

        order = nearest_neighbors(XC, YC) # ordering the hits based on the shortest path to go through all the hits
        XC = XC[order]
        YC = YC[order]
        sharp_angles = []
        colors = plt.cm.plasma(np.linspace(0, 1, len(XC)))
        for i, (hx, hy) in enumerate(zip(XC, YC)):
            #ax.scatter(hx, hy, color=colors[i], s=7, alpha=.7) #plot for the clustered hits
            if i == 0 or i == len(XC) - 1:
                sharp_angles.append([hx, hy, i]) #adding endpoints as a false vertex

            else:
                if diffusion == "nodiff":
                    density_radius = 15 #define a region to get the density of points
                else:
                    density_radius = 15
                neighbors = np.sum(np.sqrt((XC - hx)**2 + (YC - hy)**2) < density_radius)
                min_index_sep = int(max(1, neighbors // 1.5)) 
                min_dist = max(10, neighbors * 2)  # denser = need to go further

                #walk until reaching min dist. or first i
                
                i_left = max(0, i - min_index_sep)
                
                while i_left > 0 and np.sqrt((XC[i_left]-hx)**2 + (YC[i_left]-hy)**2) < min_dist:
                    i_left -= 1
    
                i_right = min(len(XC) - 1, i + min_index_sep)
                while i_right < len(XC)-1 and np.sqrt((XC[i_right]-hx)**2 + (YC[i_right]-hy)**2) < min_dist:
                    i_right += 1

                #getting the vectors and angle
                v1 = np.array([XC[i_left] - XC[i], YC[i_left] - YC[i]]) 
                v2 = np.array([XC[i_right] - XC[i], YC[i_right] - YC[i]]) 
                norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
                if norm_product == 0:
                    continue
                cos_theta = np.dot(v1, v2) / norm_product
                cos_theta = np.clip(cos_theta, -1, 1) 
                theta = np.arccos(cos_theta)

                smallest_angle=True
                dist_to_vertex = np.sqrt((hx - vx)**2 + (hy - vy)**2)

                # getting rid of any points that are too close by choosing the sharpest angle
                for j, point in enumerate(sharp_angles):
                    px, py, pi = point[0], point[1], point[2] 
                    dist_to_point = np.sqrt((hx - point[0])**2 + (hy - point[1])**2)
                    neighbors_pi = np.sum(np.sqrt((XC - px)**2 + (YC - py)**2) < density_radius)
                    
                    if diffusion == "nodiff":
                        min_dist_pi = max(10, neighbors_pi * 2)
                    else:
                        min_dist_pi = max(25, neighbors_pi * 20)
                        
                    if dist_to_point < min_dist_pi:
                        px, py, pi = point[0], point[1], point[2]

                        if pi == 0 or pi == len(XC) - 1:  # keep endpoints
                            break
                        
                        if pi <= 1 or pi >= len(XC) - 2:  # not enough room for neighbors
                            break

                        pi_left = pi - 1
                        while pi_left > 0 and np.sqrt((XC[pi_left]-px)**2 + (YC[pi_left]-py)**2) < min_dist:
                            pi_left -= 1
                        
                        pi_right = pi + 1
                        while pi_right < len(XC)-1 and np.sqrt((XC[pi_right]-px)**2 + (YC[pi_right]-py)**2) < min_dist:
                            pi_right += 1
                        
                        v1 = np.array([XC[pi_left] - px, YC[pi_left] - py])
                        v2 = np.array([XC[pi_right] - px, YC[pi_right] - py])
                        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
                        if norm_product == 0:
                            continue
                        cos_theta_new = np.dot(v1, v2) / norm_product
                        cos_theta_new = np.clip(cos_theta_new, -1, 1) 
                        theta_new = np.arccos(cos_theta_new)
                        #print(f"i={i}, neighbors={neighbors}, min_index_sep={min_index_sep}, i_left={i_left}, i_right={i_right}, theta={np.degrees(theta):.1f}")
                        if theta < theta_new:
                            sharp_angles.pop(j)

                        else:
                            smallest_angle = False
                        break
                max_angle_threshold = 5*np.pi/6  # default 30 degrees
                min_angle_threshold = np.pi/6
                if neighbors > 6:
                    max_angle_threshold = np.pi/2 
                    min_angle_threshold = 0    
                if (min_angle_threshold <= theta <= max_angle_threshold) and (dist_to_vertex > 15) and smallest_angle:
                    sharp_angles.append([hx, hy, i])

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
        #plt.show()

        #normalize for yolo (0-1)

        sharp_angle_labels = []
        #ax.scatter(vx, vy, color="black", s=10)
        for i in sharp_angles:
            #ax.scatter(i[0], i[1], color="blue", s=10)
            nx = (i[0] - xlim[0]) / (xlim[1] - xlim[0]) # dist from left / total width
            ny = 1 - ((i[1] - ylim[0]) / (ylim[1] - ylim[0]))
            sharp_angle_labels.append(f"1 {nx:.6f} {ny:.6f} {w:.6f} {h:.6f}")

        cx = (vx - xlim[0]) / (xlim[1] - xlim[0]) # dist from left / total width
        cy = 1 - ((vy - ylim[0]) / (ylim[1] - ylim[0])) # subtract from 1 since y=0 is at top in yolo


        # check if normalized coordinates are valid
        if not (0 <= cx <= 1) or not (0 <= cy <= 1):
            print(f"Skipping event {eid} due to out-of-bounds normalized coordinates: cx={cx}, cy={cy}")
            return None, None, None, None


        print(f"label: 0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        #save label
        with open(label_filename, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            for label in sharp_angle_labels:
                f.write(label + "\n")
        
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
    (XYZC, clustered, vertex) = PlotEvent3D(hits_df, part_df, eid, z_shift, pressure, diffusion)
    result = get_data(XYZC, clustered, vertex, eid, file_identify, split=split)
    if result[0] is None:
        continue
    x, y, z, axis_limits = result
    all_limits.append({"event_id": eid, "axis_limits": json.dumps(axis_limits)})
    print(f"completed event {eid}")


limits_df = pd.DataFrame(all_limits)
limits_df.to_json(f"{file_identify}_limits.jsonl", orient="records", lines=True)
