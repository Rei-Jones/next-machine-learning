# imports
import os
import pandas as pd
import numpy as np
import json
import random 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
import glob
from TrackReconstruction_functions import *
import sys

def get_zshift(pressure):
    density = 5.987*pressure
    M = 1000/0.9
    det_size = 1000*np.cbrt((4 * M) / (np.pi * density))/2.0

    return det_size

def get_vertex(part, eid):

    x_vertex = part[(part.event_id == eid) & (part.particle_id == 1)].initial_x.iloc[0]
    y_vertex = part[(part.event_id == eid) & (part.particle_id == 1)].initial_y.iloc[0]
    z_vertex = part[(part.event_id == eid) & (part.particle_id == 1)].initial_z.iloc[0]

    return x_vertex, y_vertex, z_vertex

def get_true_angle(part, eid):
    p_x1 = part[(part.event_id == eid) & (part.particle_id == 1)].initial_momentum_x.iloc[0]
    p_y1 = part[(part.event_id == eid) & (part.particle_id == 1)].initial_momentum_y.iloc[0]
    p_z1 = part[(part.event_id == eid) & (part.particle_id == 1)].initial_momentum_z.iloc[0]
    
    p_x2 = part[(part.event_id == eid) & (part.particle_id == 2)].initial_momentum_x.iloc[0]
    p_y2 = part[(part.event_id == eid) & (part.particle_id == 2)].initial_momentum_y.iloc[0]
    p_z2 = part[(part.event_id == eid) & (part.particle_id == 2)].initial_momentum_z.iloc[0]
    
    cos12 = ((p_x1 * p_x2) + (p_y1 * p_y2) + (p_z1 * p_z2))/(np.sqrt((p_x1 ** 2) + (p_y1 ** 2) + (p_z1 ** 2)) * np.sqrt((p_x2 ** 2) + (p_y2 ** 2) + (p_z2 ** 2)))
    return cos12

def get_true_e_vectors(eid, part_df, vertex, particle_ids=[1,2]):
    vertex = np.array(vertex, dtype=float)
    
    vectors=[]

    for pid in particle_ids:
        row = part_df[(part_df.event_id==eid) & (part_df.particle_id==pid)].iloc[0]

        p = np.array([row.initial_momentum_x, row.initial_momentum_y, row.initial_momentum_z], dtype=float)

        #unit dir
        direction = p / np.linalg.norm(p)

        endpoint = vertex + direction

        vectors.append((endpoint, direction))

    vectors = [
        {"endpoint": vec[0].tolist(), "direction": vec[1].tolist()}
        for vec in vectors
        ]

    return vectors

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
    #print("Mean sigma group:", mean_sigma_group)

    
    df_merged = GroupHits(data_copy, mean_sigma_group)
    # Apply clustering
    node_centers_df = []
    energy_gid_dict = {}
    for gid in sorted(df_merged.group_id.unique()):
        temp_df = df_merged[df_merged.group_id == gid]
        energy_sum = temp_df["energy"].sum()
        energy_gid_dict[gid] = energy_sum
        temp_df.reset_index(drop=True, inplace=True)
        node_centers_df.append(Cluster(temp_df, mean_sigma))
    max_energy_gid = max(energy_gid_dict, key=energy_gid_dict.get)
    node_centers_df = pd.concat(node_centers_df, ignore_index=True)
    node_centers_df = node_centers_df[node_centers_df.group_id == max_energy_gid]
    return node_centers_df

def get_reco_e_vectors(hits_df, part_df, eid, N, diffusion, pressure, z_shift, vertex=None, plot=False):

    node_centers_df = get_node_centers_df(hits_df, eid, pressure, diffusion)
    df = node_centers_df.copy() 
    if diffusion != "nodiff":
        df["z"] = df["z"] - z_shift
    
    if vertex is None:
        vertex = get_vertex(part_df, eid)
    else:
        vertex = vertex
        
    hits = df[["x", "y", "z"]].values
    
    dists = np.linalg.norm(hits - vertex, axis=1) 
    N=N #cluster hits total
    closest_hits = hits[np.argsort(dists)[:N]] # sort by dist
    hits_from_vertex = closest_hits - vertex # use vertex as reference
    radii = np.linalg.norm(hits_from_vertex, axis=1)

    # keep only hits that are not at the vertex
    mask = radii > 10 #events <1mm from vertex are not used
    hits_from_vertex = hits_from_vertex[mask]
    closest_hits = closest_hits[mask]
    radii = radii[mask]

    if len(radii)<2:
        #print("Not enough hits to cluster")
        return[]
    # normalize
    directions = hits_from_vertex / radii[:, None]
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(directions) # 2 clusters
    labels = kmeans.labels_
    
    vectors = []
    for cluster_id in [0,1]:
        cluster_hits = closest_hits[labels == cluster_id] #cluster hits
    
        if len(cluster_hits) == 0:
            #print("Not enough hits")
            continue
            
        #get vectors
        dists = np.linalg.norm(cluster_hits - vertex, axis=1) 
        endpoint = cluster_hits[np.argmin(dists)]
        direction_vector = endpoint - vertex
        vectors.append((endpoint, direction_vector))

    # plot clustered hits
    if plot == True:
        vertex_true = get_vertex(part_df, eid)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        part_1 = part_df[(part_df.event_id == eid) & (part_df.particle_id == 1)]
        part_2 = part_df[(part_df.event_id == eid) & (part_df.particle_id == 2)]
    
        p1 = part_1[['initial_momentum_x', 'initial_momentum_y', 'initial_momentum_z']].values.flatten()
        p2 = part_2[['initial_momentum_x', 'initial_momentum_y', 'initial_momentum_z']].values.flatten()    
        colors = ["red", "blue"]
            
        for cluster_id in [0, 1]:
            cluster_hits = closest_hits[labels == cluster_id]
            ax.scatter(cluster_hits[:, 0], cluster_hits[:, 1], cluster_hits[:, 2],
                           label=f"Cluster {cluster_id}", color=colors[cluster_id], s=40)
    
        ax.scatter(vertex[0], vertex[1], vertex[2], color="black", s=60, label="Vertex Smeared")
        ax.scatter(vertex_true[0], vertex_true[1], vertex_true[2], color="green", s=60, label="Vertex")
            #ax.scatter(event_hits["x"], event_hits["y"], event_hits["z"], color="black", alpha=0.1, s=30)
        ax.scatter(df["x"], df["y"], df["z"], color="purple", marker="+", alpha=0.5)
        ax.quiver(vertex[0], vertex[1], vertex[2], vectors[0][1][0], vectors[0][1][1], vectors[0][1][2], label="reco", color="black")
        ax.quiver(vertex[0], vertex[1], vertex[2], vectors[1][1][0], vectors[1][1][1], vectors[1][1][2], color="black")
        ax.quiver(vertex_true[0], vertex_true[1], vertex_true[2], p1[0], p1[1], p1[2], length=10, label="true", color="purple")
        ax.quiver(vertex_true[0], vertex_true[1], vertex_true[2], p2[0], p2[1], p2[2], length=10, color="purple")
        ax.legend()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim((vertex[0] - 50), (vertex[0] + 50))
        ax.set_ylim((vertex[1] - 50), (vertex[1] + 50))
        ax.set_zlim((vertex[2] - 50), (vertex[2] + 50))
        ax.set_title("KMeans Clustering of Hits")
    
        plt.show()

    vectors = [
        {"endpoint": vec[0].tolist(), "direction": vec[1].tolist()}
        for vec in vectors
        ]
    
    return vectors

def get_true_angle(part, eid):
    p_x1 = part[(part.event_id == eid) & (part.particle_id == 1)].initial_momentum_x.iloc[0]
    p_y1 = part[(part.event_id == eid) & (part.particle_id == 1)].initial_momentum_y.iloc[0]
    p_z1 = part[(part.event_id == eid) & (part.particle_id == 1)].initial_momentum_z.iloc[0]
    
    p_x2 = part[(part.event_id == eid) & (part.particle_id == 2)].initial_momentum_x.iloc[0]
    p_y2 = part[(part.event_id == eid) & (part.particle_id == 2)].initial_momentum_y.iloc[0]
    p_z2 = part[(part.event_id == eid) & (part.particle_id == 2)].initial_momentum_z.iloc[0]
    
    cos12 = ((p_x1 * p_x2) + (p_y1 * p_y2) + (p_z1 * p_z2))/(np.sqrt((p_x1 ** 2) + (p_y1 ** 2) + (p_z1 ** 2)) * np.sqrt((p_x2 ** 2) + (p_y2 ** 2) + (p_z2 ** 2)))
    return cos12

def add_smear(part_df, eid, smear):
    vertex = get_vertex(part_df, eid)
        
    phi = np.random.uniform(0, 2*np.pi)
    costheta = np.random.uniform(-1, 1)
    theta = np.arccos(costheta)
                    
    r = smear
    
    x_new = vertex[0] + r * np.sin(theta) * np.cos(phi)
    y_new = vertex[1] + r * np.sin(theta) * np.sin(phi)
    z_new = vertex[2] + r * np.cos(theta)
    
    vertex_smeared = np.array([x_new, y_new, z_new])

    return vertex_smeared

def convert_np_types(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert array to list
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

file = sys.argv[1]
pressure = sys.argv[2]
diffusion = sys.argv[3]
axis_limits_file = sys.argv[4]

pressure = int(pressure)

z_shift = get_zshift(pressure)

part_df = pd.read_hdf(file, "MC/particles")
hits_df = pd.read_hdf(file, "MC/hits")
axis_limits_df = pd.read_json(axis_limits_file, lines=True)

dict = {}

for eid, data in part_df.groupby("event_id"):
    print(f"starting: {eid}")
    event_dict = {}
    vertex = get_vertex(part_df, eid)
    cos12_true = get_true_angle(part_df, eid)
    vectors = get_true_e_vectors(eid, part_df, vertex)
    vectors_reco = get_reco_e_vectors(hits_df, part_df, eid, 10, diffusion, pressure, z_shift, vertex=vertex, plot=False)
    rv1 = vectors_reco[0]["direction"]
    rv2 = vectors_reco[1]["direction"]
    cos12_reco = np.dot(rv1, rv2)/(np.linalg.norm(rv1) * np.linalg.norm(rv2))
    event_dict["vertex"] = vertex
    match = axis_limits_df[axis_limits_df["event_id"] == eid]
    if not match.empty:
        event_dict["axis_limits"] = json.loads(match.iloc[0]["axis_limits"])
    else:
        event_dict["axis_limits"] = None
    event_dict["vectors_true"] = vectors
    event_dict["cos12_true"] = cos12_true
    event_dict["vectors_reco"] = vectors_reco
    event_dict["cos12_reco"] = cos12_reco
    energies = part_df[(part_df.event_id == eid) & (part_df.primary == 1)].kin_energy.tolist()
    event_dict["energies"] = energies

    for smear in range(1, 6):
        vertex_smeared = add_smear(part_df, eid, smear)
        vectors_smeared = get_reco_e_vectors(hits_df, part_df, eid, 10, diffusion, pressure, z_shift, vertex=vertex_smeared, plot=False)
        sv1 = vectors_smeared[0]["direction"]
        sv2 = vectors_smeared[1]["direction"]
        cos12_smeared = np.dot(sv1, sv2)/(np.linalg.norm(sv1) * np.linalg.norm(sv2))
        event_dict[f"vertex_{smear}mm"] = vertex_smeared
        event_dict[f"vectors_{smear}mm"] = vectors_smeared
        event_dict[f"cos12_{smear}mm"] = cos12_smeared
    
    print(event_dict)
    
    dict[eid] = event_dict

min_eid = part_df["event_id"].min()
file_out = file[:-3]
file_out = f"{file_out}_{min_eid}_data.jsonl"

with open(file_out, "w") as f:
    for event_id, event_data in dict.items():
            # Create a flat dict including file and event info
        line_data = {
            "file": file,
            "event_id": event_id,
            **event_data
        }
        f.write(json.dumps(line_data, default=convert_np_types) + "\n")