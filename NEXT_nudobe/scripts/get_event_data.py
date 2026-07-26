import pandas as pd
import numpy as np
import json
import sys
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_order
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx

pd.options.mode.chained_assignment = None  # or 'warn' to re-enable

# USAGE: python kinematics_reconstruction.py <infile> <model> <pressure> <diffusion> <axis_limits_file>
# using: https://github.com/kvjmistry/NEXT_nudobe/blob/main/scripts/kinematics_reconstruction.py

# Input file
infile     = sys.argv[1]
model      = sys.argv[2]
pressure = sys.argv[3]
diffusion = sys.argv[4]
axis_limits_file = sys.argv[5]
file_out = f"{infile}_data.jsonl"

# Event, when testing on single events
#evt_id = 0

# Maximum hit distance from vertex for electron direction calculation
R = 12 #mm 

# Reconstruction method: either 'endtrack' seed, or 'vertex' seed
reco_seed_method = 'vertex_seed'
#reco_seed_method = 'endtrack_seed'

# Particles dataframe.
df_particles_allevts = pd.read_hdf(infile, "MC/particles")

# Hits dataframe.
df_hits_allevts = pd.read_hdf(infile, "MC/hits")

axis_limits_df = pd.read_json(axis_limits_file, lines=True)

nevts = len(df_particles_allevts.event_id.unique())
event_ids = sorted(df_particles_allevts.event_id.unique())
print("Number of events: ", nevts)

# ----------------------------------------
# Function to get the z-shift of an event
def get_zshift(pressure):
    density = 5.987*pressure
    M = 1000/0.9
    det_size = 1000*np.cbrt((4 * M) / (np.pi * density))/2.0

    return det_size
# ----------------------------------------
# Function to get the true vertex location
def get_vertex(part, eid):

    x_vertex = part[(part.event_id == eid) & (part.particle_id == 1)].initial_x.iloc[0]
    y_vertex = part[(part.event_id == eid) & (part.particle_id == 1)].initial_y.iloc[0]
    z_vertex = part[(part.event_id == eid) & (part.particle_id == 1)].initial_z.iloc[0]

    return x_vertex, y_vertex, z_vertex
# ----------------------------------------
# Function to smear the vertex position by a certain distance
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
# ----------------------------------------
# Simple function to ensure T1 > T2
def reorder_electrons_fun(T1,T2):
    if T1 > T2:
        return T1, T2
    else:
        return T2, T1
# ----------------------------------------
# cos(theta) = (p1x*p2x + p1y*p2y + p1z*p2z) / (p1*p2)
def costheta_fun(p1x,p1y,p1z,p2x,p2y,p2z):
    p1 = np.sqrt(p1x**2 + p1y**2 + p1z**2)
    p2 = np.sqrt(p2x**2 + p2y**2 + p2z**2)
    costheta = (p1x*p2x + p1y*p2y + p1z*p2z) / (p1*p2)
    return costheta
# ----------------------------------------
def TrackDir(Track, vertex):
    # Convert values to an array
    track_arr = Track[['x', 'y', 'z']].to_numpy()
    return track_arr[0] - vertex
# ----------------------------------------
def CalcTrackAngle(Track1, Track2, vertex):

    t1_delta = 0
    # t2_delta = 0

    dir_track1 = TrackDir(Track1, vertex)
    dir_track2 = TrackDir(Track2, vertex)

    cosine = costheta_fun(dir_track1[0], dir_track1[1], dir_track1[2], dir_track2[0], dir_track2[1], dir_track2[2])

    # Just check that first hit was not reco in the wrong direction for track 1
    if (cosine > 0.97 and len(Track1) > 1):
        dir_track1 = TrackDir(Track1.iloc[1:2], vertex)
        dir_track2 = TrackDir(Track2, vertex)
        cosine = costheta_fun(dir_track1[0], dir_track1[1], dir_track1[2], dir_track2[0], dir_track2[1], dir_track2[2])

        t1_delta = -Track1.iloc[1:2].energy.item()
        # t2_delta = Track1.iloc[1:2].energy.item()

    # Just check that first hit was not reco in the wrong direction for track 2
    if (cosine > 0.97  and len(Track2) > 1):
        dir_track1 = TrackDir(Track1, vertex)
        dir_track2 = TrackDir(Track2.iloc[1:2], vertex)
        cosine = costheta_fun(dir_track1[0], dir_track1[1], dir_track1[2], dir_track2[0], dir_track2[1], dir_track2[2])

        t1_delta = Track2.iloc[1:2].energy.item()
        # t2_delta = -Track2.iloc[1:2].energy.item()

    return cosine, t1_delta
# ----------------------------------------
# Find primary ancestor id of a particle in the particle df
def ancestor_id_fun(particle_id_in):
    pid = particle_id_in
    if (pid == 0):
        return pid
    is_primary = 0
    while is_primary == 0:
        is_primary = df_particles[df_particles.particle_id == pid].primary.item()
        if (is_primary):
            return pid
        else:
            mid = df_particles[df_particles.particle_id == pid].mother_id.item()
            pid = mid
# ----------------------------------------
# Function returning the first principal component of 3D hits, 
# i.e. the direction vector of the 'best fit' line in the least squares sense.
# Used both for true and reco hit collections
def line_fit_fun(df_sel_hits, vertex):
    x_array = df_sel_hits.x.to_numpy()
    x_array = np.append(x_array,vertex[0]) # vertex
    y_array = df_sel_hits.y.to_numpy()
    y_array = np.append(y_array,vertex[1]) # vertex
    z_array = df_sel_hits.z.to_numpy()
    z_array = np.append(z_array,vertex[2]) # vertex
    
    data_comb = np.concatenate((x_array[:, np.newaxis], 
                                y_array[:, np.newaxis], 
                                z_array[:, np.newaxis]), axis=1)
    datamean = data_comb.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data_comb - datamean)
    vx = vv[0][0]
    vy = vv[0][1]
    vz = vv[0][2]
    return vx,vy,vz
# ----------------------------------------
def euclidean_distance_fun(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))
# ----------------------------------------
# Return global_hit_ids of nodes in reconstructed e1 and e2, ordered starting from the vertex
def divide_nodes_into_groups_fun(graph, vertex):
    start_coords = vertex
    distances = {}
    
    for node in graph.nodes():
        distances[node] = euclidean_distance_fun(start_coords, graph.nodes[node]['coords'])
    
    # Sort nodes based on their distances from the vertex
    sorted_node_ids = sorted(distances.keys(), key=lambda node: distances[node])

    # Initialize groups
    track_1 = [sorted_node_ids[0]]  # First node with the shortest distance
    track_2 = [sorted_node_ids[1]]  # Second node with the second shortest distance

    # Assign nodes to groups iteratively
    for node_id in sorted_node_ids[2:]:
        distance_to_last_node_group1 = euclidean_distance_fun(graph.nodes[track_1[-1]]['coords'], graph.nodes[node_id]['coords'])
        distance_to_last_node_group2 = euclidean_distance_fun(graph.nodes[track_2[-1]]['coords'], graph.nodes[node_id]['coords'])

        if distance_to_last_node_group1 <= distance_to_last_node_group2:
            track_1.append(node_id)
        else:
            track_2.append(node_id)

    return track_1, track_2
# ----------------------------------------
def add_global_hit_id_column_fun():
    global df_hits
    nhits = len(df_hits)
    # Add  a global_hit_id column to hits dataframe, so that hit IDs are unique within one event
    #df_hits['global_hit_id'] = np.arange(nhits) # this yields a warning, so change with next line
    df_hits = df_hits.assign(global_hit_id = np.arange(nhits))
    return df_hits
# ----------------------------------------
def add_distance_column_fun(df_hits, vertex):
    # global df_hits
    # Add hit distance to vertex, to the hits dataframe. Note that this assumes that vertex is  (0,0,0) for all events
    df_hits['distance'] = np.sqrt((df_hits.x - vertex[0])**2 + (df_hits.y - vertex[1])**2 + (df_hits.z - vertex[2])**2)
    return df_hits
# ----------------------------------------
def gen_quantities_fun():
    global df_particles
    # T_1,gen
    T1_gen = df_particles[df_particles.particle_id == 1].kin_energy.item()
    T2_gen = df_particles[df_particles.particle_id == 2].kin_energy.item()
    T1_gen, T2_gen = reorder_electrons_fun(T1_gen,T2_gen)
    #print(T1_gen,T2_gen,T1_gen+T2_gen)
    
    # cos(theta_gen)
    p1x = df_particles[df_particles.particle_id == 1].initial_momentum_x.item()
    p1y = df_particles[df_particles.particle_id == 1].initial_momentum_y.item()
    p1z = df_particles[df_particles.particle_id == 1].initial_momentum_z.item()
    
    p2x = df_particles[df_particles.particle_id == 2].initial_momentum_x.item()
    p2y = df_particles[df_particles.particle_id == 2].initial_momentum_y.item()
    p2z = df_particles[df_particles.particle_id == 2].initial_momentum_z.item()
    
    costheta_gen = costheta_fun(p1x,p1y,p1z,p2x,p2y,p2z)
    #print(costheta_gen)
    
    return T1_gen, costheta_gen
# ----------------------------------------
def add_ancestor_id_column_fun():
    global df_hits
    nhits = len(df_hits)
    # Add a ancestor_id column to the hits df, with values 1 or 2, to identify which primary electron produced the hit
    
    keys = df_hits.global_hit_id.to_numpy()
    values = np.zeros(nhits, dtype='i')
    pid_array = df_hits.particle_id.to_numpy()
    
    for i in np.arange(nhits):
        values[i] = ancestor_id_fun(pid_array[i])
        
    df_ancestor_id = pd.DataFrame({'global_hit_id': keys, 'ancestor_id': values})
    df_hits = pd.merge(df_hits, df_ancestor_id, on='global_hit_id')
    
    return df_hits
# ----------------------------------------
def true_quantities_fun(vertex):
    global df_hits
    # T1_true. Should be the same as T1_gen
    # We separate hits based on ancestor_id
    e1true_hit_mask = df_hits.ancestor_id == 1
    df_e1true_hits = df_hits[e1true_hit_mask]
    T1_true = df_e1true_hits.energy.sum()
    
    e2true_hit_mask = df_hits.ancestor_id == 2
    df_e2true_hits = df_hits[e2true_hit_mask]
    T2_true = df_e2true_hits.energy.sum()
    
    T1_true, T2_true = reorder_electrons_fun(T1_true,T2_true)
    #print(T1_true,T2_true,T1_true+T2_true)
    
    # costheta_true
    # Normally, for each electron we keep all hits whose distance from vertex is <= R. 
    # For electrons where no hits satisfy this condition, we only take the clostest hit to the vertex, even if its distance is greater than R
    distance1_mask = df_e1true_hits.distance <= R
    nhits1true = distance1_mask.sum()
    if (nhits1true >= 1):
        df_e1true_sel_hits = df_e1true_hits[distance1_mask]
    else:
        hitrow = df_e1true_hits['distance'].idxmin()
        df_e1true_sel_hits = df_e1true_hits.iloc[hitrow]
        
    distance2_mask = df_e2true_hits.distance <= R
    nhits2true = distance2_mask.sum()
    if (nhits2true >= 1):
        df_e2true_sel_hits = df_e2true_hits[distance2_mask]
    else:
        hitrow = df_e2true_hits['distance'].idxmin()
        df_e2true_sel_hits = df_e2true_hits.iloc[hitrow]

    #print(nhits1true,nhits2true)
    v1x, v1y, v1z = line_fit_fun(df_e1true_sel_hits, vertex)
    v2x, v2y, v2z = line_fit_fun(df_e2true_sel_hits, vertex)
    costheta_true = costheta_fun(v1x,v1y,v1z,v2x,v2y,v2z)
    #print(costheta_true)
    
    return T1_true, costheta_true
# ----------------------------------------
def add_reco_hit_id_column_vertex_seed_fun(vertex):
    global df_hits
    
    # First, create the graph
    # Extract from the dataframe the 3D coordinates of each hit. Hits will become the graph nodes
    graph = nx.Graph()

    # Add vertex at origin not needed. Comment out
    #graph.add_node("vertex", coords=(0., 0., 0.))

    for index, row in df_hits.iterrows():
        graph.add_node(row['global_hit_id'], coords=(row['x'],row['y'],row['z']))
    #graph.nodes(data=True)
    
    # Second, divide nodes into e1 and e2 groups
    #start_node = "vertex"
    e1reco_global_hit_ids, e2reco_global_hit_ids = divide_nodes_into_groups_fun(graph, vertex)
    #print("Track 1:", e1reco_global_hit_ids)
    #print("Track 2:", e2reco_global_hit_ids)
    #print("len(e1reco_global_hit_ids) = ", len(e1reco_global_hit_ids), "\nlen(e2reco_global_hit_ids) = ", len(e2reco_global_hit_ids), "\nlen(e1reco_global_hit_ids) + len(e2reco_global_hit_ids) =", len(e1reco_global_hit_ids)+len(e2reco_global_hit_ids))
    
    # Third, add reco_hit_id column according to ordering in e1reco_global_hit_ids and e2reco_global_hit_ids lists. Reorder rows by 'reco_hit_id'
    keys = e1reco_global_hit_ids
    values = np.zeros(nhits, dtype='i')

    my_dict1 = {}
    reco_hit_id = 1
    for key in e1reco_global_hit_ids:
        my_dict1[key] = reco_hit_id
        reco_hit_id += 1
    df_reco_particle_id1 = pd.DataFrame({'global_hit_id': list(my_dict1.keys()), 'reco_hit_id': list(my_dict1.values())})

    my_dict2 = {}
    reco_hit_id = -1
    for key in e2reco_global_hit_ids:
        my_dict2[key] = reco_hit_id
        reco_hit_id -= 1
    df_reco_particle_id2 = pd.DataFrame({'global_hit_id': list(my_dict2.keys()), 'reco_hit_id': list(my_dict2.values())})

    # df_reco_particle_id = df_reco_particle_id1.append(df_reco_particle_id2, ignore_index=True)
    df_reco_particle_id = pd.concat([df_reco_particle_id1, df_reco_particle_id2], ignore_index=True) # append depreciated in some pandas versions
    #df_reco_particle_id
    
    # Fourth, merge df with 'reco_hit_id' into hits df. Reorder rows by 'reco_hit_id'
    df_hits = pd.merge(df_hits, df_reco_particle_id, on='global_hit_id')
    df_hits = df_hits.sort_values(by = 'reco_hit_id') 
    
    return df_hits
# ----------------------------------------
def add_reco_hit_id_column_endtrack_seed_fun(vertex):
    global df_hits
    
    # First, we compute the distances between all hits. For this method, we have to add the vertex hit, at (0,0,0)
    x_array = df_hits['x'].to_numpy()
    y_array = df_hits['y'].to_numpy()
    z_array = df_hits['z'].to_numpy()
    
    x_array = np.append(x_array,vertex[0])
    y_array = np.append(y_array,vertex[1])
    z_array = np.append(z_array,vertex[2])
    nhits = len(x_array)

    matrix = np.zeros([nhits,nhits])
    for j in np.arange(nhits-1):
        for k in np.arange(j+1,nhits):
            dx = x_array[j] - x_array[k]
            dy = y_array[j] - y_array[k]
            dz = z_array[j] - z_array[k]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            matrix[j][k] = distance
            matrix[k][j] = distance
            
    # Second, apply the travelling salesman problem (TSP) algorithm starting from the most energetic hit
    # in the event, taken to be the extreme of a track. We connect the hits using the shortest distance
    emax_id  = df_hits['energy'].idxmax()
    
    checked_hit_ids = []
    visit_id = emax_id
    while len(checked_hit_ids) != nhits:
        #print(len(checked_hits))
        cond = True
        while cond:
            # This is the furthest hit from the hit being visited
            furthest_hit_id = np.argmax(matrix[visit_id])
            matrix[visit_id][visit_id]=1.e6 # This is just to ensure that closest_hit_id in next line can be computed using argmin
            closest_hit_id = np.argmin(matrix[visit_id]) #next hit to visit is the clostest to the one being visited
            if closest_hit_id in checked_hit_ids:
                cond = True
                matrix[visit_id][closest_hit_id]=matrix[visit_id][furthest_hit_id]+1 # This is to ensure that this hit is not visited again
            else:
                cond = False
                visited_id = visit_id
                checked_hit_ids.append(visited_id)
                visit_id = closest_hit_id
                
    # Third, we find the vertex hit position within the checked_hits array. The vertex hit has ID equal to nhits-1
    # Then, all hits from the vertex hit (vertex_hit_index) to the most energetic hit are assigned positive and sequential
    # reco_hit_id values, signaling e1, following the ordering in checked_hit_ids. 
    # All other hits are assigned negative and sequential reco_hit_id values 
    # Use a dictionary with keys equal to checked_hit_ids, values equal to reco_hit_ids, for that
    vertex_hit_index = checked_hit_ids.index(nhits-1)
    keys = checked_hit_ids
    values = np.zeros(nhits, dtype='i')

    reco_hit_id = 1
    for i in np.arange(vertex_hit_index-1,-1,-1):
        values[i] = reco_hit_id
        reco_hit_id += 1

    reco_hit_id = -1
    for i in np.arange(vertex_hit_index+1,nhits):
        values[i] = reco_hit_id
        reco_hit_id -= 1

    df_reco_particle_id = pd.DataFrame({'global_hit_id': keys, 'reco_hit_id': values})
    
    # Fourth, merge df with 'reco_hit_id' into hits df. Reorder rows by 'reco_hit_id'. 
    # Note that df_hits has no vertex hit, df_reco_particle_id has it. When merging, the vertex hit is ignored, as we want
    df_hits = pd.merge(df_hits, df_reco_particle_id, on='global_hit_id')
    df_hits = df_hits.sort_values(by = 'reco_hit_id') 
    
    return df_hits
# ----------------------------------------
def reco_quantities_fun(vertex):
    global df_hits
    T1_reco = -999
    costheta_reco = -999
    
    # T1_reco
    # First, we separate hits based on reco_hit_id sign
    e1reco_hit_mask = df_hits.reco_hit_id > 0
    df_e1reco_hits = df_hits[e1reco_hit_mask]
    df_e1reco_hits = df_e1reco_hits.sort_values(by = 'reco_hit_id', ascending = True)
    if (len(df_e1reco_hits) == 0):
        print('No e1reco hits, cannot reconstruct!')
        return T1_reco, costheta_reco

    e2reco_hit_mask = df_hits.reco_hit_id < 0
    df_e2reco_hits = df_hits[e2reco_hit_mask]
    df_e2reco_hits = df_e2reco_hits.sort_values(by = 'reco_hit_id', ascending = False)
    if (len(df_e2reco_hits) == 0):
        print('No e2reco hits, cannot reconstruct!')
        return T1_reco, costheta_reco

    
    # Second, we compute T1_reco and T2_reco
    T1_reco = df_e1reco_hits.energy.sum()
    T2_reco = df_e2reco_hits.energy.sum()
    T1_reco, T2_reco = reorder_electrons_fun(T1_reco,T2_reco)
    #print(T1_reco,T2_reco,T1_reco+T2_reco)
    
    # cotheta_reco
    if (len(df_e1reco_hits) == 0 or len(df_e2reco_hits) == 0):
        return -999,-999        

    costheta_reco, t1_delta = CalcTrackAngle(df_e1reco_hits, df_e2reco_hits, vertex)
    
    return T1_reco+t1_delta, costheta_reco
# ----------------------------------------
def convert_np_types(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert array to list
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
# ----------------------------------------
# ----------------------------------------
dict_out = {}
for index, evt_id in enumerate(event_ids):

    print("On Event:", index, evt_id)
    event_dict = {}

    # Only particles and hits from event_id = evt_id, to deal one event at the time
    df_particles = df_particles_allevts[df_particles_allevts.event_id == evt_id]
    df_hits = df_hits_allevts[df_hits_allevts.event_id == evt_id]

    # Step 1: Generator-level $T_1$ and $\cos\theta$
    T1_gen, costheta_gen = gen_quantities_fun()

    # Step 2: Truth-level $T_1$ and $\cos\theta$
    if ("nexus" not in infile):
        T1_true, costheta_true = 0,0
    else:
        df_hits = add_global_hit_id_column_fun()
        df_hits = add_distance_column_fun(df_hits, get_vertex(df_particles, evt_id))
        df_hits = add_ancestor_id_column_fun()
        vertex0 = get_vertex(df_particles, evt_id)
        T1_true, costheta_true = true_quantities_fun(vertex0)
        
    event_dict["T1_gen"] = T1_gen
    event_dict["T1_true"] = T1_true
    event_dict["costheta_gen"] = costheta_gen
    event_dict["costheta_true"] = costheta_true

    df_hits_full = df_hits_allevts[df_hits_allevts.event_id == evt_id]

    # Iterate over different vertex smear distances
    for smear in [0,1,2,3,4,5]:
        df_hits = df_hits_full.copy()
        vertex = get_vertex(df_particles, evt_id)
        
        if smear != 0:
            vertex = add_smear(df_particles, evt_id, smear)

        df_hits = add_distance_column_fun(df_hits, vertex)

        vertex_index = df_hits['distance'].idxmin()
        
        E_vertex = df_hits.loc[vertex_index, 'energy']

        # Drop the closest node to the vertex only if nexus is not in the filename
        if ("nexus" not in infile):
            df_hits = df_hits.drop(vertex_index) # Remove it
            df_hits = df_hits.reset_index(drop=True)

        nhits = len(df_hits)
        df_hits = add_global_hit_id_column_fun()

        # Step 3: Reco-level $T_1$ and $\cos\theta$
        if (reco_seed_method == 'vertex_seed'):
            df_hits = add_reco_hit_id_column_vertex_seed_fun(vertex)
        elif (reco_seed_method == 'endtrack_seed'):
            df_hits = add_reco_hit_id_column_endtrack_seed_fun(vertex)
        else:
            raise SystemExit('Unrecognized reco seed method, stop executing!')
        T1_reco, costheta_reco = reco_quantities_fun(vertex)

        # Add split vertex energy to T1
        if ("nexus" not in infile):
            T1_reco = T1_reco + E_vertex/2.0
        
        if smear != 0: 
            event_dict[f"costheta_{smear}mm"] = costheta_reco
            event_dict[f"T1_{smear}mm"] = T1_reco
            event_dict[f"vertex_{smear}mm"] = vertex
        else:
            event_dict["costheta_reco"] = costheta_reco
            event_dict["T1_reco"] = T1_reco
            event_dict["vertex_true"] = vertex

        match = axis_limits_df[axis_limits_df["event_id"] == evt_id]
        if not match.empty:
            event_dict["axis_limits"] = json.loads(match.iloc[0]["axis_limits"])
        else:
            event_dict["axis_limits"] = None

    dict_out[evt_id] = event_dict

file_out = infile[:-3]
file_out = f"{file_out}_data.json"

with open(file_out, "w") as f:
    for event_id, event_data in dict_out.items():
            # Create a flat dict including file and event info
        line_data = {
            "file": infile,
            "event_id": event_id,
            **event_data
        }
        f.write(json.dumps(line_data, default=convert_np_types) + "\n")


