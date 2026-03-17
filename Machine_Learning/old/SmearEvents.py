
# Script to read in the nexus files and smear the hits along the track.
# Re-bin the resultant track
import sys
import numpy  as np
import pandas as pd
from collections import Counter
import time

pd.options.mode.chained_assignment = None  # Disable the warning

# USAGE:
# python3 SmearEvents.py <name of nexus input file name (remove .h5 extension)> <Scale Factor> <CO2Percentage> <binsize> <pressure> <JOBID>
# e.g. python3 SmearEvents.py ATPC_0nubb_1bar_Efilt 1 0.1 20 1.0 1 

# Record the start time
start_time = time.time()

# Load in the hits
print("Loading hits")
print("Filename: ", sys.argv[1])
hits = pd.read_hdf(sys.argv[1], 'MC/hits')
parts = pd.read_hdf(sys.argv[1], 'MC/particles')
print("Finished loading hits")

# init the RNG
rng = np.random.default_rng()

percentage =  float(sys.argv[3])

pressure =  float(sys.argv[5])

# Diffusion values desired

# Scaling by sqrt pressure is applied to convert to whatever pressure at the same E/P
if (percentage == 0.0):
    DL = 0.9 / np.sqrt(pressure) # mm / sqrt(cm)
    DT = 3.5 / np.sqrt(pressure) # mm / sqrt(cm)
elif (percentage == 0.05): # !!HELIUM 10%!!
    DL = 0.75 / np.sqrt(pressure) # mm / sqrt(cm)
    DT = 1.60 / np.sqrt(pressure) # mm / sqrt(cm)
elif (percentage == 0.1):
    DL = 1.037 / np.sqrt(pressure) # mm / sqrt(cm)
    DT = 0.818 / np.sqrt(pressure) # mm / sqrt(cm)
elif (percentage == 0.25):
    DL = 0.627 / np.sqrt(pressure) # mm / sqrt(cm)
    DT = 0.463 / np.sqrt(pressure) # mm / sqrt(cm)
elif (percentage == 5):
    DL = 0.314 / np.sqrt(pressure) # mm / sqrt(cm)
    DT = 0.300 / np.sqrt(pressure) # mm / sqrt(cm)
else:
    print("Error CO2 percentage not defined at 60 V/cm/bar field")


# This is the scaling amount of diffusion
# scaling factor is in number of sigma
diff_scaling = float(sys.argv[2])
binsize = float(sys.argv[4])
jobid = int(sys.argv[6])

print("Scaling Factor: ", diff_scaling)
print("CO2 Percentage: ", percentage)
print("Pressure: ", pressure)
print("DL: ", DL, "mm/sqrt(cm)")
print("DT: ", DT, "mm/sqrt(cm)")
print("binsize is: ", binsize, "mm")

# Calculate the detector half-length

density = 5.987*pressure
M = 1000/0.9
det_size = 1000*np.cbrt((4 * M) / (np.pi * density))/2.0
# det_size = int(np.cbrt(6000**3/pressure)/2.0) 

# Create the bins ---- 
xbw=binsize
xmin=-det_size - binsize/2 
xmax=det_size + binsize/2

ybw=binsize
ymin=-det_size - binsize/2 
ymax=det_size + binsize/2


# This shifts the z pos of the events so 0 is at anode
# can set this to zero
z_shift = det_size
# z_shift = 0

zbw=binsize
zmin=-det_size + z_shift - binsize/2
zmax=det_size + z_shift + binsize/2 

print("Detector size is:", det_size*2)


# bins for x, y, z
xbins = np.arange(xmin, xmax+xbw, xbw)
ybins = np.arange(ymin, ymax+ybw, ybw)
zbins = np.arange(zmin, zmax+zbw, zbw)

# center bins for x, y, z
xbin_c = xbins[:-1] + xbw / 2
ybin_c = ybins[:-1] + ybw / 2
zbin_c = zbins[:-1] + zbw / 2


# Mean energy per e-. This splits up each G4 into E_hit/E_mean electrons
E_mean = 24.8e-6 # [eV]

df_smear = []

# Define a function to smear the geant4 electrons uniformly between the steps
# Each electron is sampled uniformly towards the previous hit
# The ends of the track are sampled in the backward direction only 
def generate_random(row):
    r0 = np.array([row['x'], row['y'], row['z']])
    r1 = np.array([row['x'] - row['dx'], row['y'] - row['dy'], row['z'] - row['dz']])  # Backward delta

    # Uniformly move backward from the hit by its step size
    random_number = rng.uniform(0, 1)
    new_r = r0 + random_number * (r1 - r0)

    # Apply diffusion if scaling is nonzero
    if diff_scaling != 0.0:
        z = new_r[2]  # mm
        
        # If we smeared to something negative then apply no diffusion
        # This should only be an edge effted
        if (z < 0):
            print("Negative z value! This should be uncommon...", z)
            z = 0
        
        sigma_DL = diff_scaling * DL * np.sqrt(z / 10.0)  # mm  
        sigma_DT = diff_scaling * DT * np.sqrt(z / 10.0)  # mm  

        mean = new_r
        cov = np.diag([sigma_DT**2, sigma_DT**2, sigma_DL**2])  # 3D covariance matrix

        new_r = rng.multivariate_normal(mean, cov)

    return pd.Series(new_r, index=['x_smear', 'y_smear', 'z_smear'])


# Print the number of events:
print("Number of events to process: ", len(hits.event_id.unique()))
min_event_id = min( hits.event_id.unique())

# ---------------
# Main event Loop
# ---------------
for index, e in enumerate(hits.event_id.unique()):
    print("On Event:", e - min_event_id)

    # Record the end time
    end_time = time.time()

    # Calculate and print the runtime
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.4f} seconds")

    # Select the event
    event = hits[hits.event_id == e]
    event_part = parts[parts.event_id == e]
    event["n"] = round(event["energy"]/E_mean)
    # Shift z-values so 0 is at the anode
    event.z = event.z+z_shift

    # Loop over the particles and get the differences between steps ------
    particles = event.particle_id.unique()

    smear_df = []

    for idx, p in enumerate(particles):

        # Get hits for particle i in the event
        temp_part_hits = event[event.particle_id == p]
        temp_part = event_part[event_part.particle_id == p]
        particle_name = temp_part.particle_name.iloc[0]

        nrows = len(temp_part_hits)

        # This dataframe contains the difference in distance between hits
        diff_df = temp_part_hits[['x', 'y', 'z']].diff()

        # Set the dist for the first hit as the difference to the inital position
        diff_df.iloc[0, diff_df.columns.get_loc('x')] = np.float32(temp_part_hits.iloc[0].x - temp_part.initial_x.iloc[0])
        diff_df.iloc[0, diff_df.columns.get_loc('y')] = np.float32(temp_part_hits.iloc[0].y - temp_part.initial_y.iloc[0])
        diff_df.iloc[0, diff_df.columns.get_loc('z')] = np.float32(temp_part_hits.iloc[0].z - (temp_part.initial_z.iloc[0] + z_shift))


        # Name the columns by their deltas
        diff_df = diff_df.rename(columns={'x': 'dx', 'y': 'dy', 'z': 'dz'})

        # We dont want to smear over the gamma steps
        # Only their daughter electrons
        if (particle_name == "gamma"):
            diff_df["dx"] = 0*diff_df["dx"]
            diff_df["dy"] = 0*diff_df["dy"]
            diff_df["dz"] = 0*diff_df["dz"]
        
        smear_df.append(diff_df)

    # Concatenate DataFrames along rows (axis=0)
    smear_df = pd.concat(smear_df)

    # Now merge to the main df
    event = pd.merge(event, smear_df, left_index=True, right_index=True, how='inner')

    # Create a new DataFrame with duplicated rows, so we can smear each electron by diffusion
    electrons = pd.DataFrame(np.repeat(event[["event_id",'x', 'y', 'z', 'dx', 'dy', 'dz']].values, event['n'], axis=0), columns=["event_id",'x', 'y', 'z', 'dx', 'dy', 'dz'])

    # Reset the index of the new DataFrame if needed
    electrons = electrons.reset_index(drop=True)

    # Now apply some smearing to each of the electrons
    # Apply the function to create new columns
    new_columns     = electrons.apply(generate_random, axis=1)
    electrons_smear = pd.concat([electrons, new_columns], axis=1)
    electrons_smear["energy"] = E_mean # MeV

    # We need to set this to make sure we keep the information about the unbinned positions in the weighting
    electrons_smear['x'] = electrons_smear['x_smear']
    electrons_smear['y'] = electrons_smear['y_smear']
    electrons_smear['z'] = electrons_smear['z_smear']

    # Now lets bin the data
    electrons_smear['x_smear'] = pd.cut(x=electrons_smear['x_smear'], bins=xbins,labels=xbin_c, include_lowest=True)
    electrons_smear['y_smear'] = pd.cut(x=electrons_smear['y_smear'], bins=ybins,labels=ybin_c, include_lowest=True)
    electrons_smear['z_smear'] = pd.cut(x=electrons_smear['z_smear'], bins=zbins,labels=zbin_c, include_lowest=True)

    # Loop over the rows in the dataframe and sum the energies of all electrons in a bin. 
    # Also change the bin center to use the mean x,y,z position
    x_mean_arr = []
    y_mean_arr = []
    z_mean_arr = []
    energy_mean_arr = []
    x_mean_arr_temp = np.array([])
    y_mean_arr_temp = np.array([])
    z_mean_arr_temp = np.array([])
    summed_energy = 0
    event_id = 0
    energy_temp =electrons_smear.energy.sum()

    counter = 0

    # Sort so all the bin labels are next to one another
    electrons_smear = electrons_smear.sort_values(by=['x_smear', 'y_smear', 'z_smear'])

    # Loop over all bins and aggregate to get total energy in each bin and their
    # mean x,y,z position
    for index, row in electrons_smear.iterrows():

        # First row 
        if (counter == 0):
            temp_x = row["x_smear"]
            temp_y = row["y_smear"]
            temp_z = row["z_smear"]
            summed_energy +=row["energy"]
            event_id = row["event_id"]
            x_mean_arr_temp = np.append(x_mean_arr_temp, row["x"])
            y_mean_arr_temp = np.append(y_mean_arr_temp, row["y"])
            z_mean_arr_temp = np.append(z_mean_arr_temp, row["z"])
            counter+=1
            continue

        # Same bin
        if (row["x_smear"] == temp_x and row["y_smear"] == temp_y and row["z_smear"] == temp_z):
            x_mean_arr_temp = np.append(x_mean_arr_temp, row["x"])
            y_mean_arr_temp = np.append(y_mean_arr_temp, row["y"])
            z_mean_arr_temp = np.append(z_mean_arr_temp, row["z"])
            summed_energy +=row["energy"]

            # Final row
            if index == electrons_smear.index[-1]:
                if (summed_energy != 0): 
                    x_mean_arr = np.append(x_mean_arr,np.mean(x_mean_arr_temp))
                    y_mean_arr = np.append(y_mean_arr,np.mean(y_mean_arr_temp))
                    z_mean_arr = np.append(z_mean_arr,np.mean(z_mean_arr_temp))
                    energy_mean_arr.append(summed_energy)

        # Aggregate and store for next 
        else:
            if (summed_energy != 0): 
                x_mean_arr = np.append(x_mean_arr,np.mean(x_mean_arr_temp))
                y_mean_arr = np.append(y_mean_arr,np.mean(y_mean_arr_temp))
                z_mean_arr = np.append(z_mean_arr,np.mean(z_mean_arr_temp))
                energy_mean_arr.append(summed_energy)
            
            temp_x = row["x_smear"]
            temp_y = row["y_smear"]
            temp_z = row["z_smear"]
            summed_energy = 0
            x_mean_arr_temp = np.array([])
            y_mean_arr_temp = np.array([])
            z_mean_arr_temp = np.array([])
            
            x_mean_arr_temp = np.append(x_mean_arr_temp, row["x"])
            y_mean_arr_temp = np.append(y_mean_arr_temp, row["y"])
            z_mean_arr_temp = np.append(z_mean_arr_temp, row["z"])
            summed_energy +=row["energy"]

            # Final row
            if index == electrons_smear.index[-1]:
                if (summed_energy != 0): 
                    x_mean_arr = np.append(x_mean_arr,np.mean(x_mean_arr_temp))
                    y_mean_arr = np.append(y_mean_arr,np.mean(y_mean_arr_temp))
                    z_mean_arr = np.append(z_mean_arr,np.mean(z_mean_arr_temp))
                    energy_mean_arr.append(summed_energy)

        counter+=1

    events = np.ones_like(energy_mean_arr)*event_id

    # Make the dataframe again
    electrons_smear = pd.DataFrame({  "event_id" : events, "x" : x_mean_arr,  "y" : y_mean_arr,  "z" : z_mean_arr,  "energy" : energy_mean_arr  }) 

    rounded_energy_temp = round(energy_temp, 3)
    rounded_energy_sum = round(sum(energy_mean_arr), 3)

    if rounded_energy_temp != rounded_energy_sum:
        print(f"Error! Mismatch in the summed energy: {rounded_energy_temp} != {rounded_energy_sum}")

    # File writing
    electrons_smear = electrons_smear.sort_values(by=['event_id', 'z', 'x', 'y'])

    electrons_smear['event_id'] = electrons_smear['event_id'].astype(int)
    electrons_smear['z'] = electrons_smear['z'].astype('float32')
    electrons_smear['x'] = electrons_smear['x'].astype('float32')
    electrons_smear['y'] = electrons_smear['y'].astype('float32')
    # df['energy'] = df['energy']*1e6
    electrons_smear['energy'] = electrons_smear['energy'].astype('float32')

    df_smear.append(electrons_smear)


df_smear_merge = pd.concat(df_smear, ignore_index=True)

outfile = sys.argv[1] + "_" + str(percentage) + "percent_smear_" + str(jobid) + ".h5"

if (diff_scaling == 0.0):
    outfile = sys.argv[1] + "_smear_" + str(jobid) + ".h5"

print("Saving events to file: ", outfile)
with pd.HDFStore(outfile, mode='w', complevel=5, complib='zlib') as store:
    # Write each DataFrame to the file with a unique key
    store.put('MC/particles', parts, format='table')
    store.put('MC/hits', df_smear_merge, format='table')

# Record the end time
end_time = time.time()

# Calculate and print the runtime
runtime = end_time - start_time
print(f"Runtime: {runtime:.4f} seconds")

