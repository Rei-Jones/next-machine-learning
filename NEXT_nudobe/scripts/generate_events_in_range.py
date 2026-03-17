# imports

import numpy as np
import os
import csv
import random as rd
from tqdm import tqdm

# csv file has the form: x1_dir, y1_dir, z1_dir, e1, x2_dir, y2_dir, z2_dir, e2

# generating random data, 1 million rows

def generate_random(min, max, total):
    count = 0
    list = []
    for i in tqdm(range(total)):
        element = rd.uniform(min, max)
        element = round(element, 5)
        list.append(element)
    return list

# getting dir for e1 and e2

total_rows = 1000000
angle_min = np.pi / 4
angle_max = (3 * np.pi) / 4

dir_max = 2 * np.pi

print("getting theta...")
theta_list = generate_random(angle_min, angle_max, total_rows)
print("getting phi")
phi_list = generate_random(angle_min, angle_max, total_rows)
print("getting 1st electron theta")
e1_theta_list = generate_random(0, dir_max, total_rows)
print("getting 1st electron phi")
e1_phi_list = generate_random(0, dir_max, total_rows)

e2_theta_list = []
e2_phi_list = []

print("making electron 2 data...")
for i in tqdm(range(total_rows)):
    e2_theta = e1_theta_list[i] + theta_list[i]
    e2_phi = e1_phi_list[i] + phi_list[i]
    e2_theta_list.append(e2_theta)
    e2_phi_list.append(e2_phi)
    

x1_dir_list = []
y1_dir_list = []
z1_dir_list = []
x2_dir_list = []
y2_dir_list = []
z2_dir_list = []

print("geting direction vectors...")
for i in tqdm(range(total_rows)):
    x1_dir = np.sin(e1_phi_list[i]) * np.cos(e1_theta_list[i])
    y1_dir = np.sin(e1_phi_list[i]) * np.sin(e1_theta_list[i])
    z1_dir = np.cos(e1_phi_list[i])
    mag1 = np.sqrt((x1_dir ** 2) + (y1_dir ** 2) + (z1_dir ** 2))

    x2_dir = np.sin(e2_phi_list[i]) * np.cos(e2_theta_list[i])
    y2_dir = np.sin(e2_phi_list[i]) * np.sin(e2_theta_list[i])
    z2_dir = np.cos(e2_phi_list[i])
    mag2 = np.sqrt((x2_dir ** 2) + (y2_dir ** 2) + (z2_dir ** 2))

    x1_dir_list.append(round((x1_dir/mag1), 5))
    y1_dir_list.append(round((y1_dir/mag1), 5))
    z1_dir_list.append(round((z1_dir/mag1), 5))
    x2_dir_list.append(round((x2_dir/mag2), 5))
    y2_dir_list.append(round((y2_dir/mag2), 5))
    z2_dir_list.append(round((z2_dir/mag2), 5))

#getting energy

total_energy = 3
print("getting e1...")
e1_list = generate_random(1.2, 1.8, total_rows)
e2_list = []

print("getting e2...")
for i in tqdm(range(total_rows)):
    e2 = 3 - e1_list[i]
    e2_list.append(e2)

file_path = "/home/rei/NEXT/NEXT_nudobe/config/events_45_135.txt"

with open(file_path, "a") as f:
    f.write("x1_dir,y1_dir,z1_dir,e1,x2_dir,y2_dir,z2_dir,e2\n")

    print("writing to file...")
    for i in tqdm(range(total_rows)):
        f.write(f"{x1_dir_list[i]},{y1_dir_list[i]},{z1_dir_list[i]},{e1_list[i]},{x2_dir_list[i]},{y2_dir_list[i]},{z2_dir_list[i]},{e2_list[i]}\n")
        
