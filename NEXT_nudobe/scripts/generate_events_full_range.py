import numpy as np
from tqdm import tqdm

total_rows = 1_000_000
total_energy = 2.458
file_path = "/home/rei/NEXT/NEXT_nudobe/config/events_full_range_2.txt"

print("Sampling angles...")
cos_theta1 = np.random.uniform(-1, 1, total_rows)
cos_theta2 = np.random.uniform(-1, 1, total_rows)


e1_theta = np.arccos(cos_theta1)
e1_phi = np.random.uniform(0, 2*np.pi, total_rows)
e2_theta = np.arccos(cos_theta2)
e2_phi = np.random.uniform(0, 2*np.pi, total_rows)

print("Getting dir vectors...")
x1 = np.sin(e1_theta) * np.cos(e1_phi)
y1 = np.sin(e1_theta) * np.sin(e1_phi)
z1 = np.cos(e1_theta)
mag1 = np.sqrt(x1**2 + y1**2 + z1**2)
x1, y1, z1 = x1/mag1, y1/mag1, z1/mag1

x2 = np.sin(e2_theta) * np.cos(e2_phi)
y2 = np.sin(e2_theta) * np.sin(e2_phi)
z2 = np.cos(e2_theta)
mag2 = np.sqrt(x2**2 + y2**2 + z2**2)
x2, y2, z2 = x2/mag2, y2/mag2, z2/mag2

print("Getting energies...")
values = np.linspace(1.7, 2.1, 1000)
probs = np.exp(-3 * (values - 1.7)) + 0.05 * np.exp(-3 * (2 - values))
probs /= probs.sum()
e1 = np.round(np.random.choice(values, size=total_rows, p=probs), 5)
e2 = np.round(total_energy - e1, 5)

print("Writing to file...")
header = "x1_dir,y1_dir,z1_dir,e1,x2_dir,y2_dir,z2_dir,e2\n"

with open(file_path, "w") as f:
    f.write(header)
    for i in tqdm(range(total_rows)):
        f.write(f"{x1[i]:.5f},{y1[i]:.5f},{z1[i]:.5f},{e1[i]:.5f},"
                f"{x2[i]:.5f},{y2[i]:.5f},{z2[i]:.5f},{e2[i]:.5f}\n")