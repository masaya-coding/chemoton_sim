
#this is the script for plotting

import csv
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
from collections import Counter


steps = []
alive = []
X = []
Z = []
deaths = []
deaths_lowN = []
deaths_hunger = []
deaths_shrink_other = []


with open("run_env.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(int(row["step"]))
        alive.append(int(row["alive"]))
        X.append(float(row["X_world"]))
        Z.append(float(row["Z_world"]))
        deaths.append(int(row["deaths_this_step"]))
        deaths_lowN.append(int(row["deaths_lowN"]))
        deaths_hunger.append(int(row["deaths_hunger"]))
        deaths_shrink_other.append(int(row["deaths_shrink_other"]))


print("Total deaths:", sum(deaths))
print("Total N<20 deaths:", sum(deaths_lowN))
print("Total hunger deaths:", sum(deaths_hunger))
print("Total other shrink deaths:", sum(deaths_shrink_other))



plt.figure()
plt.plot(steps, alive)
plt.xlabel("Step")
plt.ylabel("Alive protocells")
plt.title("Alive protocells over time")
plt.show()

plt.figure()
plt.plot(steps, X, label="X_world")
plt.plot(steps, Z, label="Z_world")
plt.xlabel("Step")
plt.ylabel("Food")
plt.legend()
plt.title("Food over time")
plt.show()

plt.figure()
plt.bar(steps, deaths)
plt.xlabel("Step")
plt.ylabel("death this step")
plt.title("death per step over time")
plt.show()



# ===== 2D grid world plot (cell positions at last step) =====

steps_pos = []
xs = []
ys = []

with open("run_positions.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps_pos.append(int(row["step"]))
        xs.append(int(row["x"]))
        ys.append(int(row["y"]))

last_step = max(steps_pos)

xs_last = [x for x, s in zip(xs, steps_pos) if s == last_step]
ys_last = [y for y, s in zip(ys, steps_pos) if s == last_step]

#blob stuff
counts = Counter(zip(xs_last, ys_last))
x_unique = [pos[0] for pos in counts.keys()]
y_unique = [pos[1] for pos in counts.keys()]
cell_counts = np.array(list(counts.values()))
sizes = 20 * cell_counts  # linear scaling



grid_X = np.load("final_X.npy")
vmax = np.max(grid_X)


plt.figure(figsize=(6, 6))

im = plt.imshow(grid_X, origin="lower", cmap="Greens", vmax = 1.1)
plt.scatter(x_unique, y_unique, s=sizes, c="grey", edgecolors="black", linewidths=0.3)

plt.colorbar(im, label="Food X amount")
plt.title("Final spatial distribution of food (X)")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.show()
