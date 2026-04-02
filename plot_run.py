
#this is the script for plotting

import csv
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter


steps = []
alive = []
X = []
Z = []
deaths = []
deaths_lowN = []
deaths_hunger = []
deaths_shrink_other = []

#---------------------------------------------------------- map colours

# X: green → white → grey

cmap_X = LinearSegmentedColormap.from_list(
    "X_map",
    [(0.0, "grey"), (1.0/1.1, "white"), (1.0, "green")]
)



# Z: brown → white → grey

cmap_Z = LinearSegmentedColormap.from_list(
    "Z_map",
    [(0.0, "grey"), (1.0/2.0, "white"), (1.0, "saddlebrown")]
)

#----------------------------------------------------------


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


#-----------------

N_history = np.load("N_history.npy", allow_pickle=True)
MetB_history = np.load("MetB_history.npy", allow_pickle=True)

N_values = N_history[-1]
MetB_values = MetB_history[-1]

#----------------------------------------

from evo_2_3_copy import HAS_MET_B_IDX


def plot_final_N_distribution(N_values, MetB_values, out: pathlib.Path | None = None):
    """Plot histogram of final N values in population with enhanced visuals"""
    n_values = N_values
    has_met_B = MetB_values

    if len(n_values) == 0:
        print("No viable individuals to plot")
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot N distribution
    bins = np.arange(min(n_values) - 0.5, max(n_values) + 1.5, 1)
    ax1.hist(n_values, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=20, color='r', linestyle='--', linewidth=2, label='Survival threshold')
    ax1.axvline(x=40, color='g', linestyle='--', linewidth=2, label='Met B threshold')
    ax1.axvline(x=np.mean(n_values), color='k', linestyle='-', linewidth=2, label=f'Mean: {np.mean(n_values):.2f}')
    ax1.set_xlabel('Template Length (N)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Template Lengths in Final Population', fontsize=14)
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot relationship between N and metabolism B
    N_with_B = [n for n,b in zip(n_values,has_met_B) if b]
    N_without_B = [n for n,b in zip(n_values,has_met_B) if not b]
    
    ax2.scatter(N_without_B, [0] * len(N_without_B), c='blue', alpha=0.6, s=80, label='Without Met B')
    ax2.scatter(N_with_B, [1] * len(N_with_B), c='green', alpha=0.6, s=80, label='With Met B')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No', 'Yes'])
    ax2.set_ylabel('Has Metabolism B', fontsize=12)
    ax2.set_xlabel('Template Length (N)', fontsize=12)
    ax2.set_title('Metabolism B Acquisition vs Template Length', fontsize=14)
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if requested
    if out:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "final_N_distribution.png", dpi=160, bbox_inches='tight')
    
    plt.show()

plot_final_N_distribution(N_values, MetB_values)

#----------------------------------------


def plot_evolution_stats_env(steps, alive, N_history, MetB_history):

    generations = steps

    mean_N = []
    min_N = []
    max_N = []
    percent_met_B = []

    for N_vals, metB_vals in zip(N_history, MetB_history):

        if len(N_vals) == 0:
            mean_N.append(0)
            min_N.append(0)
            max_N.append(0)
            percent_met_B.append(0)
            continue

        mean_N.append(np.mean(N_vals))
        min_N.append(np.min(N_vals))
        max_N.append(np.max(N_vals))

        percent_met_B.append(100 * np.sum(metB_vals) / len(metB_vals))

    viable_counts = alive
    pop_sizes = alive

    plt.style.use('seaborn-v0_8-whitegrid')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,10), sharex=True)

    # --- template length evolution ---
    ax1.plot(generations, mean_N, 'b-', linewidth=2, label='Mean N')
    ax1.fill_between(generations, min_N, max_N, color='b', alpha=0.2, label='Range')
    ax1.axhline(y=20, color='r', linestyle='--', linewidth=2, label='Survival threshold')
    ax1.axhline(y=40, color='g', linestyle='--', linewidth=2, label='Met B threshold')
    ax1.set_ylabel('Template Length (N)')
    ax1.set_title('Evolution of Template Length over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- metabolism B prevalence ---
    ax2.plot(generations, percent_met_B, 'g-', linewidth=2)
    ax2.set_ylabel('% with Metabolism B')
    ax2.set_title('Prevalence of Metabolism B')
    ax2.grid(True, alpha=0.3)

    # --- population viability ---
    ax3.plot(generations, viable_counts, 'purple', linewidth=2, label='Viable individuals')
    ax3.plot(generations, pop_sizes, 'gray', linestyle='--', linewidth=1.5, label='Total population')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Population Count')
    ax3.set_title('Population Viability')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evolution_stats.png", dpi=150)

    plt.show()

N_history = np.load("N_history.npy", allow_pickle=True)
MetB_history = np.load("MetB_history.npy", allow_pickle=True)

plot_evolution_stats_env(
    steps,
    alive,
    N_history,
    MetB_history
)

#----------------------------------------

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# --- 1st plot (unchanged logic) ---
ax1.plot(steps, alive)
ax1.set_xlabel("Step")
ax1.set_ylabel("Alive protocells")
ax1.set_title("Alive protocells over time")

# --- 2nd plot ---
ax2.plot(steps, X, label="X_world")
ax2.plot(steps, Z, label="Z_world")
ax2.set_xlabel("Step")
ax2.set_ylabel("Food")
ax2.legend()
ax2.set_title("Food over time")

# --- 3rd plot ---
ax3.bar(steps, deaths)
ax3.set_xlabel("Step")
ax3.set_ylabel("death this step")
ax3.set_title("death per step over time")

plt.tight_layout()
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



#-------------------------------------------------------------------

def render_timelapse(frame_dir="run_frames", out_dir="frames_png"):
    os.makedirs(out_dir, exist_ok=True)

    frame_files = sorted(
        f for f in os.listdir(frame_dir) if f.endswith(".npz")
    )

    for fname  in frame_files:
        data = np.load(os.path.join(frame_dir, fname), allow_pickle=True)        

        X = data["X"]
        Z = data["Z"]
        pos = data["pos"]
        metB = data["metB"]
        paths = data["paths"]
        daughter = data["daughter"]
        dead = data["dead"]

        fig, ax = plt.subplot(figsize=(6,6))
        imX = ax.imshow(X, vmin=0.0, vmax=1.1, cmap=cmap_X, origin="lower")
        imZ = ax.imshow(Z, vmin=0.0, vmax=2.0, cmap=cmap_Z, origin="lower", alpha=0.5)

        cbarX = plt.colorbar(imX, ax=ax, fraction=0.03, pad=0.04)
        cbarX.set_label("Food X")

        cbarZ = plt.colorbar(imZ, ax=ax, fraction=0.03, pad=0.04, location="left")
        cbarZ.set_label("Food Z")


        # draw grid cell borders---

        ax.set_xticks(np.arange(0, X.shape[1] + 1, 1))
        ax.set_yticks(np.arange(0, X.shape[0] + 1, 1))

        ax.grid(which="both", color="black", linewidth= 5)

        # remove tick labels but keep grid
        ax.set_xticklabels([])
        ax.set_yticklabels([])


        if len(pos) > 0:
            pos_int = pos.astype(int)

    # ----- parent cells -----
            parent_pos = pos_int[daughter == 0]
            if len(parent_pos) > 0:
                parent_counts = Counter(map(tuple, parent_pos))
                xy_p = np.array(list(parent_counts.keys()))
                sizes_p = 20 * np.array(list(parent_counts.values()))

                ax.scatter(
                    xy_p[:,0],
                    xy_p[:,1],
                    c="black",
                    s=sizes_p,
                )

    # ----- daughter cells -----
            daughter_pos = pos_int[daughter == 1]
            if len(daughter_pos) > 0:
                daughter_counts = Counter(map(tuple, daughter_pos))
                xy_d = np.array(list(daughter_counts.keys()))
                sizes_d = 20 * np.array(list(daughter_counts.values()))

                ax.scatter(
                    xy_d[:,0],
                    xy_d[:,1],
                    c="blue",
                    s=sizes_d,
                )

#-----------

        if len(paths) > 0:
            for path in paths:

                if len(path) < 2:
                    continue

                xs = [p[0] for p in path]
                ys = [p[1] for p in path]

                ax.plot(xs, ys, color="red", linewidth=0.3)

                dx = xs[-1] - xs[-2]
                dy = ys[-1] - ys[-2]

                ax.arrow(
                    xs[-2], ys[-2],
                    dx, dy,
                    head_width=0.12,
                    head_length=0.12,
                    fc="red",
                    ec="red",
                    length_includes_head=True
                )


        # ----- dead cells -----
        if len(dead) > 0:
            dead = dead.astype(int)
            plt.scatter(
                dead[:,0],
                dead[:,1],
                c="red",
                s=30,
            )

#-----------

        metB_cells = pos[metB == 1]
        if len(metB_cells) > 0:
            plt.scatter(
                metB_cells[:, 0],
                metB_cells[:, 1],
                facecolors="none",
                edgecolors = "yellow",
                linewidths=1.5,
                s=40,
            )

        ax.axis("off")

        out_name = fname.replace(".npz", ".png")
        plt.savefig(
            os.path.join(out_dir, out_name),
            dpi=150,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()



#------------------------------------------------------------------------

def build_video(png_dir="frames_png", output="timelapse.mp4", fps=10):
    import imageio.v2 as imageio
    import os

    files = sorted(os.listdir(png_dir))
    images = [
        imageio.imread(os.path.join(png_dir, f))
        for f in files if f.endswith(".png")
    ]

    imageio.mimsave(output, images, fps=fps)


if __name__=="__main__":
    render_timelapse()
    build_video()

#-------------------------------------------------------------------------


