# environment.py

import numpy as np
import csv
import random
import os
from evo_2_3_copy import ChemotonPopulation, X_IDX, Z_IDX, S_IDX, HAS_MET_B_IDX


#-------------------------------------------------------------------


GRID_W = 100            #changeable parameter
GRID_H = 100            #changeable parameter

center_x = GRID_W // 2
center_y = GRID_H // 2

    #cell.x = random.randrange(GRID_W)
    #cell.y = random.randrange(GRID_H)



#-------------------------------------------------------------------

def food_islands(X_grid, num_islands=5, radius=5, food_amount=1.1):

    spacing = GRID_W // num_islands

    for i in range(num_islands):

        cx = random.randint(i*spacing, (i+1)*spacing-1)
        cy = random.randint(0, GRID_H-1)


        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):

                x = cx + dx
                y = cy + dy

                if 0 <= x < GRID_W and 0 <= y < GRID_H:
                    X_grid[y][x] = food_amount

#-------------------------------------------------------------------

def run_environment():

    alive_history = []
    Vstore_history = []
    S_history = []
    X_history = []
    N_expressed_log = []
    N_history = []
    MetB_history = []
    dead_cells = []   
    paths = {}
    

    log_file = open("run_env.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "step", 
        "alive", 
        "deaths_this_step", 
        "deaths_lowN",
        "deaths_hunger",
        "deaths_shrink_other",
        "X_world", 
        "Z_world", 
        "S_avg", 
        "newborns"
    ])

    pos_file = open("run_positions.csv", "w", newline="")
    pos_writer = csv.writer(pos_file); pos_writer.writerow(["step", "x", "y"])



    # Make some protocells

    population = ChemotonPopulation(population_size=10)

    for cell in population.population:
        x = random.randint(center_x - 2, center_x + 2)
        y = random.randint(center_y - 2, center_y + 2)
        cell.x = x
        cell.y = y
        cell.is_daughter = False

    # Create spatial resource grids

    X_grid = [[0.0 for _ in range(GRID_W)] for _ in range(GRID_H)]
    Z_grid = [[0.0 for _ in range(GRID_W)] for _ in range(GRID_H)]
    
    food_islands(X_grid, num_islands=5)                              #changeable parameter
    food_islands(Z_grid, num_islands=2, radius=3, food_amount=2.0)   #changeable parameter


    dt = 1e-4
    steps = 100   #changeable parameter
    NO_FOOD = 1.0

    record_every = 10
    frame_dir = "run_frames"
    os.makedirs(frame_dir, exist_ok = True)


#-------------------------------------------------------------------


    for step in range(steps):                   #changeable parameter
        if step % 3000 == 0:
            food_islands(X_grid, num_islands=5)
        if step % 5000 == 0:
            food_islands(Z_grid, num_islands=2)

        alive_before = sum(c.alive for c in population.population)
        newborns = []
        deaths_this_step = 0
        deaths_lowN = 0
        deaths_hunger = 0
        deaths_shrink_other = 0



#-------------------------------------------------------------------

        # update ALL cells
        for cell in population.population[:]:
 
            # give this cell local food from the grid
            cell.x = max(0, min(GRID_W - 1, cell.x))
            cell.y = max(0, min(GRID_H - 1, cell.y))

            local_X = X_grid[cell.y][cell.x]
            local_Z = Z_grid[cell.y][cell.x]

            cell.state[X_IDX] = local_X
            cell.state[Z_IDX] = local_Z

#-------------------------------------------------------

            # run one step

            division_occurred = cell.simulate_step(dt, method="rk4")

            if not cell.alive:
                deaths_this_step += 1
                cause = getattr(cell, "death_cause", None)

                if cause == "N_too_small":
                    deaths_lowN += 1
                elif cause == "hunger_shrink":
                    deaths_hunger += 1
                elif cause == "shrink_other":
                    deaths_shrink_other += 1

                cell.death_timer = 30
                dead_cells.append(cell)

                population.population.remove(cell)
                continue


            #consumption rate and reflection on grid:
            cons_X_rate, cons_Z_rate = cell.compute_consumption(local_X, local_Z)
            cons_X = cons_X_rate * dt
            cons_Z = cons_Z_rate * dt

            X_grid[cell.y][cell.x] = max(0.0, local_X - cons_X)
            Z_grid[cell.y][cell.x] = max(0.0, local_Z - cons_Z)


            #after consumption
            local_X_after = X_grid[cell.y][cell.x]
            local_Z_after = Z_grid[cell.y][cell.x]


#--------------------------------------------------------

            #movement of cell:

            key = id(cell)

            if cell.want_move:
                new_x = cell.x + cell.move_dx
                new_y = cell.y + cell.move_dy

                cell.x = max(0, min(GRID_W - 1, new_x))
                cell.y = max(0, min(GRID_H - 1, new_y))
                #--------------------#

                if key not in paths:
                    paths[key] = []

                paths[key].append([cell.x, cell.y])
            else:
                # cell stopped → remove arrows
                paths[key] = []

#----------------------------------------------------------

            # handle reproduction if cell divides
            if division_occurred:
                # create daughter with the same parameters
                daughter = type(cell)(cell.parameters.copy(), cell.template, x=cell.x, y=cell.y)
                daughter.is_daughter = True

                cell.is_daughter = False

                print("before mutate N:", daughter.parameters["N"])

                daughter.mutate()

                N_val = daughter.parameters["N"]
                N_expressed_log.append(N_val)

                print("after mutate N:", N_val)

                newborns.append(daughter)


#---------------------------------------------------------

        population.population.extend(newborns) 

        # update corpses
        new_dead_cells = []
        for corpse in dead_cells:
            corpse.death_timer -= 1
            if corpse.death_timer > 0:
                new_dead_cells.append(corpse)

        dead_cells = new_dead_cells


        # update shared food pool once per step
        X_world = sum(sum(row) for row in X_grid)
        Z_world = sum(sum(row) for row in Z_grid)


        if X_world < NO_FOOD:
            X_world = 0.0
        if Z_world < NO_FOOD:
            Z_world = 0.0



#-------------------------------------------------------------------

        # print something so you see progress
        alive = sum(c.alive for c in population.population)
        deaths = max(0, alive_before - alive)
        S_vals = [c.state[S_IDX] for c in population.population if c.alive]
        S_avg = sum(S_vals) / len(S_vals) if S_vals else 0.0

        print(
            f"Step {step}: alive={alive}, "
            f"X_world={X_world:.3e}, Z_world={Z_world:.3e}, "
            f"S_avg={S_avg:.3e}"
        )

        N_values = [c.parameters["N"] for c in population.population if c.alive]
        MetB_values = [c.state[HAS_MET_B_IDX] for c in population.population if c.alive]

        N_history.append(N_values)
        MetB_history.append(MetB_values)

        writer.writerow([step, alive, deaths_this_step, deaths_lowN, deaths_hunger, deaths_shrink_other, X_world, Z_world, S_avg, len(newborns)])

        for cell in population.population:
            pos_writer.writerow([step, cell.x, cell.y])


        if step % record_every == 0:
            positions = []
            paths_frame = []
            metB_flags = []
            daughter_flags = []
    
            for cell in population.population: 
                positions.append([cell.x, cell.y])
                metB_flags.append(cell.state[HAS_MET_B_IDX] > 0.5)
                daughter_flags.append(getattr(cell, "is_daughter", False))

                key = id(cell)

                if key in paths:
                    paths_frame.append(paths[key][-record_every:])
                else:
                    paths_frame.append([])

            if positions:
                pos_array = np.array(positions)
                metB_array = np.array(metB_flags)
            else:
                pos_array = np.zeros((0,2))
                metB_array = np.zeros((0,))


            dead_positions = [[c.x, c.y] for c in dead_cells]

            np.savez(
                os.path.join(frame_dir, f"frame_{step:05d}.npz"),
                X=np.array(X_grid),
                Z=np.array(Z_grid),
                pos=pos_array,
                paths=np.array(paths_frame, dtype=object),
                metB = metB_array,
                daughter=np.array(daughter_flags),
                dead=np.array(dead_positions),
            )

    log_file.close()
    pos_file.close()

    low_N = [n for n in N_expressed_log if n < 20]
    print("\nN values < 20:")
    print(low_N)

    high_N = [n for n in N_expressed_log if n > 40]
    print("\nN values > 40:")
    print(high_N)

    print("Finished environment run.")


    np.save("final_X.npy", np.array(X_grid))
    np.save("final_Z.npy", np.array(Z_grid))
    np.save("N_history.npy", np.array(N_history, dtype=object))
    np.save("MetB_history.npy", np.array(MetB_history, dtype=object))

    arr = np.array(X_grid)
    print("X min:", arr.min(), "X max:", arr.max())
    arr2 = np.array(Z_grid)
    print("Z min:", arr2.min(), "Z max:", arr2.max())

#-------------------------------------------------------------------

if __name__ == "__main__":
    run_environment()







