import numpy as np
import os
from datetime import datetime
from twoD.environment import MapEnvironment
from twoD.dot_environment import MapDotEnvironment
from twoD.dot_building_blocks import DotBuildingBlocks2D
from twoD.building_blocks import BuildingBlocks2D
from twoD.dot_visualizer import DotVisualizer
from threeD.environment import Environment
from threeD.kinematics import UR5e_PARAMS, Transform
from threeD.building_blocks import BuildingBlocks3D
from threeD.visualizer import Visualize_UR
from AStarPlanner import AStarPlanner
from RRTMotionPlanner import RRTMotionPlanner
from RRTInspectionPlanner import RRTInspectionPlanner
from RRTStarPlanner import RRTStarPlanner
# from twoD.visualizer import Visualizer
from twoD.edited_visualizer import Visualizer
import time
import matplotlib.pyplot as plt


# MAP_DETAILS = {"json_file": "twoD/map1.json", "start": np.array([10,10]), "goal": np.array([4, 6])}
MAP_DETAILS = {"json_file": "twoD/map2.json", "start": np.array([360, 150]), "goal": np.array([100, 200])}


def compute_plan_cost(bb, plan: np.array) -> float:
    """
    Sum of edge costs along the plan using the same metric as the planners.
    plan is expected to be shape (N, 2). Returns 0.0 if plan is empty or has 1 state.
    """
    if plan is None or len(plan) < 2:
        return 0.0

    total = 0.0
    for i in range(len(plan) - 1):
        total += bb.compute_distance(plan[i], plan[i + 1])
    return float(total)



def run_dot_2d_astar():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = AStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])


    # execute plan
    plan = planner.plan()
    print("Report for epsilon = {}".format(planner.epsilon))
    print("Plan found with {} nodes expanded.".format(len(planner.expanded_nodes)))
    print("Plan cost: {}".format(compute_plan_cost(bb, plan)))
    DotVisualizer(bb).visualize_map(plan=plan, expanded_nodes=planner.expanded_nodes, show_map=True, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_dot_2d_rrt():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.2)

    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)

def run_dot_2d_rrt_star():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = RRTStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E1", goal_prob=0.2, k=None)

    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)

def run_2d_rrt_star_motion_planning():
    MAP_DETAILS = {
        "json_file": "twoD/map_mp.json",
        "start": np.array([0.78, -0.78, 0.0, 0.0]),
        "goal": np.array([0.3, 0.15, 1.0, 1.1]),
    }
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)
    vis = Visualizer(bb)
    planner = RRTStarPlanner(
        bb=bb,
        start=MAP_DETAILS["start"],
        goal=MAP_DETAILS["goal"],
        ext_mode="E2",
        goal_prob=0.05,
        max_step_size=0.3,
        visualizer=vis
    )
    # execute plan
    plan = planner.plan()
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_2d_rrt_motion_planning():
    MAP_DETAILS = {"json_file": "twoD/map_mp.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)
    visualizer = Visualizer(bb)
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.05) # visualizer just for debugging ,remove it when submit
    # execute plan
    plan = planner.plan()
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_2d_rrt_inspection_planning():
    MAP_DETAILS = {"json_file": "twoD/map_ip.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="ip")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTInspectionPlanner(bb=bb, start=MAP_DETAILS["start"], ext_mode="E2", goal_prob=0.01, coverage=0.75)

    # execute plan
    plan = planner.plan()
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"])

def run_3d():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)

    bb = BuildingBlocks3D(transform=transform,
                          ur_params=ur_params,
                          env=env,
                          resolution=0.1 )
    
    test_conf = np.deg2rad([130, -70, 90, -90, -90, 0])
    print("Sanity check (should be False):", bb.config_validity_checker(test_conf))

    visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)

    # --------- configurations-------------
    env2_start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
    # ---------------------------------------

    rrt_star_planner = RRTStarPlanner(max_step_size=0.5,
                                      start=env2_start,
                                      goal=env2_goal,
                                      max_itr=4000,
                                      stop_on_goal=True,
                                      bb=bb,
                                      goal_prob=0.05,
                                      ext_mode="E2")

    path = rrt_star_planner.plan()

    if path is not None:
        # create a folder for the experiment
        # Format the time string as desired (YYYY-MM-DD_HH-MM-SS)
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        # create the folder
        exps_folder_name = os.path.join(os.getcwd(), "exps")
        if not os.path.exists(exps_folder_name):
            os.mkdir(exps_folder_name)
        exp_folder_name = os.path.join(exps_folder_name, "exp_pbias_"+ str(rrt_star_planner.goal_prob) + "_max_step_size_" + str(rrt_star_planner.max_step_size) + "_" + time_str)
        if not os.path.exists(exp_folder_name):
            os.mkdir(exp_folder_name)

        # save the path
        np.save(os.path.join(exp_folder_name, 'path'), path)

        # save the cost of the path and time it took to compute
        with open(os.path.join(exp_folder_name, 'stats'), "w") as file:
            file.write("Path cost: {} \n".format(rrt_star_planner.compute_cost(path)))

        visualizer.show_path(path)



def report_part1_compare_extend_avg(n_runs=5, goal_bias=0.20):
    MAP_DETAILS_MP = {
        "json_file": "twoD/map_mp.json",
        "start": np.array([0.78, -0.78, 0.0, 0.0]),
        "goal": np.array([0.3, 0.15, 1.0, 1.1]),
    }
    planning_env = MapEnvironment(json_file=MAP_DETAILS_MP["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)

    ext_modes = ["E1", "E2"]
    results = {}

    for ext in ext_modes:
        times = []
        costs = []

        for i in range(n_runs):
            print(f"Running RRTMotionPlanner with ext_mode={ext}, run={i + 1}")
            planner = RRTMotionPlanner(
                bb=bb,
                start=MAP_DETAILS_MP["start"],
                goal=MAP_DETAILS_MP["goal"],
                ext_mode=ext,
                goal_prob=goal_bias
            )
            t0 = time.time()
            plan = planner.plan()
            t1 = time.time()
           # Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

            times.append(t1 - t0)
            costs.append(planner.compute_cost(plan))

        avg_time = np.mean(times)
        avg_cost = np.mean(costs)

        results[ext] = {
            "times": times,
            "costs": costs,
            "avg_time": avg_time,
            "avg_cost": avg_cost
        }

        print(f"[AVG] ext={ext}, goal_bias={goal_bias}: "
              f"time={avg_time:.3f}s, cost={avg_cost:.3f}")

    return results


def plot_extend_runtime_bars(results):
    fig, ax = plt.subplots(figsize=(8, 4))

    width = 0.35
    x = np.arange(len(results["E1"]["times"]))

    # bars
    ax.bar(x - width/2, results["E1"]["times"], width, label="E1 runs")
    ax.bar(x + width/2, results["E2"]["times"], width, label="E2 runs")

    # averages
    avg_E1 = np.mean(results["E1"]["times"])
    avg_E2 = np.mean(results["E2"]["times"])

    ax.axhline(avg_E1, linestyle="--", linewidth=2, label="E1 average")
    ax.axhline(avg_E2, linestyle=":", linewidth=2, label="E2 average")

    ax.set_xlabel("Run index")
    ax.set_ylabel("Time to solution (s)")
    ax.set_title("E1 vs E2 — Runtime per run + average")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.show()


def plot_extend_cost_bars(results):
    fig, ax = plt.subplots(figsize=(8, 4))

    width = 0.35
    x = np.arange(len(results["E1"]["costs"]))

    # bars
    ax.bar(x - width/2, results["E1"]["costs"], width, label="E1 runs")
    ax.bar(x + width/2, results["E2"]["costs"], width, label="E2 runs")

    # averages
    avg_E1 = np.mean(results["E1"]["costs"])
    avg_E2 = np.mean(results["E2"]["costs"])

    ax.axhline(avg_E1, linestyle="--", linewidth=2, label="E1 average")
    ax.axhline(avg_E2, linestyle=":", linewidth=2, label="E2 average")

    ax.set_xlabel("Run index")
    ax.set_ylabel("Path cost")
    ax.set_title("E1 vs E2 — Path cost per run + average")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.show()



def report_part2_goal_bias(ext_mode_for_bias="E1", n_runs=10):
    MAP_DETAILS_MP = {
        "json_file": "twoD/map_mp.json",
        "start": np.array([0.78, -0.78, 0.0, 0.0]),
        "goal":  np.array([0.3,  0.15, 1.0, 1.1]),
    }
    env_mp = MapEnvironment(json_file=MAP_DETAILS_MP["json_file"], task="mp")
    bb_mp = BuildingBlocks2D(env_mp)

    goal_biases = [0.05, 0.20]
    results = {}  # gb -> list of (time, cost)

    for gb in goal_biases:
        pairs = []
        for i in range(n_runs):
            print(f"Running RRTMotionPlanner with ext_mode={ext_mode_for_bias}, goal_bias={gb}, run={i + 1}")
            planner = RRTMotionPlanner(bb=bb_mp, start=MAP_DETAILS_MP["start"],
                                      goal=MAP_DETAILS_MP["goal"],
                                      ext_mode=ext_mode_for_bias, goal_prob=gb)
            t0 = time.time()
            plan = planner.plan()
            t1 = time.time()

            pairs.append((t1 - t0, planner.compute_cost(plan)))

        results[gb] = pairs

        times = np.array([p[0] for p in pairs])
        costs = np.array([p[1] for p in pairs])

        print(f"[Goal-bias] ext={ext_mode_for_bias}, gb={gb}: "
              f"time mean={times.mean():.3f}s std={times.std(ddof=1):.3f}s, "
              f"cost mean={costs.mean():.3f} std={costs.std(ddof=1):.3f}")

    # Scatter plot time vs cost
    plt.figure()
    for gb in goal_biases:
        xs = [t for (t, c) in results[gb]]
        ys = [c for (t, c) in results[gb]]
        plt.scatter(xs, ys, label=f"{int(gb*100)}% goal bias")
    plt.xlabel("Run time (s)")
    plt.ylabel("Path cost")
    plt.title(f"RRT: time vs cost (ext={ext_mode_for_bias})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.savefig(f"rrt_bias_scatter_ext-{ext_mode_for_bias}.png")
    plt.close()


def report_inspection_planning_performance(n_runs=10):
    """
    Report performance for inspection planning with coverage of 0.5 and 0.75,
    averaged over n_runs executions for each.
    """
    MAP_DETAILS_IP = {"json_file": "twoD/map_ip.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS_IP["json_file"], task="ip")
    bb = BuildingBlocks2D(planning_env)

    
    coverages = [0.5, 0.75]
    results = {}
    
    print("\n" + "="*80)
    print("INSPECTION PLANNING PERFORMANCE REPORT")
    print("="*80 + "\n")
    
    for coverage in coverages:
        times = []
        costs = []
        
        print(f"Running {n_runs} experiments for coverage = {coverage}...")
        print("-" * 60)
        
        for i in range(n_runs):
            print(f"  Run {i + 1}/{n_runs}...", end=" ")
            
            planner = RRTInspectionPlanner(
                bb=bb,
                start=MAP_DETAILS_IP["start"],
                ext_mode="E2",
                goal_prob=0.01,
                coverage=coverage
            )
            
            t0 = time.time()
            plan = planner.plan()
            t1 = time.time()
            
            run_time = t1 - t0
            path_cost = planner.compute_cost(plan) if plan is not None else 0.0
            
            times.append(run_time)
            costs.append(path_cost)
            
            print(f"Time: {run_time:.3f}s, Cost: {path_cost:.3f}")
        
        # Calculate statistics
        times_array = np.array(times)
        costs_array = np.array(costs)
        
        results[coverage] = {
            "times": times,
            "costs": costs,
            "avg_time": times_array.mean(),
            "std_time": times_array.std(ddof=1),
            "avg_cost": costs_array.mean(),
            "std_cost": costs_array.std(ddof=1)
        }
        
        print(f"\n  Results for coverage = {coverage}:")
        print(f"    Average execution time: {results[coverage]['avg_time']:.3f} , {results[coverage]['std_time']:.3f} seconds")
        print(f"    Average path cost:      {results[coverage]['avg_cost']:.3f} , {results[coverage]['std_cost']:.3f}")
        print()
    
    # Summary comparison
    print("="*80)
    print("SUMMARY")
    print("="*80)
    for coverage in coverages:
        print(f"\nCoverage {coverage}:")
        print(f"  Execution Time: avg={results[coverage]['avg_time']:.3f} , std={results[coverage]['std_time']:.3f} s")
        print(f"  Path Cost:      avg{results[coverage]['avg_cost']:.3f} , std={results[coverage]['std_cost']:.3f}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['steelblue', 'coral']
    
    # Plot 1: Execution Time - all runs as dots + average lines
    for idx, coverage in enumerate(coverages):
        times = results[coverage]['times']
        avg_time = results[coverage]['avg_time']
        std_time = results[coverage]['std_time']
        
        # Plot individual runs as scatter points with iteration number on x-axis
        x_positions = list(range(1, len(times) + 1))  # 1, 2, 3, ..., 10
        ax1.scatter(x_positions, times, 
                   alpha=0.7, s=100, color=colors[idx], 
                   label=f'Coverage {coverage}', edgecolors='black', linewidth=0.5)
        
        # Plot average as horizontal line across all iterations
        ax1.axhline(avg_time, color=colors[idx], linewidth=2.5, 
                   linestyle='--', alpha=0.8, label=f'Coverage {coverage} (avg)')
        
        # Add text annotation for avg and std
        ax1.text(len(times) + 0.5, avg_time, 
                f'avg={avg_time:.2f}s',
                fontsize=9, va='center', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[idx], alpha=0.3))
    
    ax1.set_xlabel('Run Number', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('Execution Time by Coverage\n(Individual Runs + Average)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    ax1.set_xticks(range(1, n_runs + 1))
    
    # Plot 2: Path Cost - all runs as dots + average lines
    for idx, coverage in enumerate(coverages):
        costs = results[coverage]['costs']
        avg_cost = results[coverage]['avg_cost']
        std_cost = results[coverage]['std_cost']
        
        # Plot individual runs as scatter points with iteration number on x-axis
        x_positions = list(range(1, len(costs) + 1))  # 1, 2, 3, ..., 10
        ax2.scatter(x_positions, costs, 
                   alpha=0.7, s=100, color=colors[idx], 
                   label=f'Coverage {coverage}', edgecolors='black', linewidth=0.5)
        
        # Plot average as horizontal line across all iterations
        ax2.axhline(avg_cost, color=colors[idx], linewidth=2.5, 
                   linestyle='--', alpha=0.8, label=f'Coverage {coverage} (avg)')
        
        # Add text annotation for avg and std
        ax2.text(len(costs) + 0.5, avg_cost, 
                f'avg={avg_cost:.2f}',
                fontsize=9, va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[idx], alpha=0.3))
    
    ax2.set_xlabel('Run Number', fontsize=12)
    ax2.set_ylabel('Path Cost', fontsize=12)
    ax2.set_title('Path Cost by Coverage\n(Individual Runs + Average)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    ax2.set_xticks(range(1, n_runs + 1))
    
    plt.tight_layout()
    plt.savefig('inspection_planning_performance.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'inspection_planning_performance.png'")
    plt.show()
    
    return results



def run_3d_rrtstar_planwithstats_test():
    # --- Build 3D world ---
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)

    bb = BuildingBlocks3D(
        transform=transform,
        ur_params=ur_params,
        env=env,
        resolution=0.1
    )

    # --- Sanity check requested by HW3 (window constraint) ---
    test_conf = np.deg2rad([130, -70, 90, -90, -90, 0])
    print("Sanity check (should be False):", bb.config_validity_checker(test_conf))

    # --- Start/Goal for env_idx=2 ---
    start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
    goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])

    # --- Create planner ---
    planner = RRTStarPlanner(
        bb=bb,
        ext_mode="E2",
        max_step_size=0.2,     # try also 0.05, 0.4 later
        start=start,
        goal=goal,
        max_itr=2000,          # HW3 requirement
        stop_on_goal=False,    # HW3: do not stop at first solution
        goal_prob=0.2         # p_bias
    )

    # --- Run with stats ---
    path, iters, costs, success = planner.plan_with_stats(log_every=50)

    print("\n--- plan_with_stats result ---")
    print("Logged points:", len(iters))
    print("Last iter:", iters[-1])
    print("Final success:", success[-1])
    if np.isfinite(costs[-1]):
        print("Final best cost:", costs[-1])
    else:
        print("No solution found within budget.")

    # --- Plot: success vs iteration ---
    plt.figure()
    plt.plot(iters, success)
    plt.xlabel("Iteration")
    plt.ylabel("Success (0/1)")
    plt.title("RRT* Success vs Iteration (single run)")
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- Plot: cost vs iteration (ignore inf by plotting as gaps) ---
    plt.figure()
    costs_plot = costs.copy()
    costs_plot[np.isinf(costs_plot)] = np.nan  # show gaps before first solution
    plt.plot(iters, costs_plot)
    plt.xlabel("Iteration")
    plt.ylabel("Cost (NaN before first solution)")
    plt.title("RRT* Cost vs Iteration (single run)")
    plt.grid(True, alpha=0.3)
    plt.show()

    return path, iters, costs, success



def run_3d_hw3_save_all_and_mark_best():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)

    bb = BuildingBlocks3D(
        transform=transform,
        ur_params=ur_params,
        env=env,
        resolution=0.1
    )

    # Sanity check (HW3 3.2 part 1)
    test_conf = np.deg2rad([130, -70, 90, -90, -90, 0])
    print("Sanity check (should be False):", bb.config_validity_checker(test_conf))

    # Start / Goal
    start = np.deg2rad([110, -70, 90, -90, -90, 0])
    goal  = np.deg2rad([50,  -80, 90, -90, -90, 0])

    max_step_sizes = [0.05, 0.075, 0.1, 0.125, 0.2, 0.25, 0.3, 0.4]
    p_biases = [0.05, 0.2]
    n_runs = 20

    # Root experiments folder
    exps_root = os.path.join(os.getcwd(), "exps")
    os.makedirs(exps_root, exist_ok=True)

    # Track GLOBAL best
    best_cost = np.inf
    best_info = None   # dict with metadata

    for p_bias in p_biases:
        for step in max_step_sizes:
            print(f"\n=== p_bias={p_bias}, max_step_size={step} ===")

            for run_idx in range(1, n_runs + 1):
                print(f"  Run {run_idx}/{n_runs}...", end=" ")

                planner = RRTStarPlanner(
                    bb=bb,
                    ext_mode="E2",
                    max_step_size=step,
                    start=start,
                    goal=goal,
                    max_itr=2000,
                    stop_on_goal=False,   
                    goal_prob=p_bias
                )

                path = planner.plan()

                if path is None or len(path) == 0:
                    print("no solution")
                    continue

                cost = planner.compute_cost(path)
                print(f"cost={cost:.4f}")

                # ---- Save THIS run (same style as example) ----
                now = datetime.now()
                time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

                run_folder = os.path.join(
                    exps_root,
                    f"exp_pbias_{p_bias}_max_step_size_{step}_run_{run_idx}_{time_str}"
                )
                os.makedirs(run_folder, exist_ok=True)

                np.save(os.path.join(run_folder, "path.npy"), path)

                with open(os.path.join(run_folder, "stats.txt"), "w") as f:
                    f.write(f"p_bias: {p_bias}\n")
                    f.write(f"max_step_size: {step}\n")
                    f.write(f"run_idx: {run_idx}\n")
                    f.write(f"path_cost: {cost}\n")
                    f.write(f"num_states: {len(path)}\n")

                # ---- Update GLOBAL best ----
                if cost < best_cost:
                    best_cost = cost
                    best_info = {
                        "p_bias": p_bias,
                        "max_step_size": step,
                        "run_idx": run_idx,
                        "cost": cost,
                        "folder": run_folder,
                        "path": path
                    }

    # -------- FINAL SUMMARY (this is what you wanted) --------
    print("\n" + "="*70)
    if best_info is None:
        print("NO solution found in any run.")
        return

    print("BEST PATH FOUND ✅")
    print(f"  cost          : {best_info['cost']}")
    print(f"  p_bias        : {best_info['p_bias']}")
    print(f"  max_step_size : {best_info['max_step_size']}")
    print(f"  run_idx       : {best_info['run_idx']}")
    print(f"  folder        : {best_info['folder']}")
    print("="*70 + "\n")

    # Optional: visualize best path immediately
    visualizer = Visualize_UR(
        ur_params,
        env=env,
        transform=transform,
        bb=bb
    )
    visualizer.show_path(best_info["path"])




if __name__ == "__main__":
    #run_dot_2d_astar()
    #run_dot_2d_rrt()
    # run_dot_2d_rrt_star()
    # run_2d_rrt_motion_planning()
    # run_2d_rrt_inspection_planning()
    #run_2d_rrt_star_motion_planning()
    #run_3d()
    #run_3d_rrtstar_planwithstats_test()

    #results = report_part1_compare_extend_avg(n_runs=10, goal_bias=0.20)
    #plot_extend_runtime_bars(results)
    #plot_extend_cost_bars(results)

    #report_part2_goal_bias()
    
    # Run inspection planning performance report
    #report_inspection_planning_performance(n_runs=10)