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
from twoD.visualizer import Visualizer
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
    planner = RRTStarPlanner(
        bb=bb,
        start=MAP_DETAILS["start"],
        goal=MAP_DETAILS["goal"],
        ext_mode="E2",
        goal_prob=0.05,
        max_step_size=0.3,
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
    planner = RRTInspectionPlanner(bb=bb, start=MAP_DETAILS["start"], ext_mode="E2", goal_prob=0.01, coverage=0.5)

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


if __name__ == "__main__":
    run_dot_2d_astar()
    #run_dot_2d_rrt()
    # run_dot_2d_rrt_star()
    # run_2d_rrt_motion_planning()
    # run_2d_rrt_inspection_planning()
    # run_2d_rrt_star_motion_planning()
    # run_3d()

    #results = report_part1_compare_extend_avg(n_runs=10, goal_bias=0.20)
    #plot_extend_runtime_bars(results)
    #plot_extend_cost_bars(results)

    #report_part2_goal_bias()