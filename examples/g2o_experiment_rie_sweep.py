import sys
import random
import numpy as np
import networkx as nx
from timeit import default_timer as timer
from pose_graph_utils import split_edges, read_g2o_file, plot_poses, rpm_to_mac, RelativePoseMeasurement, poses_ate_tran, poses_rpe_rot
import pickle

# MAC requirements
from mac.solvers import MAC, NaiveGreedy
from mac.utils.graphs import Edge
from mac.utils.rounding import round_madow
from tqdm import tqdm
from mac.optimization.line_search import ConstantAlphaLineSearcher, ConstantStepSizeLineSearcher, ShrinkingAlphaLineSearcher, ShrinkingStepSizeLineSearcher

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 16})

# SE-Sync setup
# sesync_lib_path = "/Users/kevin/repos/SESync/C++/build/lib"
sesync_lib_path = "/home/jungle/Git/SE-Sync/C++/build/lib"
sesync_lib_path = "/home/jungle/.local/lib/python3.10/site-packages/PySESync.cpython-310-x86_64-linux-gnu.so"
sys.path.insert(0, sesync_lib_path)

def uniquify(filenmae):
    """
    Create a unique file name by appending a number if the file already exists.
    """
    import os
    base, ext = os.path.splitext(filenmae)
    i = 1
    while os.path.exists(f"{base}_{i}{ext}"):
        i += 1
    return f"{base}_{i}{ext}"

def run_and_save_Mac(num_lc, odom_edges, lc_edges, lc_measurements, num_poses, rie_alg, memory, step_size_searcher):
    mac = MAC(odom_edges, lc_edges, num_poses, rie_alg=rie_alg, memory=memory, step_size_searcher=step_size_searcher)
    print("Num LC to accept: ", num_lc)
    rie_init = np.ones(len(lc_measurements)) * num_lc / len(lc_measurements)
    mac.reset_log()
    start_r = timer()
    result_rie, unrounded_rie, upper_rie, rtime_rie = mac.solve_rie(num_lc, rie_init, max_iters=30, rounding="nearest", return_rounding_time=True, use_cache=False, verbose=True)
    end_r = timer()
    solve_time_rie = end_r-start_r

    output_dict = {
        "costs_per_iteration_rie": mac.optimizer._log['iterations']['cost'],
        "num_iterations_rie": len(mac.optimizer._log['iterations']['iteration']),
        "time": solve_time_rie,
        "result": result_rie,
        "unrounded_result": unrounded_rie,
        "upper_bound": upper_rie,
        "rie_fiedler_iters": mac.prob_rie_run_info['fiedler_iters'],
        "rie_fiedler_times": mac.prob_rie_run_info['fiedler_runtimes'],
        "rie_fiedler_ratio": mac.prob_rie_run_info['ratio']
    }
    return output_dict

def main(args):
    run_greedy = False
    # if len(sys.argv) > 2:
    #     if sys.argv[2] == "--run-greedy":
    #         run_greedy = True
    #         pass
    #     else:
    #         print(f"Unknown argument: {sys.argv[2]}")
    #         print(f"Usage: {sys.argv[0]} [.g2o file] [optional: --run-greedy]")
    #         sys.exit()
    #         pass
        # pass
    # rie_alg="rgd"
    # dataset_name = sys.argv[1].split('/')[-1].split('.')[0]
    dataset_name = args.g2o.split('/')[-1].split('.')[0]

    print(f"Loading dataset: {dataset_name}")

    # Load a g2o file
    print("Reading g2o file")
    start = timer()
    # measurements, num_poses = read_g2o_file(sys.argv[1])
    measurements, num_poses = read_g2o_file(args.g2o)
    end = timer()
    print("Success! elapsed time: ", (end - start))

    # Split measurements into odom and loop closures
    odom_measurements, lc_measurements = split_edges(measurements)

    # Convert measurements to MAC edge format
    odom_edges = rpm_to_mac(odom_measurements)
    lc_edges = rpm_to_mac(lc_measurements)

    # Print dataset stats
    print(f"Loaded {len(measurements)} total measurements with: ")
    print(f"\t {len(odom_measurements)} base (odometry) measurements and")
    print(f"\t {len(lc_measurements)} candidate (loop closure) measurements")


    #############################
    # Running the tests!
    #############################

    # Test between 100% and 0% loop closures
    # NOTE: If running greedy, these must be in increasing order!
    # percent_lc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # percent_lc = [0.1, 0.3, 0.5, 0.7, 0.9]
    percent_lc = [0.2]

    for starting_stepsize in tqdm([0.1,1.0,10.0]):
        for pct_lc in percent_lc:
            num_lc = int(pct_lc * len(lc_measurements))
            for rie_alg in ["rbfgs"]:
                if rie_alg == "rbfgs":
                    for memory in [1,5,10, 15]:
                    # for memory in [1,5,10, 15]:
                            const_alpha_dict = run_and_save_Mac(num_lc, odom_edges, lc_edges, 
                                            lc_measurements, 
                                            num_poses, 
                                            rie_alg, 
                                            memory, 
                                            step_size_searcher=ConstantAlphaLineSearcher(alpha=starting_stepsize))
                            filename = uniquify(f"{rie_alg}_g2o_experiment_{dataset_name}_mem{memory}_const_alpha_{starting_stepsize}.pkl")
                            pickle.dump(const_alpha_dict, open(filename, 'wb'))
                            const_step_size_dict = run_and_save_Mac(num_lc, odom_edges, lc_edges, 
                                            lc_measurements, 
                                            num_poses, 
                                            rie_alg, 
                                            memory, 
                                            step_size_searcher=ConstantStepSizeLineSearcher(step_size=starting_stepsize))
                            filename=uniquify(f"{rie_alg}_g2o_experiment_{dataset_name}_mem{memory}_const_step_size_{starting_stepsize}.pkl")
                            pickle.dump(const_step_size_dict, open(filename, 'wb'))
                            for decay_rate in [2,4,10]:
                                shrink_alpha_dict =  run_and_save_Mac(num_lc, odom_edges, lc_edges, 
                                            lc_measurements, 
                                            num_poses, 
                                            rie_alg, 
                                            memory, 
                                            step_size_searcher=ShrinkingStepSizeLineSearcher(step_size=starting_stepsize, decay_rate=decay_rate))
                                filename=uniquify(f"{rie_alg}_g2o_experiment_{dataset_name}_mem{memory}_shrink_step_size_{starting_stepsize}_{decay_rate}.pkl")
                                pickle.dump(shrink_alpha_dict, open(filename, 'wb'))
                                shrink_step_size_dict =  run_and_save_Mac(num_lc, odom_edges, lc_edges, 
                                            lc_measurements, 
                                            num_poses, 
                                            rie_alg, 
                                            memory, 
                                            step_size_searcher=ShrinkingAlphaLineSearcher(alpha=starting_stepsize, decay_rate=decay_rate))
                                filename=uniquify(f"{rie_alg}_g2o_experiment_{dataset_name}_mem{memory}_shrink_step_size_{starting_stepsize}_{decay_rate}.pkl")
                                pickle.dump(shrink_step_size_dict, open(filename, 'wb'))


    #############################
    # Plot the Results
    #############################

    # Debug Fiedler Calculation Plots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(num_total_list)):
        num_calls_mac = len(mac_fiedler_iters[i])
        ax.plot([num_total_list[i]]*num_calls_mac, 
                np.arange(num_calls_mac), 
                mac_fiedler_times[i], label="MAC", linestyle='--', marker='o', color='b')

        num_calls_rie = len(rie_fiedler_iters[i])
        ax.plot([num_total_list[i]]*num_calls_rie, 
                np.arange(num_calls_rie), 
                rie_fiedler_times[i], label="Riemannian", linestyle='--', marker='o', color='r')
    
    ax.set_xlabel("Num Total Edges")
    ax.set_ylabel("Iterations")
    ax.set_zlabel("Fiedler Tracemin Average Time")
    ax.set_title("Fiedler Tracemin Loop Avg Time vs Num Total Edges and Iterations")
    plt.savefig(f"{init_strategy}_{rie_alg}_fiedler_loop_runtimes_vs_edges_and_iterations.png")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(num_total_list)):
        num_calls_mac = len(mac_fiedler_iters[i])
        ax.plot([num_total_list[i]]*num_calls_mac, 
                np.arange(num_calls_mac), 
                mac_fiedler_iters[i], label="MAC", linestyle='--', marker='o', color='b')

        num_calls_rie = len(rie_fiedler_iters[i])
        ax.plot([num_total_list[i]]*num_calls_rie, 
                np.arange(num_calls_rie), 
                rie_fiedler_iters[i], label="Riemannian", linestyle='--', marker='o', color='r')
    
    ax.set_xlabel("Num Total Edges")
    ax.set_ylabel("Iterations")
    ax.set_zlabel("Fiedler Tracemin Average Iterations")
    ax.set_title("Fiedler Tracemin Average Iterations vs Num Total Edges and Iterations")
    plt.savefig(f"{init_strategy}_{rie_alg}_fiedler_tracemeing_iters_vs_edges_and_iterations.png")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(num_total_list)):
        num_calls_mac = len(mac_fiedler_iters[i])
        ax.plot([num_total_list[i]]*num_calls_mac, 
                np.arange(num_calls_mac), 
                mac_fiedler_ratio[i], label="MAC", linestyle='--', marker='o', color='b')

        num_calls_rie = len(rie_fiedler_iters[i])
        ax.plot([num_total_list[i]]*num_calls_rie, 
                np.arange(num_calls_rie), 
                rie_fiedler_ratio[i], label="Riemannian", linestyle='--', marker='o', color='r')
    
    ax.set_xlabel("Num Total Edges")
    ax.set_ylabel("Iterations")
    ax.set_zlabel("Fiedler Tracemin ratio")
    ax.set_title("Fiedler Tracemin ratio vs Num Total Edges and Iterations")
    plt.savefig(f"{init_strategy}_{rie_alg}_fiedler_tracemeing_ratio_vs_edges_and_iterations.png")

    # Plot Other Imoprtant stuff

    colors = {
            "MAC Nearest (Ours)": "C0",
              "MAC Madow (Ours)": "C4",
              "Rie Nearest (Ours)": "C5",
              "Rie Madow (Ours)": "C6",
              "Unrounded": "C2",
              "Dual Upper Bound": "C0",
              "Greedy ESP": "C1",
              "Naive Method": "C3"}

    # plot connectivity vs. percent_lc
    naive_objective_vals = [mac.evaluate_objective(naive_result) for naive_result in naive_results]

    mac_objective_vals = [mac.evaluate_objective(result) for result in results]
    unrounded_objective_vals = [mac.evaluate_objective(unrounded) for unrounded in unrounded_results]
    madow_objective_vals = [mac.evaluate_objective(madow) for madow in madow_results]

    mac_objective_vals_rie = [mac.evaluate_objective(result) for result in results_rie]
    unrounded_objective_vals_rie = [mac.evaluate_objective(unrounded) for unrounded in unrounded_results_rie]
    madow_objective_vals_rie = [mac.evaluate_objective(madow) for madow in madow_results_rie]

    if run_greedy:
        # greedy_eig_objective_vals = [mac.evaluate_objective(ge_result) for ge_result in greedy_eig_results]
        greedy_esp_objective_vals = [mac.evaluate_objective(ge_result) for ge_result in greedy_esp_results]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(percent_lc)):
        ax.plot([percent_lc[i]]*num_iterations_mac[i], 
                np.arange(num_iterations_mac[i]), 
                costs_per_iteration_mac[i], label="MAC", linestyle='--', marker='o', color='b')

        ax.plot([percent_lc[i]]*num_iterations_rie[i], 
                np.arange(num_iterations_rie[i]), 
                costs_per_iteration_rie[i], label="Riemannian", linestyle='--', marker='o', color='r')
    ax.set_xlabel("Num Total Edges")
    ax.set_ylabel("Iterations")
    ax.set_zlabel("Cost")
    ax.set_title("Cost vs Num Total Edges and Iterations")
    # ax.legend()
    plt.savefig(f"{init_strategy}_{rie_alg}_cost_vs_edges_and_iterations_{dataset_name}.png")

    plt.figure()
    plt.plot(100.0*np.array(percent_lc), madow_objective_vals, label='MAC Madow', marker='o', color=colors["MAC Madow (Ours)"])
    plt.plot(100.0*np.array(percent_lc), madow_objective_vals_rie, label='Rie Madow (Ours)', marker='o', color=colors["Rie Madow (Ours)"])
    plt.plot(100.0*np.array(percent_lc), unrounded_objective_vals, label='Unrounded Mac', color=colors["Unrounded"])
    plt.plot(100.0*np.array(percent_lc), unrounded_objective_vals_rie, label='Unrounded Rie', color=colors["Unrounded"], linestyle="--")
    plt.plot(100.0*np.array(percent_lc), naive_objective_vals, label='Naive Method', marker='o', color=colors["Naive Method"])

    plt.ylabel(r'Algebraic Connectivity $\lambda_2$')
    plt.xlabel(r'\% Edges Added')
    plt.legend()
    plt.savefig(f"{init_strategy}_alg_conn_{dataset_name}.png", dpi=600, bbox_inches='tight')
    # plt.savefig(f"alg_conn_{dataset_name}_300.png", dpi=300, bbox_inches='tight')
    # plt.savefig(f"alg_conn_{dataset_name}.svg", transparent=True,bbox_inches='tight')

    # Plot computation time vs. percent_lc
    plt.figure()
    plt.semilogy(100.0*np.array(percent_lc), madow_times, label='MAC Madow', marker='o', color=colors["MAC Madow (Ours)"])
    
    plt.semilogy(100.0*np.array(percent_lc), madow_times_rie, label='Rie Madow (Ours)', marker='o', color=colors["Rie Madow (Ours)"])
    if run_greedy:
        # plt.plot(100.0*np.array(percent_lc), greedy_eig_times, label='Greedy E-Opt', color='orange')
        plt.semilogy(100.0*np.array(percent_lc), greedy_esp_times, label='Greedy ESP', marker='o', color=colors["Greedy ESP"])
    plt.xlim([0.0, 100.0])
    plt.ylabel(r'Time (s)')
    plt.xlabel(r'\% Edges Added')
    plt.legend()
    plt.savefig(f"{init_strategy}_comp_time_{dataset_name}.png", dpi=600, bbox_inches='tight')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2o", type=str, help="Path to the g2o file")
    parser.add_argument("--init", type=str, help="initialization method", default="naive")
    args = parser.parse_args()
    main(args)
    # plt.show()
    # if len(sys.argv) < 2:
    #     print(f"Usage: {sys.argv[0]} [.g2o file] [optional: --run-greedy]")
    #     sys.exit()
    #     pass
