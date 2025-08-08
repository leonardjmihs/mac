import sys
import random
import numpy as np
import networkx as nx
from timeit import default_timer as timer
from pose_graph_utils import split_edges, read_g2o_file, plot_poses, rpm_to_mac, RelativePoseMeasurement, poses_ate_tran, poses_rpe_rot

# MAC requirements
from mac.solvers import MAC, NaiveGreedy
from mac.utils.graphs import Edge
from mac.utils.rounding import round_madow

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 16})

# SE-Sync setup
# sesync_lib_path = "/Users/kevin/repos/SESync/C++/build/lib"
sesync_lib_path = "/home/jungle/Git/SE-Sync/C++/build/lib"
sesync_lib_path = "/home/jungle/.local/lib/python3.10/site-packages/PySESync.cpython-310-x86_64-linux-gnu.so"
sys.path.insert(0, sesync_lib_path)

import PySESync

def orbit_distance_dS(X, Y, compute_G_S=False):
    """
    Compute SO(d) orbit distance between X and Y
    if compute_G_S = True, also returns optimal registration
    """
    d = X.shape[0]
    n = int(X.shape[1] / d)

    XYt = X @ Y.T

    u, s, vh = np.linalg.svd(XYt)

    uvh = u @ vh
    Sigma = np.diag(s)

    Xi_diag = np.diag(np.ones(d))
    Xi_diag[d-1, d-1] = np.copysign(1.0, np.linalg.det(uvh))

    dS = np.sqrt(np.abs(
        2 * d * n - 2 * np.trace(Xi_diag @ Sigma)
        ))

    if compute_G_S:
        G_S = u @ Xi_diag @ vh
        return dS, G_S
    return dS

def construct_LGrho(measurements):
    d = len(measurements[0].t) if len(measurements) > 0 else 0

    # Scan through the measurements to determine the total number of poses
    num_poses = 0
    for measurement in measurements:
        max_pair = max(measurement.i, measurement.j)
        if max_pair > num_poses:
            num_poses = max_pair
    num_poses = num_poses + 1

    # Build empty rotation graph Laplacian matrix
    LGrho = np.zeros([d * num_poses, d * num_poses])

    # Now scan through measurements again and populate M
    for measurement in measurements:
        i = measurement.i
        j = measurement.j

        # Add elements for L(G^rho)
        # Elements of ith block-diagonal
        for k in range(0, d):
            LGrho[d * i + k, d * i + k] += measurement.kappa

        # Element of jth block-diagonal
        for k in range(0, d):
            LGrho[d * j + k, d * j + k] += measurement.kappa

        # Elements of ij block
        for r in range(0, d):
            for c in range(0, d):
                LGrho[i * d + r, j * d + c] += -measurement.kappa * measurement.R[r, c]

        # Elements of ji block
        for r in range(0, d):
            for c in range(0, d):
                LGrho[j * d + r, i * d + c] += -measurement.kappa * measurement.R[c, r]

    return LGrho

def evaluate_sesync_rotation_objective(LGrho, R):
    return np.trace(R @ LGrho @ R.T)

def construct_sesync_quadratic_form_matrix(measurements):
    """
    This function emulates the behavior of the `construct_quadratic_form_matrix`
    function in SE-Sync. It builds the matrix M for the SE(d) synchronization
    problem defined as

    min [t; vec(R)]' (M \otimes I_d) [t; vec(R)]

    Note: This is the "translation explicit" version of the problem (i.e.
    without translations marginalized out).

    """

    d = len(measurements[0].t) if len(measurements) > 0 else 0

    # Scan through the measurements to determine the total number of poses
    num_poses = 0
    for measurement in measurements:
        max_pair = max(measurement.i, measurement.j)
        if max_pair > num_poses:
            num_poses = max_pair
    num_poses = num_poses + 1

    # Build empty quadratic form matrix
    M = np.zeros([(d + 1) * num_poses, (d + 1) * num_poses])

    # Now scan through measurements again and populate M
    for measurement in measurements:
        i = measurement.i
        j = measurement.j

        # Add elements for L(W^tau)
        M[i, i] += measurement.tau
        M[j, j] += measurement.tau
        M[i, j] += -measurement.tau
        M[j, i] += -measurement.tau

        # Add elements for V (upper-right block)
        for k in range(0, d):
            M[i, num_poses + i * d + k] += measurement.tau * measurement.t[k]
        for k in range(0, d):
            M[j, num_poses + i * d + k] += -measurement.tau * measurement.t[k]

        # Add elements for V' (lower-left block)
        for k in range(0, d):
            M[num_poses + i * d + k, i] += measurement.tau * measurement.t[k]
        for k in range(0, d):
            M[num_poses + i * d + k, j] += -measurement.tau * measurement.t[k]

        # Add elements for L(G^rho)
        # Elements of ith block-diagonal
        for k in range(0, d):
            M[num_poses + d * i + k,
              num_poses + d * i + k] += measurement.kappa

        # Element of jth block-diagonal
        for k in range(0, d):
            M[num_poses + d * j + k,
              num_poses + d * j + k] += measurement.kappa

        # Elements of ij block
        for r in range(0, d):
            for c in range(0, d):
                M[num_poses + i * d + r, num_poses + j * d +
                  c] += -measurement.kappa * measurement.R[r, c]

        # Elements of ji block
        for r in range(0, d):
            for c in range(0, d):
                M[num_poses + j * d + r, num_poses + i * d +
                  c] += -measurement.kappa * measurement.R[c, r]

        # Add elements for Sigma
        for r in range(0, d):
            for c in range(0, d):
                M[num_poses + i * d + r, num_poses + i * d +
                  c] += measurement.tau * measurement.t[r] * measurement.t[c]

    return M

def evaluate_sesync_objective(M, Xhat):
    return np.trace(Xhat @ M @ Xhat.T)


def nx_rot_G_w(measurements, num_poses):
    G = nx.generators.classic.empty_graph(num_poses)
    for measurement in measurements:
        i = measurement.i
        j = measurement.j
        w = measurement.kappa
        G.add_edge(i, j, weight=w)
    return G

def nx_tran_G_w(measurements, num_poses):
    G = nx.generators.classic.empty_graph(num_poses)
    for measurement in measurements:
        i = measurement.i
        j = measurement.j
        w = measurement.tau
        G.add_edge(i, j, weight=w)
    return G

def select_measurements(measurements, w):
    assert(len(measurements) == len(w))
    meas_out = []
    for i, meas in enumerate(measurements):
        if w[i] == 1.0:
            meas_out.append(meas)
    return meas_out

def to_sesync_format(measurements):
    """
    Convert a set of RelativePoseMeasurement to PySESync
    RelativePoseMeasurement. Requires PySESync import.
    """
    sesync_measurements = []
    for meas in measurements:
        sesync_meas = PySESync.RelativePoseMeasurement()
        sesync_meas.i = meas.i
        sesync_meas.j = meas.j
        sesync_meas.kappa = meas.kappa
        sesync_meas.tau = meas.tau
        sesync_meas.R = meas.R
        sesync_meas.t = meas.t
        sesync_measurements.append(sesync_meas)
    return sesync_measurements

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2o", type=str, help="Path to the g2o file")
    parser.add_argument("--init", type=str, help="initialization method", default="naive")
    args = parser.parse_args()
    # plt.show()
    # if len(sys.argv) < 2:
    #     print(f"Usage: {sys.argv[0]} [.g2o file] [optional: --run-greedy]")
    #     sys.exit()
    #     pass

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
    rie_alg="rbfgs"
    # rie_alg="rgd"
    # dataset_name = sys.argv[1].split('/')[-1].split('.')[0]
    dataset_name = args.g2o.split('/')[-1].split('.')[0]
    init_strategy = args.init

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

    # solvers = [("MAC", MAC(odom_edges, lc_edges, num_poses)),
    #            ("Naive", NaiveGreedy(lc_edges))]

    # if run_greedy:
    #     solvers.append(("GreedyESP", Greedy(odom_edges, lc_edges, num_poses, lazy=True)))
    #     pass

    # Make a MAC Solver
    # mac = MAC(odom_edges, lc_edges, num_poses, fiedler_method="tracemin_cholesky", rie_alg=rie_alg)
    mac = MAC(odom_edges, lc_edges, num_poses, rie_alg=rie_alg)

    # Make a Naive Solver
    naive = NaiveGreedy(lc_edges)

    # Make a GreedyEig Solver
    if run_greedy:
        from mac.solvers.greedy_esp import GreedyESP
        greedy_esp = GreedyESP(odom_edges, lc_edges, num_poses, lazy=True)

    #############################
    # Running the tests!
    #############################

    # Test between 100% and 0% loop closures
    # NOTE: If running greedy, these must be in increasing order!
    # percent_lc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # percent_lc = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # percent_lc = [0.1, 0.2, 0.3]
    percent_lc = [0.3, 0.5, 0.7, 0.9]
    # percent_lc = [0.5, 0.9]

    # Container for Naive results
    naive_results = []

   # Containers for MAC results
    results = []
    unrounded_results = []
    upper_bounds = []
    times = []


    results_rie = []
    unrounded_results_rie = []
    upper_bounds_rie = []
    times_rie = []

    madow_results = []
    madow_times = []

    madow_results_rie = []
    madow_times_rie = []

    # Container for GreedyEig results
    greedy_eig_results = []
    greedy_eig_times = []
    
    # Container for GreedyESP results
    greedy_esp_results = []
    greedy_esp_times = []

    costs_per_iteration_mac = []
    costs_per_iteration_rie = []

    num_iterations_mac = []
    num_iterations_rie = []

    mac_fiedler_times = []
    mac_fiedler_iters = []
    mac_fiedler_ratio= []

    rie_fiedler_times = []
    rie_fiedler_iters = []
    rie_fiedler_ratio = []

    num_total_list = []

    for pct_lc in percent_lc:
        num_lc = int(pct_lc * len(lc_measurements))
        num_total_list.append(num_lc)
        print("Num LC to accept: ", num_lc)

        # Compute a solution using the naive method. This serves both as a
        # baseline and as a sparse initializer for our method.
        naive_result = naive.subset(num_lc)
        naive_results.append(naive_result)
        

        fw_init = naive_result
        rie_init = np.ones(len(lc_measurements)) * num_lc / len(lc_measurements)

        if init_strategy == "naive":
            rie_init = fw_init
        elif init_strategy == "center":
            fw_init=rie_init
        elif init_strategy == "both":
            pass
        else:
            print(f"Unknown initialization strategy: {init_strategy}")
            sys.exit(1)

        # fw_init = rie_init
        # rie_init = fw_init

        # allows for magical jit stuff to happen without factoringi into solve time
        # mac.solve_rie(num_lc, w_init, max_iters=2, rounding="nearest", return_rounding_time=True, use_cache=True, verbose=True)
        mac.reset_log()

        start_r = timer()
        result_rie, unrounded_rie, upper_rie, rtime_rie = mac.solve_rie(num_lc, rie_init, max_iters=20, rounding="nearest", return_rounding_time=True, use_cache=False, verbose=True)
        end_r = timer()
        solve_time_rie = end_r-start_r
        times_rie.append(solve_time_rie)
        results_rie.append(result_rie)
        upper_bounds_rie.append(upper_rie)
        unrounded_results_rie.append(unrounded_rie)
        
        start = timer()
        madow_rounded_rie = round_madow(unrounded_rie, num_lc, seed=np.random.RandomState(42))
        end = timer()
        madow_results_rie.append(madow_rounded_rie)
        # Time for Madow rounded solution is total MAC time (including nearest
        # neighbor rounding) plus the time to perform Madow rounding, minus the
        # nearest neighbor rounding time. Because Madow and nearest differ only
        # in the rounding procedure, we don't need to re-compute the interior
        # point solution every time.
        madow_times_rie.append(solve_time_rie + (end - start) - rtime_rie)

        # Solve the relaxed maximum algebraic connectivity augmentation problem.
        start = timer()
        result, unrounded, upper, rtime = mac.solve(num_lc, fw_init, max_iters=20, rounding="nearest", verbose=True, return_rounding_time=True, use_cache=False)
        end = timer()
        solve_time = end - start
        times.append(solve_time)
        results.append(result)
        upper_bounds.append(upper)
        unrounded_results.append(unrounded)

        start = timer()
        madow_rounded = round_madow(unrounded, num_lc, seed=np.random.RandomState(42))
        end = timer()
        madow_results.append(madow_rounded)
        # Time for Madow rounded solution is total MAC time (including nearest
        # neighbor rounding) plus the time to perform Madow rounding, minus the
        # nearest neighbor rounding time. Because Madow and nearest differ only
        # in the rounding procedure, we don't need to re-compute the interior
        # point solution every time.
        madow_times.append(solve_time + (end - start) - rtime)


        rie_log = mac.optimizer._log
        rie_costs = -1*np.array(rie_log['iterations']['cost'])
        rie_iterations = rie_log['iterations']['iteration']

        mac_log = mac.mac_log
        mac_costs = mac_log['iterations']['cost']
        mac_iterations = mac_log['iterations']['iteration']

        costs_per_iteration_mac.append(mac_costs)
        costs_per_iteration_rie.append(rie_costs)

        num_iterations_mac.append(len(mac_iterations))
        num_iterations_rie.append(len(rie_iterations))

        mac_fiedler_iters.append(mac.prob_run_info['fiedler_iters'])
        mac_fiedler_times.append(mac.prob_run_info['fiedler_runtimes'])
        mac_fiedler_ratio.append(mac.prob_run_info['ratio'])

        rie_fiedler_iters.append(mac.prob_rie_run_info['fiedler_iters'])
        rie_fiedler_times.append(mac.prob_rie_run_info['fiedler_runtimes'])
        rie_fiedler_ratio.append(mac.prob_rie_run_info['ratio'])

    import pickle
    import tempfile
    output_dict = {
        "costs_per_iteration_mac": costs_per_iteration_mac,
        "costs_per_iteration_rie": costs_per_iteration_rie,
        "num_iterations_mac": num_iterations_mac,
        "num_iterations_rie": num_iterations_rie
    }

    f = tempfile.NamedTemporaryFile(delete=False)
    pickle.dump(output_dict, f)

    # Solve the greedy k-edge selection problem
    if run_greedy:
        num_lcs = [int(pct_lc * len(lc_measurements)) for pct_lc in percent_lc]
        greedy_esp_results, _, greedy_esp_times = greedy_esp.subsets_lazy(num_lcs, verbose=True)
        pass
    '''
    # Display the algebraic connectivity for each method
    for i in range(len(naive_results)):
        pct_lc = percent_lc[i]
        print(f"Naive AC at {pct_lc * 100.0} % loop closures: {mac.evaluate_objective(naive_results[i])}")
        print(f"MAC AC at {pct_lc * 100.0} % loop closures: {mac.evaluate_objective(results[i])}")
        print(f"MAC unrounded AC at {pct_lc * 100.0} % loop closures: {mac.evaluate_objective(unrounded_results[i])}")
        print(f"Dual at {pct_lc * 100.0} % loop closures: {upper_bounds[i]}")

        print(f"Rie AC at {pct_lc * 100.0} % loop closures: {mac.evaluate_objective(results_rie[i])}")
        print(f"Rie unrounded AC at {pct_lc * 100.0} % loop closures: {mac.evaluate_objective(unrounded_results_rie[i])}")
        if run_greedy:
            # print(f"Greedy Eig AC at {pct_lc * 100.0} % loop closures: {mac.evaluate_objective(greedy_eig_results[i])}")
            print(f"Greedy ESP AC at {pct_lc * 100.0} % loop closures: {mac.evaluate_objective(greedy_esp_results[i])}")
            pass
        pass
    '''

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
    # plt.savefig(f"comp_time_{dataset_name}_300.png", dpi=300, bbox_inches='tight')
    # plt.savefig(f"comp_time_{dataset_name}.svg", transparent=True, bbox_inches='tight')

    '''
    #############################
    # Run SE-Sync
    #############################

    # SE-Sync Setup
    d = odom_measurements[0].R.shape[0]
    opts = PySESync.SESyncOpts()
    opts.num_threads = 4
    opts.verbose=True
    opts.r0 = d + 1  # Start at level d + 1 of the Riemannian Staircase

    sesync_results = []
    sesync_rie_results = []
    sesync_madow = []
    sesync_rie_madow = []
    sesync_naive = []
    sesync_eig = []
    sesync_esp = []
    for i in range(len(naive_results)):
        our_selected_lc = select_measurements(lc_measurements, results[i])
        our_meas = odom_measurements + our_selected_lc
        sesync_our_meas = to_sesync_format(our_meas)
        sesync_result = PySESync.SESync(sesync_our_meas, opts)
        sesync_results.append(sesync_result)

        madow_selected_lc = select_measurements(lc_measurements, madow_results[i])
        madow_meas = odom_measurements + madow_selected_lc
        sesync_result_madow = PySESync.SESync(to_sesync_format(madow_meas), opts)
        sesync_madow.append(sesync_result_madow)

        naive_selected_lc = select_measurements(lc_measurements, naive_results[i])
        naive_meas = odom_measurements + naive_selected_lc
        sesync_result_naive = PySESync.SESync(to_sesync_format(naive_meas), opts)
        sesync_naive.append(sesync_result_naive)

        rie_selected_lc = select_measurements(lc_measurements, results_rie[i])
        rie_meas = odom_measurements + rie_selected_lc
        sesync_rie_meas = to_sesync_format(rie_meas)
        sesync_rie_result = PySESync.SESync(sesync_rie_meas, opts)
        sesync_rie_results.append(sesync_rie_result)

        madow_rie_selected_lc = select_measurements(lc_measurements, madow_results_rie[i])
        madow_rie_meas = odom_measurements + madow_rie_selected_lc
        sesync_rie_result_madow = PySESync.SESync(to_sesync_format(madow_rie_meas), opts)
        sesync_rie_madow.append(sesync_rie_result_madow)

        if run_greedy:
            esp_selected_lc = select_measurements(lc_measurements, greedy_esp_results[i])
            esp_meas = odom_measurements + esp_selected_lc
            sesync_result_esp = PySESync.SESync(to_sesync_format(esp_meas), opts)
            sesync_esp.append(sesync_result_esp)

    plt.figure()
    plt.plot(100.0*np.array(percent_lc), [res.total_computation_time for res in sesync_results], label='MAC Nearest', marker='s', color=colors["MAC Nearest (Ours)"])
    plt.plot(100.0*np.array(percent_lc), [res.total_computation_time for res in sesync_madow], label='MAC Madow', marker='o', color=colors["MAC Madow (Ours)"])

    plt.plot(100.0*np.array(percent_lc), [res.total_computation_time for res in sesync_rie_results], label='Rie Nearest (Ours)', marker='s', color=colors["Rie Nearest (Ours)"])
    plt.plot(100.0*np.array(percent_lc), [res.total_computation_time for res in sesync_rie_madow], label='Rie Madow (Ours)', marker='o', color=colors["Rie Madow (Ours)"])

    if run_greedy:
        plt.plot(100.0*np.array(percent_lc), [res.total_computation_time for res in sesync_esp], label='Greedy ESP', marker='o', color=colors["Greedy ESP"])
    plt.plot(100.0*np.array(percent_lc), [res.total_computation_time for res in sesync_naive], label='Naive Method', marker='o', color=colors["Naive Method"])
    plt.legend()
    plt.xlabel(r'\% Edges Added')
    plt.ylabel(r'Time (s)')
    plt.savefig(f"sesync_comp_time_{dataset_name}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"sesync_comp_time_{dataset_name}_300.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"sesync_comp_time_{dataset_name}.svg", transparent=True, bbox_inches='tight')
    # plt.show()

    # Compute the SE-Sync cost for each solution under the *full* set of measurements
    M = construct_sesync_quadratic_form_matrix(measurements)
    LGrho = construct_LGrho(measurements)
    sesync_full = PySESync.SESync(to_sesync_format(measurements), opts)

    our_rot_costs = []
    our_full_costs = []
    our_SOd_orbdists = []
    our_ate_trans = []
    our_rpe_rots = []

    madow_rot_costs = []
    madow_full_costs = []
    madow_SOd_orbdists = []
    madow_ate_trans = []
    madow_rpe_rots = []

    rie_rot_costs = []
    rie_full_costs = []
    rie_SOd_orbdists = []
    rie_ate_trans = []
    rie_rpe_rots = []

    madow_rie_rot_costs = []
    madow_rie_full_costs = []
    madow_rie_SOd_orbdists = []
    madow_rie_ate_trans = []
    madow_rie_rpe_rots = []

    naive_rot_costs = []
    naive_full_costs = []
    naive_SOd_orbdists = []
    naive_ate_trans = []
    naive_rpe_rots = []

    esp_rot_costs = []
    esp_full_costs = []
    esp_SOd_orbdists = []
    esp_ate_trans = []
    esp_rpe_rots = []
    for i in range(len(percent_lc)):
        print(f"Percent LC: {percent_lc[i]}")

        xhat_ours = sesync_results[i].xhat
        xhat_madow = sesync_madow[i].xhat
        xhat_rie = sesync_rie_results[i].xhat
        xhat_madow_rie = sesync_rie_madow[i].xhat
        xhat_naive = sesync_naive[i].xhat

        our_selected_lc = select_measurements(lc_measurements, results[i])
        our_meas = odom_measurements + our_selected_lc

        madow_selected_lc = select_measurements(lc_measurements, madow_results[i])
        madow_meas = odom_measurements + madow_selected_lc

        rie_selected_lc = select_measurements(lc_measurements, results_rie[i])
        rie_meas = odom_measurements + rie_selected_lc

        madow_rie_selected_lc = select_measurements(lc_measurements, madow_results_rie[i])
        madow_rie_meas = odom_measurements + madow_rie_selected_lc

        naive_selected_lc = select_measurements(lc_measurements, naive_results[i])
        naive_meas = odom_measurements + naive_selected_lc

        our_rot_cost = evaluate_sesync_rotation_objective(LGrho, xhat_ours[:, num_poses:])
        our_full_cost = evaluate_sesync_objective(M, xhat_ours)
        our_SOd_orbdist = orbit_distance_dS(sesync_full.xhat[:,num_poses:], xhat_ours[:,num_poses:])
        our_ate_tran = poses_ate_tran(xhat_ours, sesync_full.xhat)
        our_rpe_rot = poses_rpe_rot(xhat_ours, sesync_full.xhat)

        our_rot_costs.append(our_rot_cost)
        our_full_costs.append(our_full_cost)
        our_SOd_orbdists.append(our_SOd_orbdist)
        our_ate_trans.append(our_ate_tran)
        our_rpe_rots.append(our_rpe_rot)


        plot_poses(xhat_ours, our_meas, show=False, color=colors["MAC Nearest (Ours)"])
        plt.savefig(f"ours_{dataset_name}_{str(percent_lc[i])}.png", dpi=600)
        plt.savefig(f"ours_{dataset_name}_{str(percent_lc[i])}_300.png", dpi=300)
        plt.savefig(f"ours_{dataset_name}_{str(percent_lc[i])}.svg", transparent=True)
        plt.close()

        rie_rot_cost = evaluate_sesync_rotation_objective(LGrho, xhat_rie[:, num_poses:])
        rie_full_cost = evaluate_sesync_objective(M, xhat_rie)
        rie_SOd_orbdist = orbit_distance_dS(sesync_full.xhat[:,num_poses:], xhat_rie[:,num_poses:])
        rie_ate_tran = poses_ate_tran(xhat_rie, sesync_full.xhat)
        rie_rpe_rot = poses_rpe_rot(xhat_rie, sesync_full.xhat)

        rie_rot_costs.append(rie_rot_cost)
        rie_full_costs.append(rie_full_cost)
        rie_SOd_orbdists.append(rie_SOd_orbdist)
        rie_ate_trans.append(rie_ate_tran)
        rie_rpe_rots.append(rie_rpe_rot)

        plot_poses(xhat_rie, rie_meas, show=False, color=colors["Rie Nearest (Ours)"])
        plt.savefig(f"rie_{dataset_name}_{str(percent_lc[i])}.png", dpi=600)
        plt.savefig(f"rie_{dataset_name}_{str(percent_lc[i])}_300.png", dpi=300)
        plt.savefig(f"rie_{dataset_name}_{str(percent_lc[i])}.svg", transparent=True)
        plt.close()

        # Print error for our method
        # print(f"Our rotation cost: {our_rot_cost}")
        # print(f"Our full cost {our_full_cost}")
        # print(f"Our SO orbdist: {our_SOd_orbdist}")

        madow_rot_cost = evaluate_sesync_rotation_objective(LGrho, xhat_madow[:, num_poses:])
        madow_full_cost = evaluate_sesync_objective(M, xhat_madow)
        madow_SOd_orbdist = orbit_distance_dS(sesync_full.xhat[:,num_poses:], xhat_madow[:,num_poses:])
        madow_ate_tran = poses_ate_tran(xhat_madow, sesync_full.xhat)
        madow_rpe_rot = poses_rpe_rot(xhat_madow, sesync_full.xhat)

        madow_rot_costs.append(madow_rot_cost)
        madow_full_costs.append(madow_full_cost)
        madow_SOd_orbdists.append(madow_SOd_orbdist)
        madow_ate_trans.append(madow_ate_tran)
        madow_rpe_rots.append(madow_rpe_rot)


        plot_poses(xhat_madow, madow_meas, show=False, color=colors["MAC Madow (Ours)"])
        plt.savefig(f"madow_{dataset_name}_{str(percent_lc[i])}.png", dpi=600)
        plt.savefig(f"madow_{dataset_name}_{str(percent_lc[i])}_300.png", dpi=300)
        plt.savefig(f"madow_{dataset_name}_{str(percent_lc[i])}.svg", transparent=True)
        plt.close()

        madow_rie_rot_cost = evaluate_sesync_rotation_objective(LGrho, xhat_madow_rie[:, num_poses:])
        madow_rie_full_cost = evaluate_sesync_objective(M, xhat_madow_rie)
        madow_rie_SOd_orbdist = orbit_distance_dS(sesync_full.xhat[:,num_poses:], xhat_madow_rie[:,num_poses:])
        madow_rie_ate_tran = poses_ate_tran(xhat_madow_rie, sesync_full.xhat)
        madow_rie_rpe_rot = poses_rpe_rot(xhat_madow_rie, sesync_full.xhat)

        madow_rie_rot_costs.append(madow_rie_rot_cost)
        madow_rie_full_costs.append(madow_rie_full_cost)
        madow_rie_SOd_orbdists.append(madow_rie_SOd_orbdist)
        madow_rie_ate_trans.append(madow_rie_ate_tran)
        madow_rie_rpe_rots.append(madow_rie_rpe_rot)

        plot_poses(xhat_madow_rie, madow_rie_meas, show=False, color=colors["Rie Madow (Ours)"])
        plt.savefig(f"madow_rie_{dataset_name}_{str(percent_lc[i])}.png", dpi=600)
        plt.savefig(f"madow_rie_{dataset_name}_{str(percent_lc[i])}_300.png", dpi=300)
        plt.savefig(f"madow_rie_{dataset_name}_{str(percent_lc[i])}.svg", transparent=True)
        plt.close()

        naive_rot_cost = evaluate_sesync_rotation_objective(LGrho, xhat_naive[:, num_poses:])
        naive_full_cost = evaluate_sesync_objective(M, xhat_naive)
        naive_SOd_orbdist = orbit_distance_dS(sesync_full.xhat[:,num_poses:], xhat_naive[:,num_poses:])
        naive_ate_tran = poses_ate_tran(xhat_naive, sesync_full.xhat)
        naive_rpe_rot = poses_rpe_rot(xhat_naive, sesync_full.xhat)

        naive_rot_costs.append(naive_rot_cost)
        naive_full_costs.append(naive_full_cost)
        naive_SOd_orbdists.append(naive_SOd_orbdist)
        naive_ate_trans.append(naive_ate_tran)
        naive_rpe_rots.append(naive_rpe_rot)

        plot_poses(xhat_naive, naive_meas, show=False, color=colors["Naive Method"])
        plt.savefig(f"naive_{dataset_name}_{str(percent_lc[i])}.png", dpi=600)
        plt.savefig(f"naive_{dataset_name}_{str(percent_lc[i])}_300.png", dpi=300)
        plt.savefig(f"naive_{dataset_name}_{str(percent_lc[i])}.svg", transparent=True)
        plt.close()

        # Print Error for naive method
        # print(f"Naive rotation cost: {naive_rot_cost}")
        # print(f"Naive full cost: {naive_full_cost}")
        # print(f"Naive SO orbdist: {naive_SOd_orbdist}")

        if run_greedy:
            xhat_esp = sesync_esp[i].xhat
            esp_selected_lc = select_measurements(lc_measurements, greedy_esp_results[i])
            esp_meas = odom_measurements + esp_selected_lc

            esp_rot_cost = evaluate_sesync_rotation_objective(LGrho, xhat_esp[:, num_poses:])
            esp_full_cost = evaluate_sesync_objective(M, xhat_esp)
            esp_SOd_orbdist = orbit_distance_dS(sesync_full.xhat[:,num_poses:], xhat_esp[:,num_poses:])
            esp_ate_tran = poses_ate_tran(xhat_esp, sesync_full.xhat)
            esp_rpe_rot = poses_rpe_rot(xhat_esp, sesync_full.xhat)

            esp_rot_costs.append(esp_rot_cost)
            esp_full_costs.append(esp_full_cost)
            esp_SOd_orbdists.append(esp_SOd_orbdist)
            esp_ate_trans.append(esp_ate_tran)
            esp_rpe_rots.append(esp_rpe_rot)

            plot_poses(xhat_esp, esp_meas, show=False, color=colors["Greedy ESP"])
            plt.savefig(f"esp_{dataset_name}_{str(percent_lc[i])}.png", dpi=600)
            plt.savefig(f"esp_{dataset_name}_{str(percent_lc[i])}_300.png", dpi=300)
            plt.savefig(f"esp_{dataset_name}_{str(percent_lc[i])}.svg", transparent=True)
            plt.close()


    plt.figure()
    plt.plot(100.0*np.array(percent_lc), our_ate_trans, label='MAC Nearest', marker='s', color=colors["MAC Nearest (Ours)"])
    plt.plot(100.0*np.array(percent_lc), madow_ate_trans, label='MAC Madow', marker='o', color=colors["MAC Madow (Ours)"])
    plt.plot(100.0*np.array(percent_lc), rie_ate_trans, label='Rie Nearest (Ours)', marker='s', color=colors["Rie Nearest (Ours)"])
    plt.plot(100.0*np.array(percent_lc), madow_rie_ate_trans, label='Rie Madow (Ours)', marker='o', color=colors["Rie Madow (Ours)"])
    if run_greedy:
        plt.plot(100.0*np.array(percent_lc), esp_ate_trans, label='Greedy ESP', marker='o', color=colors["Greedy ESP"])
    plt.plot(100.0*np.array(percent_lc), naive_ate_trans, label='Naive Method', marker='o', color=colors["Naive Method"])
    plt.xlabel(r'\% Edges Added')
    plt.ylabel(r'ATE Translation [m]')
    plt.legend()
    plt.savefig(f"ate_tran_{dataset_name}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"ate_tran_{dataset_name}_300.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"ate_tran_{dataset_name}.svg", transparent=True, bbox_inches='tight')
    # plt.show()

    plt.figure()
    plt.semilogy(100.0*np.array(percent_lc[:-1]), our_ate_trans[:-1], label='MAC Nearest', marker='s', color=colors["MAC Nearest (Ours)"])
    plt.semilogy(100.0*np.array(percent_lc[:-1]), madow_ate_trans[:-1], label='MAC Madow', marker='o', color=colors["MAC Madow (Ours)"])
    plt.semilogy(100.0*np.array(percent_lc[:-1]), rie_ate_trans[:-1], label='Rie Nearest (Ours)', marker='s', color=colors["Rie Nearest (Ours)"])
    plt.semilogy(100.0*np.array(percent_lc[:-1]), madow_rie_ate_trans[:-1], label='Rie Madow (Ours)', marker='o', color=colors["Rie Madow (Ours)"])
    if run_greedy:
        plt.semilogy(100.0*np.array(percent_lc[:-1]), esp_ate_trans[:-1], label='Greedy ESP', marker='o', color=colors["Greedy ESP"])
    plt.semilogy(100.0*np.array(percent_lc[:-1]), naive_ate_trans[:-1], label='Naive Method', marker='o', color=colors["Naive Method"])
    plt.xlabel(r'\% Edges Added')
    plt.ylabel(r'ATE Translation [m]')
    plt.xlim([0.0, 100.0])
    plt.legend()
    plt.savefig(f"ate_tran_{dataset_name}_log.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"ate_tran_{dataset_name}_log_300.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"ate_tran_{dataset_name}_log.svg", transparent=True, bbox_inches='tight')

    plt.figure()
    plt.plot(100.0*np.array(percent_lc), our_rpe_rots, label='MAC Nearest', marker='s', color=colors["MAC Nearest (Ours)"])
    plt.plot(100.0*np.array(percent_lc), madow_rpe_rots, label='MAC Madow', marker='o', color=colors["MAC Madow (Ours)"])
    plt.plot(100.0*np.array(percent_lc), rie_rpe_rots, label='Rie Nearest (Ours)', marker='s', color=colors["Rie Nearest (Ours)"])
    plt.plot(100.0*np.array(percent_lc), madow_rie_rpe_rots, label='Rie Madow (Ours)', marker='o', color=colors["Rie Madow (Ours)"])
    if run_greedy:
        plt.plot(100.0*np.array(percent_lc), esp_rpe_rots, label='Greedy ESP', marker='o', color=colors["Greedy ESP"])
    plt.plot(100.0*np.array(percent_lc), naive_rpe_rots, label='Naive Method', marker='o', color=colors["Naive Method"])
    plt.xlabel(r'\% Edges Added')
    plt.ylabel(r'RPE Rotation [deg]')
    plt.legend()
    plt.savefig(f"rpe_rot_{dataset_name}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"rpe_rot_{dataset_name}_300.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"rpe_rot_{dataset_name}.svg", transparent=True, bbox_inches='tight')
    # plt.show()

    plt.figure()
    plt.semilogy(100.0*np.array(percent_lc[:-1]), our_rpe_rots[:-1], label='MAC Nearest', marker='s', color=colors["MAC Nearest (Ours)"])
    plt.semilogy(100.0*np.array(percent_lc[:-1]), madow_rpe_rots[:-1], label='MAC Madow', marker='o', color=colors["MAC Madow (Ours)"])
    plt.semilogy(100.0*np.array(percent_lc[:-1]), rie_rpe_rots[:-1], label='Rie Nearest (Ours)', marker='s', color=colors["Rie Nearest (Ours)"])
    plt.semilogy(100.0*np.array(percent_lc[:-1]), madow_rie_rpe_rots[:-1], label='Rie Madow (Ours)', marker='o', color=colors["Rie Madow (Ours)"])
    if run_greedy:
        plt.semilogy(100.0*np.array(percent_lc[:-1]), esp_rpe_rots[:-1], label='Greedy ESP', marker='o', color=colors["Greedy ESP"])
    plt.semilogy(100.0*np.array(percent_lc[:-1]), naive_rpe_rots[:-1], label='Naive Method', marker='o', color=colors["Naive Method"])
    plt.xlabel(r'\% Edges Added')
    plt.ylabel(r'RPE Rotation [deg]')
    plt.xlim([0.0, 100.0])
    plt.legend()
    plt.savefig(f"rpe_rot_{dataset_name}_log.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"rpe_rot_{dataset_name}_log_300.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"rpe_rot_{dataset_name}_log.svg", transparent=True, bbox_inches='tight')
    # plt.show()

    plt.figure()
    plt.semilogy(100.0*np.array(percent_lc), our_full_costs, label='MAC Nearest', marker='s', color=colors["MAC Nearest (Ours)"])
    plt.semilogy(100.0*np.array(percent_lc), madow_full_costs, label='MAC Madow', marker='o', color=colors["MAC Madow (Ours)"])
    plt.semilogy(100.0*np.array(percent_lc), rie_full_costs, label='Rie Nearest (Ours)', marker='s', color=colors["Rie Nearest (Ours)"])
    plt.semilogy(100.0*np.array(percent_lc), madow_rie_full_costs, label='Rie Madow (Ours)', marker='o', color=colors["Rie Madow (Ours)"])
    if run_greedy:
        plt.semilogy(100.0*np.array(percent_lc), esp_full_costs, label='Greedy ESP', marker='o', color=colors["Greedy ESP"])
    plt.semilogy(100.0*np.array(percent_lc), naive_full_costs, label='Naive Method', marker='o', color=colors["Naive Method"])
    plt.xlabel(r'\% Edges Added')
    plt.ylabel(r'Objective Value')
    plt.legend()
    plt.savefig(f"obj_val_{dataset_name}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"obj_val_{dataset_name}_300.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"obj_val_{dataset_name}.svg", transparent=True, bbox_inches='tight')
    # plt.show()

    plt.figure()
    plt.plot(100.0*np.array(percent_lc), our_SOd_orbdists, label='MAC Nearest', marker='s', color=colors["MAC Nearest (Ours)"])
    plt.plot(100.0*np.array(percent_lc), madow_SOd_orbdists, label='MAC Madow', marker='o', color=colors["MAC Madow (Ours)"])
    plt.plot(100.0*np.array(percent_lc), rie_SOd_orbdists, label='Rie Nearest (Ours)', marker='s', color=colors["Rie Nearest (Ours)"])
    plt.plot(100.0*np.array(percent_lc), madow_rie_SOd_orbdists, label='Rie Madow (Ours)', marker='o', color=colors["Rie Madow (Ours)"])
    if run_greedy:
        plt.plot(100.0*np.array(percent_lc), esp_SOd_orbdists, label='Greedy ESP', marker='o', color=colors["Greedy ESP"])
    plt.plot(100.0*np.array(percent_lc), naive_SOd_orbdists, label='Naive Method', marker='o', color=colors["Naive Method"])
    plt.ylabel(r'$\mathrm{SO}(d)$ orbit distance')
    plt.xlabel(r'\% Edges Added')
    plt.legend()
    plt.savefig(f"orbdist_{dataset_name}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"orbdist_{dataset_name}_300.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"orbdist_{dataset_name}.svg", transparent=True, bbox_inches='tight')
    # plt.show()


    # Extract translational states from solution xhat
    # xhat = sesync_result.xhat
    # print(xhat.shape)
    # R0inv = np.linalg.inv(xhat[:, num_poses : num_poses + d])
    # t = np.matmul(R0inv, xhat[:, 0:num_poses])

    # print("Our rotation cost: ", evaluate_sesync_rotation_objective(LGrho, xhat[:, num_poses:]))
    # print("Our cost: ", evaluate_sesync_objective(M, xhat))

    # plot_poses(xhat, our_meas)

    # xhat = sesync_result_naive.xhat
    # R0inv = np.linalg.inv(xhat[:, num_poses : num_poses + d])
    # t = np.matmul(R0inv, xhat[:, 0:num_poses])

    # print("Naive rotation cost: ", evaluate_sesync_rotation_objective(LGrho, xhat[:, num_poses:]))
    # print("Naive cost: ", evaluate_sesync_objective(M, xhat))

'''
# plt.show()