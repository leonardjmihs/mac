import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mac.utils.graphs import select_edges, Edge
from mac.utils.conversions import nx_to_mac, mac_to_nx
from mac.solvers import MAC, MACH, NaiveGreedy, MACHR
import cProfile 
import time
import json

seed = 2002
np.random.seed(seed)
# np.random.seed(2002)

rie_alg = "rgd"
# rie_alg = "rbfgs"


mac_times = []
rie_times= []
costs_unrounded_mac= []
costs_unrounded_rie= []
costs_unrounded_hybrid= []
costs_rounded_mac = []
costs_rounded_rie= []
costs_rounded_hybrid= []
num_candidates_list = []
num_total_list = []
output_dict_list = []

costs_per_iteration_mac = []
costs_per_iteration_rie = []
costs_per_iteration_hybrid = []

num_iterations_mac = []
num_iterations_rie = []
num_iterations_hybrid = []

prob_mac_runtimes = []
prob_rie_runtimes = []
hybrid_times = []

mac_solutions = []
rie_solutions = []
hybrid_solutions = []

mac_fiedler_times = []
mac_fiedler_iters = []
mac_fiedler_ratio= []

rie_fiedler_times = []
rie_fiedler_iters = []
rie_fiedler_ratio = []

# for n in np.logspace(1.5, 2.5, 20).astype(int):
# for n in np.logspace(1, 2, 20).astype(int):
# for n in np.logspace(2, 2, 1).astype(int):
# for n in [100, 201, 302, 403]:
# for n in [100, 201, 302]:
# for n in [200, 201, 202, 203]:
# for n in [300]:
# for n in [30, 31, 32, 33]:
# for n in [10,30, 200]:
for n in [10]:
    randomizer_state = list(np.random.get_state())
    randomizer_state[1] = randomizer_state[1].tolist()

    p = 0.6
    G = nx.erdos_renyi_graph(n, p,seed=np.random)
    prob_def = {'randomizer_state': randomizer_state, "seed":seed, "n":n, "p":p}
    output_dict_list.append(prob_def)
    seed +=1


    # Add a chain
    for i in range(n-1):
        if G.has_edge(i+1, i):
            G.remove_edge(i+1, i)
        if not G.has_edge(i, i+1):
            G.add_edge(i, i+1)

    # Split the graph into a tree part and a "loop" part
    spanning_tree = nx.minimum_spanning_tree(G)
    loop_graph = nx.difference(G, spanning_tree)
    # set

    # nx.draw(G)
    # plt.title("Original Graph")
    # plt.show()

    # Ensure G is connected before proceeding
    assert(nx.is_connected(G))


    fixed_edges = nx_to_mac(spanning_tree)
    candidate_edges = nx_to_mac(loop_graph)
    pct_candidates = 0.2
    num_candidates = int(pct_candidates * len(candidate_edges))

    # Save randomizer state and graph to a file
    np.savez(f"graph_sparse_{n}.npz", G=G, fixed_edges=fixed_edges, candidate_edges=candidate_edges)

    print(f"Num Candidates: {num_candidates}")
    print(f"Length Candidate Edges: {len(candidate_edges)}")
    num_candidates_list.append(num_candidates)
    num_total_list.append(len(candidate_edges))

    mac = MAC(fixed_edges, candidate_edges, n, rie_alg=rie_alg)
    mach = MACH(fixed_edges, candidate_edges, n, rie_alg=rie_alg)
    naive = NaiveGreedy(candidate_edges)

    # w_init = np.zeros(len(candidate_edges))
    # w_init[:num_candidates] = 1.0
    # np.random.shuffle(w_init)

    rie_init = np.ones(len(candidate_edges)) * num_candidates/ len(candidate_edges)
    # fw_init = naive.subset(num_candidates)

    fw_init = rie_init
    mac.evaluate_objective(fw_init)
    # rie_init = fw_init


    rie_profiler = cProfile.Profile()
    mac_profiler = cProfile.Profile()
    hybrid_profiler = cProfile.Profile()
    # Allow stuff to jit
    mac.optimizer.reset_log()

    start_time = time.time()
    rie_profiler.enable()
    result_rie, unrounded_rie, upper_rie = mac.solve_rie(num_candidates, rie_init.copy(), max_iters=100, rounding="madow", use_cache=True, verbose=True, min_relative_cost_reduction=0.0)
    rie_profiler.disable()
    rie_profiler.dump_stats(f'graph_sparse_rie_{rie_alg}.prof')
    end_time = time.time()
    time_rie = end_time - start_time

    start_time = time.time()
    mac_profiler.enable()
    result_mac, unrounded_mac, upper_mac = mac.solve(num_candidates, fw_init.copy(), max_iters=100, rounding="madow", use_cache=True, verbose=True)
    mac_profiler.disable()
    mac_profiler.dump_stats('graph_sparse_mac.prof')
    end_time = time.time()
    time_mac = end_time - start_time

    # start_time = time.time()
    # hybrid_profiler.enable()
    # result_h, unrounded_h, upper_h = mach.solve(num_candidates, rie_init.copy(), max_iters=100, rounding="madow", use_cache=True, min_relative_cost_reduction=0.0)
    # hybrid_profiler.disable()
    # hybrid_profiler.dump_stats('graph_sparse_mac.prof')
    # end_time = time.time()
    # time_hybrid = end_time - start_time

    mac_solutions.append(unrounded_mac)
    rie_solutions.append(unrounded_rie)
    # hybrid_solutions.append(unrounded_h)

    cost_mac_unrounded = mac.evaluate_objective(unrounded_mac)
    cost_rie_unrounded = mac.evaluate_objective(unrounded_rie)
    # cost_hybrid_unrounded = mac.evaluate_objective(unrounded_h)

    cost_mac_rounded = mac.evaluate_objective(result_mac)
    cost_rie_rounded = mac.evaluate_objective(result_rie)
    # cost_hybrid_rounded = mac.evaluate_objective(result_h)

    mac_times.append(time_mac)
    rie_times.append(time_rie)
    # hybrid_times.append(time_hybrid)

    costs_unrounded_mac.append(cost_mac_unrounded)
    costs_unrounded_rie.append(cost_rie_unrounded)
    # costs_unrounded_hybrid.append(cost_hybrid_unrounded)

    costs_rounded_mac.append(cost_mac_rounded)
    costs_rounded_rie.append(cost_rie_rounded)
    # costs_rounded_hybrid.append(cost_hybrid_rounded)

    rie_log = mac.optimizer._log
    rie_costs = -1*np.array(rie_log['iterations']['cost'])
    rie_iterations = rie_log['iterations']['iteration']

    mac_log = mac.mac_log
    mac_costs = mac_log['iterations']['cost']
    mac_iterations = mac_log['iterations']['iteration']

    # mach_log = mach.mac_log
    # mach_costs = mach_log['iterations']['cost']
    # mach_iterations = mach_log['iterations']['iteration']

    costs_per_iteration_mac.append(mac_costs)
    costs_per_iteration_rie.append(rie_costs)
    # costs_per_iteration_hybrid.append(mach_costs)

    num_iterations_mac.append(len(mac_iterations))
    num_iterations_rie.append(len(rie_iterations))
    # num_iterations_hybrid.append(len(mach_iterations))
    
    prob_mac_runtimes.append(mac.prob_run_info['runtimes'])
    prob_rie_runtimes.append(mac.prob_rie_run_info['runtimes'])

    mac_fiedler_iters.append(mac.prob_run_info['fiedler_iters'])
    mac_fiedler_times.append(mac.prob_run_info['fiedler_runtimes'])
    mac_fiedler_ratio.append(mac.prob_run_info['ratio'])

    rie_fiedler_iters.append(mac.prob_rie_run_info['fiedler_iters'])
    rie_fiedler_times.append(mac.prob_rie_run_info['fiedler_runtimes'])
    rie_fiedler_ratio.append(mac.prob_rie_run_info['ratio'])

max_length = max

# Make a 3d plot that lots iterations vs cost vs num_total_list
# make the plot have a 3d view
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(num_total_list)):
    ax.plot([num_total_list[i]]*num_iterations_mac[i], 
            np.arange(num_iterations_mac[i]), 
            costs_per_iteration_mac[i], label="MAC", linestyle='--', marker='o', color='b')

    ax.plot([num_total_list[i]]*num_iterations_rie[i], 
            np.arange(num_iterations_rie[i]), 
            costs_per_iteration_rie[i], label="Riemannian", linestyle='--', marker='o', color='r')

    # ax.plot([num_total_list[i]]*num_iterations_hybrid[i], 
    #         np.arange(num_iterations_hybrid[i]), 
    #         costs_per_iteration_hybrid[i], label="Hybrid", linestyle='--', marker='o', color='g')

ax.set_xlabel("Num Total Edges")
ax.set_ylabel("Iterations")
ax.set_zlabel("Cost")
ax.set_title("Cost vs Num Total Edges and Iterations")
# ax.legend()
plt.savefig(f"{rie_alg}_cost_vs_edges_and_iterations.png")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(num_total_list)):
    num_calls_mac = len(prob_mac_runtimes[i])
    ax.plot([num_total_list[i]]*num_calls_mac, 
            np.arange(num_calls_mac), 
            prob_mac_runtimes[i], label="MAC", linestyle='--', marker='o', color='b')
    num_calls_rie = len(prob_rie_runtimes[i])
    ax.plot([num_total_list[i]]*num_calls_rie, 
            np.arange(num_calls_rie), 
            prob_rie_runtimes[i], label="Riemannian", linestyle='--', marker='o', color='r')
ax.set_xlabel("Num Total Edges")
ax.set_ylabel("Iterations")
ax.set_zlabel("Time")
ax.set_title("Function Call time vs Num Total Edges and Iterations")
# ax.legend()
plt.savefig(f"{rie_alg}_iter_runtimes_vs_edges_and_iterations.png")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(num_total_list)):
#     num_calls_mac = len(mac_fiedler_iters[i])
#     ax.plot([num_total_list[i]]*num_calls_mac, 
#             np.arange(num_calls_mac), 
#             mac_fiedler_times[i], label="MAC", linestyle='--', marker='o', color='b')

#     num_calls_rie = len(rie_fiedler_iters[i])
#     ax.plot([num_total_list[i]]*num_calls_rie, 
#             np.arange(num_calls_rie), 
#             rie_fiedler_times[i], label="Riemannian", linestyle='--', marker='o', color='r')
    
# ax.set_xlabel("Num Total Edges")
# ax.set_ylabel("Iterations")
# ax.set_zlabel("Fiedler Tracemin Average Time")
# ax.set_title("Fiedler Tracemin Loop Avg Time vs Num Total Edges and Iterations")
# plt.savefig(f"{rie_alg}_fiedler_loop_runtimes_vs_edges_and_iterations.png")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(num_total_list)):
#     num_calls_mac = len(mac_fiedler_iters[i])
#     ax.plot([num_total_list[i]]*num_calls_mac, 
#             np.arange(num_calls_mac), 
#             mac_fiedler_iters[i], label="MAC", linestyle='--', marker='o', color='b')

#     num_calls_rie = len(rie_fiedler_iters[i])
#     ax.plot([num_total_list[i]]*num_calls_rie, 
#             np.arange(num_calls_rie), 
#             rie_fiedler_iters[i], label="Riemannian", linestyle='--', marker='o', color='r')
    
# ax.set_xlabel("Num Total Edges")
# ax.set_ylabel("Iterations")
# ax.set_zlabel("Fiedler Tracemin Average Iterations")
# ax.set_title("Fiedler Tracemin Average Iterations vs Num Total Edges and Iterations")
# plt.savefig(f"{rie_alg}_fiedler_tracemeing_iters_vs_edges_and_iterations.png")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(num_total_list)):
#     num_calls_mac = len(mac_fiedler_iters[i])
#     ax.plot([num_total_list[i]]*num_calls_mac, 
#             np.arange(num_calls_mac), 
#             mac_fiedler_ratio[i], label="MAC", linestyle='--', marker='o', color='b')

#     num_calls_rie = len(rie_fiedler_iters[i])
#     ax.plot([num_total_list[i]]*num_calls_rie, 
#             np.arange(num_calls_rie), 
#             rie_fiedler_ratio[i], label="Riemannian", linestyle='--', marker='o', color='r')
    
# ax.set_xlabel("Num Total Edges")
# ax.set_ylabel("Iterations")
# ax.set_zlabel("Fiedler Tracemin ratio")
# ax.set_title("Fiedler Tracemin ratio vs Num Total Edges and Iterations")
# plt.savefig(f"{rie_alg}_fiedler_tracemeing_ratio_vs_edges_and_iterations.png")

# write lists to a dict
import pickle
import tempfile
output_dict = {
    "mac_times": mac_times,
    "rie_times": rie_times,
    "costs_unrounded_mac": costs_unrounded_mac,
    "costs_unrounded_rie": costs_unrounded_rie,
    "costs_rounded_mac": costs_rounded_mac,
    "costs_rounded_rie": costs_rounded_rie,
    "num_candidates_list": num_candidates_list,
    "num_total_list": num_total_list,
    "output_dict_list": output_dict_list,
    "costs_per_iteration_mac": costs_per_iteration_mac,
    "costs_per_iteration_rie": costs_per_iteration_rie,
    "num_iterations_mac": num_iterations_mac,
    "num_iterations_rie": num_iterations_rie
}

f = tempfile.NamedTemporaryFile(delete=False)
pickle.dump(output_dict, f)

# write output_dict_list to file
# output_file = open("problem_specs.json", 'w', encoding='utf-8')
# for dic in output_dict_list:
#     json.dump(dic, output_file)
#     output_file.write('\n')

# mac_times = np.array(mac_times)
# rie_times = np.array(rie_times)
# costs_unrounded_mac = np.array(costs_unrounded_mac)
# costs_unrounded_rie = np.array(costs_unrounded_rie)
# costs_rounded_mac = np.array(costs_rounded_mac)
# costs_rounded_rie = np.array(costs_rounded_rie)

# num_candidates_list = np.array(num_candidates_list)
# num_total_list = np.array(num_total_list)

# # Dump above 6 arrays to file
# data = {'MAC': [mac_times, costs_unrounded_mac, costs_rounded_mac],
#         'Rie': [rie_times, costs_unrounded_rie, costs_rounded_rie]}
# indices = ["time", "unrounded cost", "rounded cost"]
# df = pd.DataFrame(data, indices)
# df.to_csv("graph_sparse_time.csv")

# plot the mac times and rie_times against the num_total_list
plt.figure()
plt.plot(num_total_list, mac_times, linestyle='--', marker='o', color="b", label="MAC")
plt.plot(num_total_list, rie_times, linestyle='--', marker='o', color="r", label="Riemannian")
# plt.plot(num_total_list, hybrid_times, linestyle='--', marker='o', color="g", label="Riemannian")
plt.xlabel("Num Total Edges")
plt.ylabel("Time (s)")
plt.title("Time vs Num Total Edges")
plt.legend()
plt.savefig(f"{rie_alg}_edges_vs_time.png")

# plt.figure()
# plt.plot(num_total_list, mac_times, label="MAC")
# plt.plot(num_total_list, rie_times, label="Riemannian")
# plt.xlabel("Num Total Edges")
# plt.ylabel("Time (s)")
# plt.legend()

# plot the costs_unrounded_mac and costs_unrounded_rie against the num_total_list
plt.figure()
plt.plot(num_total_list, costs_unrounded_mac, linestyle='--', marker='o',color="b",label="MAC Unrounded")
plt.plot(num_total_list, costs_unrounded_rie, linestyle='--', marker='o',color="r",label="Riemannian Unrounded")
# plt.plot(num_total_list, costs_unrounded_hybrid, linestyle='--', marker='o',color="g",label="Riemannian Unrounded")
plt.xlabel("Num Total Edges")
plt.ylabel("Cost Unrounded")
plt.title("Unrounded Cost vs Num Total Edges")
plt.legend()
plt.savefig(f"{rie_alg}_unrounded_vs_edges.png")

# plot the number of iterations against the num_total_list
plt.figure()
plt.plot(num_total_list, num_iterations_mac, linestyle='--', marker='o', color="b",label="MAC")
plt.plot(num_total_list, num_iterations_rie, linestyle='--', marker='o', color="r",label="Riemannian")
# plt.plot(num_total_list, num_iterations_hybrid, linestyle='--', marker='o', color="g",label="Riemannian")
plt.xlabel("Num Total Edges")
plt.ylabel("Num Iterations")
plt.title("Number of Iterations vs Num Total Edges")
plt.legend()
plt.savefig(f"{rie_alg}_iteration_vs_edges.png")
plt.show()