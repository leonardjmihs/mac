import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mac.utils.graphs import select_edges, Edge
from mac.utils.conversions import nx_to_mac, mac_to_nx
from mac.solvers import MAC
import cProfile 
import time

seed = 2002
np.random.seed(seed)
# np.random.seed(2002)
mac_times = []
rie_times= []

costs_unrounded_mac= []
costs_unrounded_rie= []

costs_rounded_mac = []
costs_rounded_rie= []

num_candidates_list = []
num_total_list = []

# for n in np.logspace(1.5, 2.5, 20).astype(int):
# for n in np.logspace(2, 2, 1).astype(int):
for n in [20]:
    p = 0.6
    G = nx.erdos_renyi_graph(n, p,seed=seed)
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
    print(f"Num Candidates: {num_candidates}")
    print(f"Length Candidate Edges: {len(candidate_edges)}")
    num_candidates_list.append(num_candidates)
    num_total_list.append(len(candidate_edges))

    mac = MAC(fixed_edges, candidate_edges, n)

    # w_init = np.zeros(len(candidate_edges))
    # w_init[:num_candidates] = 1.0
    # np.random.shuffle(w_init)

    w_init = np.ones(len(candidate_edges)) * n/ len(candidate_edges)

    print(f"lambda2 Initial: {mac.evaluate_objective(w_init)}")


    rie_profiler = cProfile.Profile()
    mac_profiler = cProfile.Profile()
    # Allow stuff to jit
    result_rie, unrounded_rie, upper_rie = mac.solve_rie(num_candidates, w_init, max_iters=100, rounding="madow", use_cache=False)
    mac.optimizer.reset_log()

    start_time = time.time()

    rie_profiler.enable()
    result_rie, unrounded_rie, upper_rie = mac.solve_rie(num_candidates, w_init, max_iters=100, rounding="madow", use_cache=False)
    rie_profiler.disable()
    rie_profiler.dump_stats('graph_sparse_rie.prof')

    end_time = time.time()
    time_rie = end_time - start_time

    start_time = time.time()

    mac_profiler.enable()
    result_mac, unrounded_mac, upper_mac = mac.solve(num_candidates, w_init, max_iters=100, rounding="madow", use_cache=False)
    mac_profiler.disable()
    mac_profiler.dump_stats('graph_sparse_mac.prof')

    end_time = time.time()
    time_mac = end_time - start_time



    cost_mac_unrounded = mac.evaluate_objective(unrounded_mac)
    cost_rie_unrounded = mac.evaluate_objective(unrounded_rie)

    cost_mac_rounded = mac.evaluate_objective(result_mac)
    cost_rie_rounded = mac.evaluate_objective(result_rie)

    mac_times.append(time_mac)
    rie_times.append(time_rie)

    costs_unrounded_mac.append(cost_mac_unrounded)
    costs_unrounded_rie.append(cost_rie_unrounded)

    costs_rounded_mac.append(cost_mac_rounded)
    costs_rounded_rie.append(cost_rie_rounded)

import pandas as pd
rie_log = mac.optimizer._log
rie_costs = -1*np.array(rie_log['iterations']['cost'])
rie_iterations = rie_log['iterations']['iteration']

mac_log = mac.mac_log
mac_costs = mac_log['iterations']['cost']
mac_iterations = mac_log['iterations']['iteration']
plt.plot(rie_iterations, rie_costs, label="Riemannian")
plt.plot(mac_iterations, mac_costs, label="MAC")
plt.legend()
plt.show()
breakpoint()

# write np array to npz
np.savez("graph_sparse_data_2.npz", mac_times=mac_times, rie_times=rie_times,
         costs_unrounded_mac=costs_unrounded_mac, costs_unrounded_rie=costs_unrounded_rie,
         costs_rounded_mac=costs_rounded_mac, costs_rounded_rie=costs_rounded_rie,
         num_candidates_list=num_candidates_list, num_total_list=num_total_list)

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
plt.plot(num_total_list, mac_times, label="MAC")
plt.plot(num_total_list, rie_times, label="Riemannian")
plt.xlabel("Num Total Edges")
plt.ylabel("Time (s)")
plt.legend()

plt.figure()
plt.plot(num_total_list, mac_times, label="MAC")
plt.plot(num_total_list, rie_times, label="Riemannian")
plt.xlabel("Num Total Edges")
plt.ylabel("Time (s)")
plt.legend()

# plot the costs_unrounded_mac and costs_unrounded_rie against the num_total_list
plt.figure()
plt.plot(num_total_list, costs_unrounded_mac, label="MAC Unrounded")
plt.plot(num_total_list, costs_unrounded_rie, label="Riemannian Unrounded")
plt.xlabel("Num Total Edges")
plt.ylabel("Cost Unrounded")
plt.legend()
plt.show()


# data = {'MAC': [time_mac, cost_mac_unrounded, cost_mac_rounded],
#         'Rie': [time_rie, cost_rie_unrounded, cost_rie_rounded]}
# indices = ["time", "unrounded cost", "rounded cost"]
# df = pd.DataFrame(data, indices)
# df.to_csv("graph_sparse_time.csv")

# print(df)

# print(mac.prob_run_info['calls'])
# print(mac.prob_run_info['runtimes'])
# print(mac.prob_run_info['num_idx'])
# print(mac.prob_run_info['lap_rt'])
# print(mac.prob_run_info['wgle_rt'])

# print(mac.prob_rie_run_info['calls'])
# print(mac.prob_rie_run_info['runtimes'])
# print(mac.prob_rie_run_info['num_idx'])
# print(mac.prob_rie_run_info['lap_rt'])
# print(mac.prob_rie_run_info['wgle_rt'])

# init_selected = select_edges(candidate_edges, w_init)
# selected = select_edges(candidate_edges, result)

# init_selected_G = mac_to_nx(fixed_edges + init_selected)
# selected_G = mac_to_nx(fixed_edges + selected)

# print(f"lambda2 Initial: {mac.evaluate_objective(w_init)}")
# print(f"lambda2 Ours: {mac.evaluate_objective(result)}")

# plt.subplot(121)
# nx.draw(init_selected_G)
# plt.title(f"Initial Selection\n$\lambda_2$ = {mac.evaluate_objective(w_init):.2f}")
# plt.subplot(122)
# nx.draw(selected_G)
# plt.title(f"Ours\n$\lambda_2$ = {mac.evaluate_objective(result):.2f}")
# plt.show()

