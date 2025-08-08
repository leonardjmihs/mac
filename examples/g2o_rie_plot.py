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

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 16})

# SE-Sync setup
# sesync_lib_path = "/Users/kevin/repos/SESync/C++/build/lib"
sesync_lib_path = "/home/jungle/Git/SE-Sync/C++/build/lib"
sesync_lib_path = "/home/jungle/.local/lib/python3.10/site-packages/PySESync.cpython-310-x86_64-linux-gnu.so"
sys.path.insert(0, sesync_lib_path)

import PySESync


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [tmp file]")
        sys.exit()
        pass
    filename = sys.argv[1]
    results = pickle.load(open(filename, 'rb'))

    # output_dict = {
    #     "costs_per_iteration_mac": costs_per_iteration_mac,
    #     "costs_per_iteration_rie": costs_per_iteration_rie,
    #     "num_iterations_mac": num_iterations_mac,
    #     "num_iterations_rie": num_iterations_rie
    # }

    costs_per_iteration_mac = results['costs_per_iteration_mac']
    costs_per_iteration_rie = results['costs_per_iteration_rie']
    num_iterations_mac = results['num_iterations_mac']
    num_iterations_rie = results['num_iterations_rie']
    breakpoint()