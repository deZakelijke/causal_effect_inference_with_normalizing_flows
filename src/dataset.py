import tensorflow as tf
import numpy as np

def IHDP(path_data="../datasets/IHDP/csv/", file_prefix="ihdp_npci_"):
    binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    contfeats = [i for i in range(25) if i not in binfeats]

    nr_files = 10
    for i in range(nr_files):
        data = np.loadtxt(f"{path_data}{file_prefix}{i}.csv", delimiter=',')
        for line in data:
            t = line[0]
            y = line[1]
            y_cf = line[2]
            mu_0 = line[3]
            mu_1 = line[4]
            x = line[5:]
            x[13] -= 1 # Value is in [1, 2] instead of [0, 1]
            x_bin = x[binfeats]
            x_cont = x[contfeats]
            yield t, y, y_cf, mu_0, mu_1, x_bin, x_cont


