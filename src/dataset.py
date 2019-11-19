import tensorflow as tf
import numpy as np

def IHDP_dataset(path_data="datasets/IHDP/csv/", file_prefix="ihdp_npci_"):
    binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    contfeats = [i for i in range(25) if i not in binfeats]
    nr_files = 10

    t = []
    y = []
    y_cf = []
    mu_0 = []
    mu_1 = []
    x_bin = []
    x_cont = []
    for i in range(1, nr_files + 1):
        data = np.loadtxt(f"{path_data}{file_prefix}{i}.csv", delimiter=',')
        for line in data:
            t.append(line[0])
            y.append(line[1])
            y_cf.append(line[2])
            mu_0.append(line[3])
            mu_1.append(line[4])
            x = line[5:]
            x[13] -= 1 # Value is in [1, 2] instead of [0, 1]
            x_bin.append(x[binfeats])
            x_cont.append(x[contfeats])

    x_cont = np.array(x_cont)
    def IHDP():
        #return tf.data.Dataset.from_tensor_slices((t, y, y_cf, mu_0, mu_1, x_bin, x_cont))
        print("#####################")
        return tf.data.Dataset.from_tensor_slices(((x_cont, y, t), ()))

    return IHDP

