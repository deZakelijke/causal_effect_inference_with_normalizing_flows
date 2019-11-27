import tensorflow as tf
import numpy as np

def IHDP_dataset(params, path_data="datasets/IHDP/csv/", file_prefix="ihdp_npci_"):
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
        data = np.loadtxt(f"{path_data}{file_prefix}{i}.csv", delimiter=',', dtype=np.float64)
        for line in data:
            t.append(line[0])       # Treatment true/false
            y.append(line[1])       # Outcome
            y_cf.append(line[2])    # Counterfactual outcome
            mu_0.append(line[3])    # Still not sure what these are 
            mu_1.append(line[4])    # ?????
            x = line[5:]            # Proxy features
            x[13] -= 1              # Value is in [1, 2] instead of [0, 1]
            x_bin.append(x[binfeats])
            x_cont.append(x[contfeats])

    x_cont = np.array(x_cont)
    x_bin = np.array(x_bin)
    t = np.expand_dims(np.array(t), axis=1)
    y = np.expand_dims(np.array(y), axis=1) # Y needs to have zero mean and std 1 during training
    y_mean, y_std = np.mean(y), np.std(y)
    y = (y - y_mean) / y_std
    y_cf = np.array(y_cf)
    #self.y_cf__mean, self.y_cf_std = np.mean(y_cf), np.std(y_cf)
    mu_0 = np.array(mu_0)
    mu_1 = np.array(mu_1)

    return tf.data.Dataset.from_tensor_slices(((x_bin, 
                                                x_cont, 
                                                t, 
                                                y, 
                                                y_cf, 
                                                mu_0, 
                                                mu_1), 
                                               ()))


