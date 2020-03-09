import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def IHDP_dataset(params, path_data="datasets/IHDP/csv/", separate_files=False, file_index=None):
    binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    contfeats = [i for i in range(25) if i not in binfeats]
    catfeats = binfeats
    nr_files = 10
    file_prefix="ihdp_npci_"

    t = []
    y = []
    y_cf = []
    mu_0 = []
    mu_1 = []
    x_bin = []
    x_cont = []
    
    if separate_files:
        assert file_index is not None, "No file index given"
        assert file_index < nr_files, "File index invalid"
        data = np.loadtxt(f"{path_data}{file_prefix}{file_index + 1}.csv", delimiter=',', dtype=np.float64)
        for line in data:
            t.append(line[0])       # Treatment true/false
            y.append(line[1])       # Outcome
            y_cf.append(line[2])    # Counterfactual outcome
            mu_0.append(line[3])    # Outcome of the experiment if the dataset had been created with
            mu_1.append(line[4])    # a randomised double lbind trial
            x = line[5:]            # Proxy features
            x[13] -= 1              # Value is in [1, 2] instead of [0, 1]
            x_bin.append(x[binfeats])
            x_cont.append(x[contfeats])
    else:
        for i in range(1, nr_files + 1):
            data = np.loadtxt(f"{path_data}{file_prefix}{i}.csv", delimiter=',', dtype=np.float64)
            for line in data:
                t.append(line[0])      
                y.append(line[1])       
                y_cf.append(line[2])    
                mu_0.append(line[3])    
                mu_1.append(line[4])    
                x = line[5:]            
                x[13] -= 1              
                x_bin.append(x[binfeats])
                x_cont.append(x[contfeats])

    x_cont = np.array(x_cont)
    x_bin = np.array(x_bin, dtype=int)
    x_bin = tf.one_hot(x_bin, 2, axis=-1, dtype=tf.float64)
    x_bin = tf.reshape(x_bin, (len(x_bin), len(binfeats) * 2))

    t = np.expand_dims(np.array(t), axis=1)
    y = np.expand_dims(np.array(y), axis=1) 
    y_cf = np.expand_dims(np.array(y_cf), axis=1)

    y_mean, y_std = np.mean(tf.concat([y, y_cf], 1)), np.std(tf.concat([y, y_cf], 1))
    y = (y - y_mean) / y_std
    y_cf = (y_cf - y_mean) / y_std

    mu_0 = np.expand_dims(np.array(mu_0), axis=1)
    mu_1 = np.expand_dims(np.array(mu_1), axis=1)
    scaling_data = (y_mean, y_std)

    metadata = (scaling_data, 2)

    return tf.data.Dataset.from_tensor_slices(((x_bin, 
                                                x_cont, 
                                                t, 
                                                y, 
                                                y_cf, 
                                                mu_0, 
                                                mu_1))), metadata

def TWINS_dataset(params, path_data="datasets/TWINS/", do_preprocessing=True, separate_files=None, file_index=None):

    data_t = np.loadtxt(f"{path_data}twin_pairs_T_3years_samesex.csv", 
            delimiter=',', dtype=np.float64, skiprows=1)[:, 1:]
    data_y = np.loadtxt(f"{path_data}twin_pairs_Y_3years_samesex.csv", 
            delimiter=',', dtype=np.float64, skiprows=1)[:, 1:]
 
    indices = np.logical_and(data_t[:, 0] < 2000.0, data_t[:, 1] < 2000.0)

    unused = ["infant_id_0", "infant_id_1"]
    covar_types = eval(open(f"{path_data}covar_type.txt", 'r').read())
    cont_keys = []
    cat_keys = []
    bin_keys = []
    for key, value in covar_types.items():
        if value == "ord":
            cont_keys.append(key)
        if value == "cat":
            cat_keys.append(key)
        if value == "bin":
            if key != 'bord':
                bin_keys.append(key)

    cat_keys += bin_keys
    x = pd.read_csv(f"{path_data}twin_pairs_X_3years_samesex.csv", sep=',', dtype=np.float64)
    x = x.loc[indices]
    x = x.interpolate(method='pad', axis=0).fillna(method='backfill', axis=0)
    nr_unique_values = np.max(x[cat_keys].nunique())
    x_cont = x[cont_keys].to_numpy()
    x_cat = x[cat_keys].to_numpy()
    x_cat = tf.one_hot(x_cat, nr_unique_values, axis=-1, dtype=tf.float64)
    x_cat = tf.reshape(x_cat, (len(x_cat), x_cat.shape[1] * nr_unique_values))
    # x_bin = x[bin_keys].to_numpy()
    
   
    # t_0 = np.expand_dims(data_t[indices, 0], axis=1)
    # t_1 = np.expand_dims(data_t[indices, 1], axis=1)
    t = np.expand_dims(indices.astype(float), axis=1)
    y_0 = np.expand_dims(data_y[indices, 0], axis=1)
    y_1 = np.expand_dims(data_y[indices, 1], axis=1)
    # y = tf.where(


    scaling_data = None
    metadata = (scaling_data, nr_unique_values)

    return tf.data.Dataset.from_tensor_slices(((x_cat, x_cont,
                                                t, y_0, y_1))), metadata


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    params = {}
    # data, metadata = IHDP_dataset(params)
    # nr_unique_values = metadata[1]
    # batch_mapper = metadata[2]
    #
    # for _, data_sample in data.batch(5).enumerate():
    #     data_sample = batch_mapper(*data_sample)
    #     print(data_sample)
    #     break

    print()
    data, metadata = TWINS_dataset(params, do_preprocessing=True)
    nr_unique_values = metadata[1]

    for _, data_sample in data.batch(5).enumerate():
        print(data_sample)
        break

