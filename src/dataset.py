import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.special import expit
from tensorflow import math


def IHDP_dataset(params, path_data="datasets/IHDP/csv/", separate_files=False,
                 file_index=None):
    """ Tensorflow Dataset generator for the IHDP dataset.

    Parameters
    ----------
    params : dict
        Dictionary that contains all hyperparameters of the program. Use
        the function parse_arguments() in main.py to generate it.

    path_data : str
        Path to the folder that contains the csv files with data
 
    separate_files : bool
        Flag to determine if the files should all create a separate Dataset
        object or if they should become one large dataset.

    file_index : int
        Index used to pick a specific csv file to create a Dataset with. Only
        used if the separate_files flag is set.

    Returns
    -------
    dataset : tensorflow.data.Dataset
        The generated dataset

    metadata : ((float, float), int)
        Metadata used with the dataset. The first two floats are the original
        mean and std of the data, needed to rescale the data back to its
        original version. The third number is the number of classes for each
        categorical variable.
    """

    binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    contfeats = [i for i in range(25) if i not in binfeats]
    catfeats = binfeats
    nr_files = 10
    file_prefix = "ihdp_npci_"

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
    """Tensorflow Dataset generator for the TWINS dataset.

    Parameters
    ----------
    params : dict
        Dictionary that contains all hyperparameters of the program. Use
        the function parse_arguments() in main.py to generate it.

    do_preprocessing : str

    separate_files : bool
        kept_for compatibility with other datasets.

    file_index : int
        kept_for compatibility with other datasets.

    Returns
    -------
    dataset : tensorflow.data.Dataset
        The generated dataset

    metadata : ((float, float), int)
        Metadata used with the dataset. The first two floats are the original
        mean and std of the data, needed to rescale the data back to its
        original version. The third number is the number of classes for each
        categorical variable.

    """

    flip_prob = 0.3
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
    z_key = 'gestat10'
    for key, value in covar_types.items():
        if value == "ord":
            cont_keys.append(key)
        if value == "cat":
            if key != z_key:
                cat_keys.append(key)
        if value == "bin":
            if key != 'bord':
                bin_keys.append(key)
    cat_keys += bin_keys
    cat_keys += cont_keys
    w_o = np.random.normal(loc=0, scale=0.1, size=(len(cat_keys), 1))
    w_h = np.random.normal(loc=5, scale=0.1)

    x = pd.read_csv(f"{path_data}twin_pairs_X_3years_samesex.csv", sep=',', dtype=np.float64)
    x = x.loc[indices]
    x = x.interpolate(method='pad', axis=0).fillna(method='backfill', axis=0)
    z = x[z_key].to_numpy(int)
    x = x[cat_keys].to_numpy()
    x_w = x @ w_o
    z_w = np.expand_dims((z / 10 - 0.1) * w_h, axis=1)
    t = np.squeeze(np.random.binomial(1, expit(x_w + z_w)))

    proxy = tf.one_hot(z, 10, axis=-1, dtype=tf.float64)
    proxy = tf.concat([proxy, proxy, proxy], axis=1)
    noise = tf.random.uniform(proxy.shape, minval=0, maxval=1)
    noise = math.less(noise, flip_prob)
    noisy_proxy = tf.zeros_like(proxy)
    noisy_proxy += tf.scatter_nd(tf.where(noise), tf.boolean_mask(1 - proxy, noise), (len(z), 30))
    noise = math.logical_not(noise)
    noisy_proxy += tf.scatter_nd(tf.where(noise), tf.boolean_mask(proxy, noise), (len(z), 30))

    x_cat = noisy_proxy
    x_cont = tf.zeros((len(x_cat), 0), dtype=tf.float64)
    y = tf.cast(tf.expand_dims(data_y[indices, t], axis=1), tf.float64)
    y_cf = tf.cast(tf.expand_dims(data_y[indices, 1-t], axis=1), tf.float64)
    t = tf.cast(tf.expand_dims(t, axis=1), tf.float64)
    mu_1 = tf.cast(tf.expand_dims(data_y[indices, 1], axis=1), tf.float64)
    mu_0 = tf.cast(tf.expand_dims(data_y[indices, 0], axis=1), tf.float64)

    scaling_data = (0, 1)
    nr_unique_values = 10
    metadata = (scaling_data, nr_unique_values)

    return tf.data.Dataset.from_tensor_slices(((x_cat, x_cont,
                                                t, y, y_cf,
                                                mu_1, mu_0))), metadata


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    params = {}
    data, metadata = IHDP_dataset(params)
    nr_unique_values = metadata[1]

    for _, data_sample in data.batch(5).enumerate():
        for data in data_sample:
            print(data)
        break

    print()
    data, metadata = TWINS_dataset(params, do_preprocessing=True)
    nr_unique_values = metadata[1]

    for _, data_sample in data.batch(5).enumerate():
        for data in data_sample:
            print(data)
        break
