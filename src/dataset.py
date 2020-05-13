import h5py
import sys
import tensorflow as tf
import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.special import expit


def IHDP(params, path_data="datasets/IHDP/csv/", separate_files=False,
                 file_index=None):
    """ Tensorflow Dataset generator for the IHDP dataset.

    Parameters
    ----------
    params : dict
        Dictionary that contains all hyperparameters of the program. Use
        the function parse_arguments() in main.py to generate it. This is
        also where all shape information of all variables will be stored

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
    params["x_bin_size"] = 19
    params["x_cat_size"] = 0 + 19
    params["x_cont_size"] = 6
    params["y_size"] = 1
    params["z_size"] = 16

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
        data = np.loadtxt(f"{path_data}{file_prefix}{file_index + 1}.csv",
                          delimiter=',', dtype=np.float64)
        t = data[:, 0]
        y = data[:, 1]
        y_cf = data[:, 2]
        mu_0 = data[:, 3]
        mu_1 = data[:, 4]
        x = data[:, 5:]
        x[:, 13] -= 1
        x_bin = x[:, binfeats]
        x_cont = x[:, contfeats]
    else:
        for i in range(1, nr_files + 1):
            data = np.loadtxt(f"{path_data}{file_prefix}{i}.csv",
                              delimiter=',', dtype=np.float64)
            t = data[:, 0]
            y = data[:, 1]
            y_cf = data[:, 2]
            mu_0 = data[:, 3]
            mu_1 = data[:, 4]
            x = data[:, 5:]
            x[:, 13] -= 1
            x_bin = x[:, binfeats]
            x_cont = x[:, contfeats]

    idx_tr, idx_te = train_test_split(np.arange(x.shape[0]), test_size=0.1,
                                      random_state=1)
    x_cont = np.array(x_cont)
    x_bin = np.array(x_bin, dtype=int)
    enc = OneHotEncoder(categories='auto', sparse=False)
    x_bin = enc.fit(x_bin).transform(x_bin)

    t = np.expand_dims(np.array(t), axis=1)
    t = enc.fit(t).transform(t)
    y = np.expand_dims(np.array(y), axis=1)
    y_cf = np.expand_dims(np.array(y_cf), axis=1)

    y_mean = np.mean(tf.concat([y, y_cf], 1))
    y_std = np.std(tf.concat([y, y_cf], 1))
    y = (y - y_mean) / y_std
    y_cf = (y_cf - y_mean) / y_std

    mu_0 = np.expand_dims(np.array(mu_0), axis=1)
    mu_1 = np.expand_dims(np.array(mu_1), axis=1)
    scaling_data = (y_mean, y_std)

    nr_unique_values = (2, 2)
    metadata = (scaling_data, nr_unique_values)

    train_set = tf.data.Dataset.from_tensor_slices(((x_bin[idx_tr],
                                                     x_cont[idx_tr],
                                                     t[idx_tr],
                                                     y[idx_tr],
                                                     y_cf[idx_tr],
                                                     mu_0[idx_tr],
                                                     mu_1[idx_tr])))
    test_set = tf.data.Dataset.from_tensor_slices(((x_bin[idx_te],
                                                    x_cont[idx_te],
                                                    t[idx_te],
                                                    y[idx_te],
                                                    y_cf[idx_te],
                                                    mu_0[idx_te],
                                                    mu_1[idx_te])))

    return train_set, test_set,  metadata


def TWINS(params, path_data="datasets/TWINS/", do_preprocessing=True,
                  separate_files=None, file_index=None):
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

    params["x_bin_size"] = 0
    params["x_cat_size"] = 3
    params["x_cont_size"] = 0
    params["y_size"] = 1
    params["z_size"] = 16

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

    x = pd.read_csv(f"{path_data}twin_pairs_X_3years_samesex.csv", sep=',',
                    dtype=np.float64)
    x = x.loc[indices]
    x = x.interpolate(method='pad', axis=0).fillna(method='backfill', axis=0)
    z = x[z_key].to_numpy(int)
    x = x[cat_keys].to_numpy()
    x_w = x @ w_o
    z_w = np.expand_dims((z / 10 - 0.1) * w_h, axis=1)
    t = np.squeeze(np.random.binomial(1, expit(x_w + z_w)))
    z = np.expand_dims(z, axis=1)

    enc = OneHotEncoder(categories='auto', sparse=False)
    proxy = enc.fit(z).transform(z)
    proxy = np.concatenate([proxy, proxy, proxy], axis=1).astype(int)
    noise = np.random.uniform(size=proxy.shape)
    noise = noise < flip_prob
    noisy_proxy = np.zeros(proxy.shape)
    noisy_proxy[noise] = np.logical_not(proxy[noise])
    noisy_proxy[np.logical_not(noise)] = proxy[np.logical_not(noise)]

    x_cat = noisy_proxy
    x_cont = np.zeros((len(x_cat), 0))
    y = np.expand_dims(data_y[indices, t], axis=1)
    y_cf = np.expand_dims(data_y[indices, 1-t], axis=1)
    t = np.expand_dims(t, axis=1).astype(float)
    enc = OneHotEncoder(categories='auto', sparse=False)
    t = enc.fit(t).transform(t)

    mu_1 = np.expand_dims(data_y[indices, 1], axis=1)
    mu_0 = np.expand_dims(data_y[indices, 0], axis=1)

    idx_tr, idx_te = train_test_split(np.arange(x_cat.shape[0]), test_size=0.1,
                                      random_state=1)
    train_set = tf.data.Dataset.from_tensor_slices(((x_cat[idx_tr],
                                                     x_cont[idx_tr],
                                                     t[idx_tr],
                                                     y[idx_tr],
                                                     y_cf[idx_tr],
                                                     mu_1[idx_tr],
                                                     mu_0[idx_tr])))
    test_set = tf.data.Dataset.from_tensor_slices(((x_cat[idx_te],
                                                    x_cont[idx_te],
                                                    t[idx_te],
                                                    y[idx_te],
                                                    y_cf[idx_te],
                                                    mu_1[idx_te],
                                                    mu_0[idx_te])))
    scaling_data = (0, 1)
    nr_unique_values = (10, 2)
    metadata = (scaling_data, nr_unique_values)

    return train_set, test_set, metadata


def SHAPES(params, path_data="datasets/SHAPES/", separate_files=None,
           file_index=None):
    """ Loader for she shapes dataset.

    This dataset is used as a sanity check for the model. It doesn't have
    any (latent) confounding and the mapping from z to x is the identity
    map without any noise. That means that the 'counterfactual' outcome is
    the outcome when no action was taken, so just the original image.
    """

    params["x_bin_size"] = (0, 0, 0)
    params["x_cat_size"] = (50, 50, 0)
    params["x_cont_size"] = (50, 50, 3)
    params["y_size"] = (50, 50, 3)
    params["z_size"] = (50, 50, 3)

    train_name = "shapes_train.h5"
    test_name = "shapes_test.h5"

    train_array_dict = load_list_dict_h5py(path_data + train_name)
    # test_array_dict = 
    x_cont = np.rollaxis(train_array_dict['obs'], 1, 4).astype(float)
    x_cat = np.zeros((len(x_cont), 50, 50, 0))
    # t = np.reshape(train_array_dict['action'],
    #                (len(train_array_dict['action']), 1, 1, 1)).astype(float)

    # x_bin = np.array(x_bin, dtype=int)
    t = train_array_dict['action']
    enc = OneHotEncoder(categories='auto', sparse=False)
    # x_bin = enc.fit(x_bin).transform(x_bin)
    t = enc.fit(t).transform(t)
    # Reshape t?
    y = np.rollaxis(train_array_dict['next_obs'], 1, 4).astype(float)
    y_cf = np.rollaxis(train_array_dict['obs'], 1, 4).astype(float)
    mu_1 = np.zeros((len(x_cont), 1, 1, 1))
    mu_0 = np.zeros((len(x_cont), 1, 1, 1))
    train_set = tf.data.Dataset.from_tensor_slices(((x_cat,
                                                     x_cont,
                                                     t,
                                                     y,
                                                     y_cf,
                                                     mu_1,
                                                     mu_0)))

    test_array_dict = load_list_dict_h5py(path_data + train_name)
    x_cont = np.rollaxis(train_array_dict['obs'], 1, 4).astype(float)
    x_cat = np.zeros((len(x_cont), 50, 50, 0))
    t = np.reshape(train_array_dict['action'],
                   (len(train_array_dict['action']), 1, 1, 1)).astype(float)
    y = np.rollaxis(train_array_dict['next_obs'], 1, 4).astype(float)
    y_cf = np.rollaxis(train_array_dict['obs'], 1, 4).astype(float)
    mu_1 = np.zeros((len(x_cont), 0, 0, 0))
    mu_0 = np.zeros((len(x_cont), 0, 0, 0))

    test_set = tf.data.Dataset.from_tensor_slices(((x_cat,
                                                    x_cont,
                                                    t,
                                                    y,
                                                    y_cf,
                                                    mu_1,
                                                    mu_0)))

    scaling_data = (0, 1)
    nr_unique_values = (0, 16)
    metadata = (scaling_data, nr_unique_values)

    return train_set, test_set, metadata


def load_list_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = defaultdict(list)
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            for key in hf[grp].keys():
                array_dict[key].append(hf[grp][key][:])
    for key, val in array_dict.items():
        val = np.array(val)
        val = np.reshape(val, (val.shape[0] * val.shape[1], *val.shape[2:]))
        array_dict[key] = val
    return array_dict


def test_IHDP():
    params = {}
    train_data, test_data, metadata = IHDP(params, separate_files=True,
                                                   file_index=0)
    nr_unique_values = metadata[1]
    print(metadata)

    for _, data_sample in train_data.batch(5).enumerate():
        for data in data_sample:
            print(data)
        break


def test_TWINS():
    params = {}
    train_data, test_data, metadata = TWINS(params,
                                                    do_preprocessing=True)
    nr_unique_values = metadata[1]

    for _, data_sample in train_data.batch(5).enumerate():
        for data in data_sample:
            for var in data:
                print(var.shape)
            # print(data)
        break


def test_SHAPES():
    params = {}
    train_data, test_data, metadata = SHAPES(params)
    for _, data_sample in train_data.batch(5).enumerate():
        for data in data_sample:
            for var in data:
                print(var.shape)
        break

   
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test_SHAPES()
    print()
    test_TWINS()

