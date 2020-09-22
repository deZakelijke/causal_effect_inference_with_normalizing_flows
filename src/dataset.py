import h5py
import sys
import tensorflow as tf
import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.special import expit
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow_probability import distributions as tfd


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
    train_dataset : tensorflow.data.Dataset
        The generated dataset

    test_dataset : tensorflow.data.Dataset
        The generated dataset

    """
    params["x_dims"] = 44
    params["x_cat_dims"] = 19
    params["x_cont_dims"] = 6
    params["t_dims"] = 2
    params["t_type"] = 'Categorical'
    params["y_dims"] = 1
    params["y_type"] = "Normal"
    params["z_dims"] = 16
    params['category_sizes'] = 2
    params['architecture_type'] = 'FC_net'

    binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24]
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

    # Make an array of the two value for t we want to compare in the ITE
    t = np.expand_dims(np.array(t), axis=1)
    t = enc.fit(t).transform(t)
    t_predict = np.zeros((len(t), 2, 2))
    t_predict[:, 0, 0] = 1
    t_predict[:, 1, 1] = 1
    y = np.expand_dims(np.array(y), axis=1)
    y_cf = np.expand_dims(np.array(y_cf), axis=1)

    # we pick either y or y_cf depending on the value of t
    y_predict = np.zeros((len(y), 2, 1))
    idx_f = (t_predict[:, 0] == t)[:, 0]
    idx_cf = (t_predict[:, 0] != t)[:, 0]
    y_predict[idx_f, 0] = y[idx_f]
    y_predict[~idx_f, 1] = y[~idx_f]
    y_predict[~idx_f, 0] = y_cf[~idx_f]
    y_predict[idx_f, 1] = y_cf[idx_f]

    y_mean = np.mean(tf.concat([y, y_cf], 1))
    y_std = np.std(tf.concat([y, y_cf], 1))
    y = (y - y_mean) / y_std
    y_cf = (y_cf - y_mean) / y_std

    mu_0 = np.expand_dims(np.array(mu_0), axis=1)
    mu_1 = np.expand_dims(np.array(mu_1), axis=1)
    scaling_data = (y_mean, y_std)

    params['scaling_data'] = scaling_data

    train_set = tf.data.Dataset.from_tensor_slices(((x_bin[idx_tr],
                                                     x_cont[idx_tr],
                                                     t[idx_tr],
                                                     t_predict[idx_tr],
                                                     y[idx_tr],
                                                     y_predict[idx_tr],
                                                     mu_0[idx_tr],
                                                     mu_1[idx_tr])))
    test_set = tf.data.Dataset.from_tensor_slices(((x_bin[idx_te],
                                                    x_cont[idx_te],
                                                    t[idx_te],
                                                    t_predict[idx_te],
                                                    y[idx_te],
                                                    y_predict[idx_te],
                                                    mu_0[idx_te],
                                                    mu_1[idx_te])))

    return train_set, test_set


def IHDP_LARGE(params, path_data="datasets/IHDP_LARGE/", separate_files=False,
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
    train_dataset : tensorflow.data.Dataset
        The generated dataset

    test_dataset : tensorflow.data.Dataset
        The generated dataset

    """
    params["x_dims"] = 44
    params["x_cat_dims"] = 19
    params["x_cont_dims"] = 6
    params["t_dims"] = 2
    params["t_type"] = 'Categorical'
    params["y_dims"] = 1
    params["y_type"] = "Normal"
    params["z_dims"] = 16
    params['category_sizes'] = 2
    params['architecture_type'] = 'FC_net'

    binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24]
    contfeats = [i for i in range(25) if i not in binfeats]
    catfeats = binfeats
    nr_files = 100
    file_name = "ihdp_npci_1-100"

    with np.load(f"{path_data}{file_name}.train.npz") as train_data, \
            np.load(f"{path_data}{file_name}.test.npz") as test_data:
        x = np.concatenate([train_data['x'], test_data['x']])
        t = np.concatenate([train_data['t'], test_data['t']])
        y_f = np.concatenate([train_data['yf'], test_data['yf']])
        y_cf = np.concatenate([train_data['ycf'], test_data['ycf']])
        mu0 = np.concatenate([train_data['mu0'], test_data['mu0']])
        mu1 = np.concatenate([train_data['mu1'], test_data['mu1']])

    if separate_files:
        x = x[..., file_index]
        t = t[..., file_index]
        y_f = y_f[..., file_index]
        y_cf = y_cf[..., file_index]
        mu0 = mu0[..., file_index]
        mu1 = mu1[..., file_index]
    else:
        x = np.concatenate([x[..., i] for i in np.arange(x.shape[-1])])
        t = np.concatenate([t[..., i] for i in np.arange(t.shape[-1])])
        y_f = np.concatenate([y_f[..., i] for i in np.arange(y_f.shape[-1])])
        y_cf = np.concatenate([y_cf[..., i] for i in np.arange(y_cf.shape[-1])])
        mu0 = np.concatenate([mu0[..., i] for i in np.arange(mu0.shape[-1])])
        mu1 = np.concatenate([mu1[..., i] for i in np.arange(mu1.shape[-1])])

    idx_tr, idx_te = train_test_split(np.arange(x.shape[0]), test_size=0.1,
                                      random_state=1)


    x_cont = x[:, contfeats]
    x_cat = np.array(x[:, binfeats], dtype=int)
    enc = OneHotEncoder(categories='auto', sparse=False)
    x_cat = enc.fit(x_cat).transform(x_cat)

    # Make an array of the two value for t we want to compare in the ITE
    t = np.expand_dims(t, axis=1)
    t = enc.fit(t).transform(t)
    t_predict = np.zeros((len(t), 2, 2))
    t_predict[:, 0, 0] = 1
    t_predict[:, 1, 1] = 1
    y_f = np.expand_dims(np.array(y_f), axis=1)
    y_cf = np.expand_dims(np.array(y_cf), axis=1)

    # we pick either y or y_cf depending on the value of t
    y_predict = np.zeros((len(y_f), 2, 1))
    idx_f = (t_predict[:, 0] == t)[:, 0]
    idx_cf = (t_predict[:, 0] != t)[:, 0]
    y_predict[idx_f, 0] = y_f[idx_f]
    y_predict[~idx_f, 1] = y_f[~idx_f]
    y_predict[~idx_f, 0] = y_cf[~idx_f]
    y_predict[idx_f, 1] = y_cf[idx_f]

    y_mean = np.mean(tf.concat([y_f, y_cf], 1))
    y_std = np.std(tf.concat([y_f, y_cf], 1))
    y_f = (y_f - y_mean) / y_std
    y_cf = (y_cf - y_mean) / y_std

    mu_0 = np.expand_dims(np.array(mu0), axis=1)
    mu_1 = np.expand_dims(np.array(mu1), axis=1)
    scaling_data = (y_mean, y_std)

    params['scaling_data'] = scaling_data

    train_set = tf.data.Dataset.from_tensor_slices(((x_cat[idx_tr],
                                                     x_cont[idx_tr],
                                                     t[idx_tr],
                                                     t_predict[idx_tr],
                                                     y_f[idx_tr],
                                                     y_predict[idx_tr],
                                                     mu_0[idx_tr],
                                                     mu_1[idx_tr])))
    test_set = tf.data.Dataset.from_tensor_slices(((x_cat[idx_te],
                                                    x_cont[idx_te],
                                                    t[idx_te],
                                                    t_predict[idx_te],
                                                    y_f[idx_te],
                                                    y_predict[idx_te],
                                                    mu_0[idx_te],
                                                    mu_1[idx_te])))

    return train_set, test_set


def TWINS(params, path_data="datasets/TWINS/",
          separate_files=None, file_index=None):
    """Tensorflow Dataset generator for the TWINS dataset.

    Parameters
    ----------
    params : dict
        Dictionary that contains all hyperparameters of the program. Use
        the function parse_arguments() in main.py to generate it.

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

    params["x_dims"] = 30
    params["x_cat_dims"] = 3
    params["x_cont_dims"] = 0
    params["t_dims"] = 2
    params["t_type"] = "Categorical"
    params["y_dims"] = 2
    params["y_type"] = "Categorical"
    params["z_dims"] = 16
    params["category_sizes"] = 10
    params['architecture_type'] = 'FC_net'

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
    y = enc.fit(y).transform(y)
    y_cf = enc.transform(y_cf)

    # Make an array of the two value for t we want to compare in the ITE
    t = np.expand_dims(np.array(t), axis=1)
    t = enc.fit(t).transform(t)
    t_predict = np.zeros((len(t), 2, 2))
    t_predict[:, 0, 0] = 1
    t_predict[:, 1, 1] = 1

    # we pick either y or y_cf depending on the value of t
    y_predict = np.zeros((len(y), 2, 2))
    idx_f = (t_predict[:, 0]==t)
    idx_cf = (t_predict[:, 0]!=t)[:, 0]
    y_predict[idx_f, 0] = y[idx_f]
    y_predict[~idx_f, 1] = y[~idx_f]
    y_predict[~idx_f, 0] = y_cf[~idx_f]
    y_predict[idx_f, 1] = y_cf[idx_f]

    mu_1 = np.expand_dims(data_y[indices, 1], axis=1)
    mu_0 = np.expand_dims(data_y[indices, 0], axis=1)

    idx_tr, idx_te = train_test_split(np.arange(x_cat.shape[0]), test_size=0.1,
                                      random_state=1)
    train_set = tf.data.Dataset.from_tensor_slices(((x_cat[idx_tr],
                                                     x_cont[idx_tr],
                                                     t[idx_tr],
                                                     t_predict[idx_tr],
                                                     y[idx_tr],
                                                     y_predict[idx_tr],
                                                     mu_1[idx_tr],
                                                     mu_0[idx_tr])))
    test_set = tf.data.Dataset.from_tensor_slices(((x_cat[idx_te],
                                                    x_cont[idx_te],
                                                    t[idx_te],
                                                    t_predict[idx_te],
                                                    y[idx_te],
                                                    y_predict[idx_te],
                                                    mu_1[idx_te],
                                                    mu_0[idx_te])))
    scaling_data = (0, 1)
    params["scaling_data"] = scaling_data
    return train_set, test_set


def SHAPES(params, path_data="datasets/SHAPES/", separate_files=None,
           file_index=None):
    """ Loader for she shapes dataset.

    This dataset is used as a sanity check for the model. It doesn't have
    any (latent) confounding and the mapping from z to x is the identity
    map without any noise. That means that the 'counterfactual' outcome is
    the outcome when no action was taken, so just the original image.
    """

    params["x_dims"] = (50, 50, 3)
    params["x_cat_dims"] = (50, 50, 0)
    params["x_cont_dims"] = (50, 50, 3)
    params["t_dims"] = 20
    params["t_loss"] = CategoricalCrossentropy()
    params["y_dims"] = (50, 50, 3)
    params["y_loss"] = MeanSquaredError()
    params["z_dims"] = (50, 50, 3)
    params["category_sizes"] = 0
    params['architecture_type'] = 'ResNet'

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
    t = np.expand_dims(t, axis=1).astype(float)
    enc = OneHotEncoder(categories='auto', sparse=False)
    # x_bin = enc.fit(x_bin).transform(x_bin)
    t = enc.fit(t).transform(t)
    t_cf = np.zeros_like(t)
    # Reshape t?
    y = np.rollaxis(train_array_dict['next_obs'], 1, 4).astype(float)
    y_cf = np.rollaxis(train_array_dict['obs'], 1, 4).astype(float)
    mu_1 = np.zeros((len(x_cont), 1, 1, 1))
    mu_0 = np.zeros((len(x_cont), 1, 1, 1))
    train_set = tf.data.Dataset.from_tensor_slices(((x_cat,
                                                     x_cont,
                                                     t,
                                                     t_cf,
                                                     y,
                                                     y_cf,
                                                     mu_1,
                                                     mu_0)))

    test_array_dict = load_list_dict_h5py(path_data + test_name)
    x_cont = np.rollaxis(test_array_dict['obs'], 1, 4).astype(float)
    x_cat = np.zeros((len(x_cont), 50, 50, 0))
    t = np.reshape(test_array_dict['action'],
                   (len(test_array_dict['action']), 1, 1, 1)).astype(float)
    t_cf = np.zeros_like(t)
    y = np.rollaxis(test_array_dict['next_obs'], 1, 4).astype(float)
    y_cf = np.rollaxis(test_array_dict['obs'], 1, 4).astype(float)
    mu_1 = np.zeros((len(x_cont), 0, 0, 0))
    mu_0 = np.zeros((len(x_cont), 0, 0, 0))

    test_set = tf.data.Dataset.from_tensor_slices(((x_cat,
                                                    x_cont,
                                                    t,
                                                    t_cf,
                                                    y,
                                                    y_cf,
                                                    mu_1,
                                                    mu_0)))

    scaling_data = (0, 1)
    params["scaling_data"] = scaling_data
    return train_set, test_set


def SPACE(params, path_data='datasets/SPACE/', separate_files=None,
          file_index=None):
    """ """
    params["x_dims"] = (60, 60, 3)
    params["x_cat_dims"] = (60, 60, 0)
    params["x_cont_dims"] = (60, 60, 3)
    params["t_dims"] = 2
    params["t_type"] = "Normal"
    params["y_dims"] = 1
    params["y_type"] = "Normal"
    params["z_dims"] = 256
    params["category_sizes"] = 0
    params['architecture_type'] = 'ResNet'

    with h5py.File(f"{path_data}space_data_x.hdf5", "r") as f:
        x = np.array(f['Space_dataset_x'])
    with h5py.File(f"{path_data}space_data_t.hdf5", "r") as f:
        t = np.array(f['Space_dataset_t'])
    with h5py.File(f"{path_data}space_data_t_predict.hdf5", "r") as f:
        t_predict = np.array(f['Space_dataset_t_predict'])
    with h5py.File(f"{path_data}space_data_y.hdf5", "r") as f:
        y = np.array(f['Space_dataset_y'])
    with h5py.File(f"{path_data}space_data_y_predict.hdf5", "r") as f:
        y_predict = np.array(f['Space_dataset_y_predict'])

    idx_tr, idx_te = train_test_split(np.arange(x.shape[0]), test_size=0.1,
                                      random_state=1)

    x_cont = x
    x_cat = np.zeros((len(x_cont), 60, 60, 0))
    y = np.expand_dims(np.array(y), axis=1)
    # y_cf = np.expand_dims(np.array(y_cf), axis=1)
    # y_mean = np.mean(tf.concat([y, y_cf], 1))
    y_mean = np.mean(y)
    # y_std = np.std(tf.concat([y, y_cf], 1))
    y_std = np.std(y)
    y = (y - y_mean) / y_std
    # y_cf = (y_cf - y_mean) / y_std
    scaling_data = (y_mean, y_std)
    params['scaling_data'] = scaling_data
    y_predict = np.expand_dims(y_predict, -1)

    mu_1 = np.zeros((len(x_cont), 1))
    mu_0 = np.zeros((len(x_cont), 1))

    train_set = tf.data.Dataset.from_tensor_slices(((x_cat[idx_tr],
                                                     x_cont[idx_tr],
                                                     t[idx_tr],
                                                     t_predict[idx_tr],
                                                     y[idx_tr],
                                                     y_predict[idx_tr],
                                                     mu_0[idx_tr],
                                                     mu_1[idx_tr])))
    test_set = tf.data.Dataset.from_tensor_slices(((x_cat[idx_te],
                                                    x_cont[idx_te],
                                                    t[idx_te],
                                                    t_predict[idx_te],
                                                    y[idx_te],
                                                    y_predict[idx_te],
                                                    mu_0[idx_te],
                                                    mu_1[idx_te])))
    return train_set, test_set


def SPACE_NO_GRAV(params, path_data='datasets/SPACE_NO_GRAV/',
                  separate_files=None,
                  file_index=None):
    return SPACE(params, path_data, separate_files, file_index)


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
    train_data, test_data = IHDP(params, separate_files=True, file_index=0)

    for _, data_sample in train_data.batch(5).enumerate():
        for data in data_sample:
            print(data)
        break


def test_IHDP_LARGE():
    params = {}
    train_data, test_data = IHDP_LARGE(params, separate_files=True, file_index=0)

    for _, data_sample in train_data.batch(5).enumerate():
        for data in data_sample:
            print(data)
        break


def test_TWINS():
    params = {}
    train_data, test_data = TWINS(params)

    for _, data_sample in train_data.batch(5).enumerate():
        print(data_sample[0].shape)
        print(data_sample[1].shape)
        print(data_sample[2].shape)
        print(data_sample[3].shape)
        print(data_sample[4].shape)
        print(data_sample[5].shape)
        print(data_sample[6].shape)
        print(data_sample[7].shape)
        for data in data_sample:
            for var in data:
                print(var.shape)
            # print(data)
        break


def test_SHAPES():
    params = {}
    train_data, test_data = SHAPES(params)
    for _, data_sample in train_data.batch(5).enumerate():
        for data in data_sample:
            for var in data:
                print(var.shape)
        break


def test_SPACE():
    params = {}
    train_data, test_data = SPACE(params)
    for _, data_sample in train_data.batch(5).enumerate():
        for data in data_sample:
            for var in data:
                print(var.shape)
                print(tf.math.reduce_min(var))
                print(tf.math.reduce_max(var))
        break


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # test_IHDP()
    # print()
    # test_IHDP_LARGE()
    print()
    test_TWINS()
    # print()
    # test_SHAPES()
    # print()
    # test_SPACE()
