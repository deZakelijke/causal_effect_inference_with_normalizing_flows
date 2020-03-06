import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def remove_nans(data, binfeats, catfeats, contfeats):
    # Iterate over columns that still contains nans. Select one with fewest nans
    # Full up nans of selected column with knn function, based on all columns that don't have nans anymore

    # Get indices of all columns that do have nans, sorted ascending on the amount of nans.
    # Get indices of all columns that don't hav nans
    # Iterate over nan indices and remove nans with current non-nan columns. Impute nans
    # Add last index to cloumns that don't have nans.
    


    nan_counts = np.isnan(data).sum(axis=0)
    full_indices = np.argwhere(nan_counts == 0)
    nan_indices = np.nonzero(nan_counts)[0]
    nan_indices = nan_indices[np.argsort(nan_counts[nan_indices])]

    assert np.intersect1d(full_indices, nan_indices).size == 0
    for index in nan_indices:
        print(f"Removing nans in column: {index}")
        result = np.squeeze(knn_impute(data[:, index], np.squeeze(data[:, full_indices]), k))
        print(f"Nans remaining: {np.isnan(result).sum()}")
        data[:, index] = result
        full_indices = np.append(full_indices, index)
    assert len(full_indices) == data.shape[1]



def IHDP_dataset(params, path_data="datasets/IHDP/csv/", file_prefix="ihdp_npci_", separate_files=False, file_index=None):
    binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    contfeats = [i for i in range(25) if i not in binfeats]
    catfeats = binfeats
    nr_files = 10

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
    enc = OneHotEncoder(categories='auto')
    enc.fit(x_bin)
    x_bin = enc.transform(x_bin).toarray()
    t = np.expand_dims(np.array(t), axis=1)
    y = np.expand_dims(np.array(y), axis=1) 
    y_cf = np.expand_dims(np.array(y_cf), axis=1)

    y_mean, y_std = np.mean(tf.concat([y, y_cf], 1)), np.std(tf.concat([y, y_cf], 1))
    y = (y - y_mean) / y_std
    y_cf = (y_cf - y_mean) / y_std

    mu_0 = np.expand_dims(np.array(mu_0), axis=1)
    mu_1 = np.expand_dims(np.array(mu_1), axis=1)
    scaling_data = (y_mean, y_std)
    category_sizes = np.repeat(2, len(binfeats))

    metadata = (scaling_data, category_sizes, enc)

    return tf.data.Dataset.from_tensor_slices(((x_bin, 
                                                x_cont, 
                                                t, 
                                                y, 
                                                y_cf, 
                                                mu_0, 
                                                mu_1))), scaling_data, category_sizes

def TWINS_dataset(params, path_data="datasets/TWINS/", do_preprocessing=False):

    binfeats = [2, 3, 6, 9, 10, 13, 16, 18, 21, 25, 26, 27, 28, 30, 39, 40, 42, 43, 44, 45, 48, 49]
    contfeats = [17, 20] # Actually ordinal data but fuck it
    catfeats = [1, 4, 5, 7, 8, 11, 12, 14, 15, 19, 22, 23, 24, 31, 33, 33, 34, 35, 36, 37, 38, 41, 46, 47]
    catfeats += binfeats

    if do_preprocessing:
        x = np.genfromtxt(f"{path_data}twin_pairs_X_3years_samesex.csv", 
                               delimiter=',', skip_header=1)[:, 1:]
        pandas_x = pd.DataFrame(x)
        pandas_x = pandas_x.interpolate(method='pad', axis=1).fillna(method='backfill', axis=0)
        nr_unique_values = pandas_x.nunique()
        x = pandas_x.to_numpy()
        
        print(np.sum(nr_unique_values))
        x_cat = x[:, catfeats]
        category_sizes = nr_unique_values[catfeats]
        x_cont = x[:, contfeats]
        np.savetxt(f"{path_data}twin_pairs_X_preprocessed_3years_samesex.csv",
                   x, delimiter=',', newline='\n')

    else:
        raise NotImplementedError("Doesn't work any more, just preprocess every time")
        x = np.loadtxt(f"{path_data}twin_pairs_X_preprocessed_3years_samesex.csv", 
                               delimiter=',')
        x_bin = x[:, :len(binfeats)]
        x_cat = x[:, len(binfeats):len(binfeats) + len(catfeats)]
        x_cont = x[:, len(binfeats) + len(catfeats):]

    enc = OneHotEncoder(categories='auto')
    enc.fit(x_cat)
    x_cat = enc.transform(x_cat[:10000]).toarray()
    # TODO do this transformation per batch

    data_t = np.loadtxt(f"{path_data}twin_pairs_T_3years_samesex.csv", 
                        delimiter=',', dtype=np.float64, skiprows=1)
    data_y = np.loadtxt(f"{path_data}twin_pairs_Y_3years_samesex.csv", 
                        delimiter=',', dtype=np.float64, skiprows=1)
    
    t_0 = np.expand_dims(data_t[:, 1], axis=1)
    t_1 = np.expand_dims(data_t[:, 2], axis=1)
    y_0 = np.expand_dims(data_y[:, 1], axis=1)
    y_1 = np.expand_dims(data_y[:, 2], axis=1)

    scaling_data = None

    metadata = (scaling_data, category_sizes, enc)

    return tf.data.Dataset.from_tensor_slices(((x_cat, x_cont,
                                                t_0, t_1, 
                                                y_0, y_1))), scaling_data, category_sizes


if __name__ == "__main__":
    #import matplotlib.pyplot as plt
    
    params = {}
    data, _, _ = IHDP_dataset(params)
    print(data)
    for _, data_sample in data.batch(5).enumerate():
       print(data_sample[0])
       break

    print()
    data, _, nr_unique_values = TWINS_dataset(params, do_preprocessing=True)
    print(np.sum(nr_unique_values))

    for _, data_sample in data.batch(5).enumerate():
        print(data_sample)
        break

