import tensorflow as tf
import numpy as np

def IHDP_dataset(params, path_data="datasets/IHDP/csv/", file_prefix="ihdp_npci_", separate_files=False, file_index=None):
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
    
    if separate_files:
        assert file_index is not None, "No file index given"
        assert file_index < nr_files, "File index invalid"
        data = np.loadtxt(f"{path_data}{file_prefix}{file_index + 1}.csv", delimiter=',', dtype=np.float64)
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
    x_bin = np.array(x_bin)
    t = np.expand_dims(np.array(t), axis=1)
    y = np.expand_dims(np.array(y), axis=1) # Y needs to have zero mean and std 1 during training
    y_mean, y_std = np.mean(y), np.std(y)
    y = (y - y_mean) / y_std
    y_cf = np.expand_dims(np.array(y_cf), axis=1)
    #self.y_cf__mean, self.y_cf_std = np.mean(y_cf), np.std(y_cf)
    mu_0 = np.expand_dims(np.array(mu_0), axis=1)
    mu_1 = np.expand_dims(np.array(mu_1), axis=1)

    return tf.data.Dataset.from_tensor_slices(((x_bin, 
                                                x_cont, 
                                                t, 
                                                y, 
                                                y_cf, 
                                                mu_0, 
                                                mu_1)))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    params = {}
    data = IHDP_dataset(params)
    for _, data_sample in data.batch(7470).enumerate():
        print(data_sample[5])

