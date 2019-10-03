import tensorflow as tf
import numpy as np

def IHDP(replications, path_data="../datasets/IHDP/csv/", file_prefix="ihdp_npci_"):
    self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    self.contfeats = [i for i in range(25) if i not in self.binfeats]

dataset = tf.data.Dataset.from_generator(gen, tf.float32)

