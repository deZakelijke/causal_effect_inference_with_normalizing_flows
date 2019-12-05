import argparse
import numpy as np
import logging
import os
import sys
import tensorflow as tf
import time

from cevae import CEVAE
from cenf import CENF
from dataset import IHDP_dataset
from evaluation import calc_stats

VALID_MODELS = ["cevae", "cenf"]
VALID_DATASETS = ["IHDP"]

tf.keras.backend.set_floatx('float64')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Causal effect Normalizing Flow trainer")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default=16")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate of hte optmiser (default: 1e-3)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training iterations (default: 100)")
    parser.add_argument("--mode", type=str, default="train",
                        help="Switch between train and test mode (default: train)")
    parser.add_argument("--model", type=str, default="cevae",
                        help="The type of model used for predictions (default: cevae)")
    parser.add_argument("--dataset", type=str, default="IHDP",
                        help="Dataset used (default: IHDP")
    parser.add_argument("--model_dir", type=str, default="/home/mdegroot/logs/",
                        help="The directory to save the model to (default: ~/logs/)")
    parser.add_argument("--log_steps", type=int, default=10,
                        help="Save/print log every n steps (default: 10)")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Turn on debugging mode. All it does now is turn off summary writer")

    args = parser.parse_args()

    if args.debug:
        tf.random.set_seed(0)
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        logging.getLogger("tensorflow").setLevel(logging.WARNING)

    if not args.model in VALID_MODELS:
       raise NotImplementedError(f"Model {args.model} is not implemented")

    if not args.dataset in VALID_DATASETS:
       raise NotImplementedError(f"Dataset {args.dataset} is not implemented")

    for arg, value in vars(args).items():
        print(f"Argument: {arg:<16} {value:<5}")
    print()

    return vars(args)

def train(config, dataset, len_dataset):
    params["x_bin_size"] = 19
    params["x_cont_size"] = 6
    params["z_size"] = 16

    timestamp = str(int(time.time()))[2:] 
    logdir = f"{params['model_dir']}{params['model']}/{params['dataset']}/{params['learning_rate']}/{timestamp}"
    if not params["debug"]:
        writer = tf.summary.create_file_writer(logdir)

    if config["model"] == "cevae":
        model = CEVAE(params)
    if config["model"] == "cenf":
        model = CENF(params)

    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])

    if params["debug"]:
        for epoch in range(5):
            print(f"Epoch: {epoch}")
            step_start = epoch * (len_dataset // params["batch_size"] + 1)
            for step, features in dataset.batch(params["batch_size"]).enumerate(step_start):
                loss_value, grads = model.grad(features, step, params)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                #break
            print(f"Epoch: {epoch}, loss: {loss_value}")
            stats = calc_stats(model, dataset, params)
            print(f"Average ite: {stats[0]:.4f}, abs ate: {stats[1]:.4f}, pehe; {stats[2]:.4f}")
            print("Epoch done")
        return


    with writer.as_default():
        for epoch in range(params["epochs"]):
            step_start = epoch * (len_dataset // params["batch_size"] + 1)
            for step, features in dataset.batch(params["batch_size"]).enumerate(step_start):
                loss_value, grads = model.grad(features, step, params)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if epoch % params["log_steps"] == 0:
                print(f"Epoch: {epoch}, loss: {loss_value}")
                stats = calc_stats(model, dataset, params)
                print(f"Average ite: {stats[0]:.4f}, abs ate: {stats[1]:.4f}, pehe; {stats[2]:.4f}")
                tf.summary.scalar("metrics/loss", loss_value, step=epoch)
                tf.summary.scalar("metrics/ite", stats[0], step=epoch)
                tf.summary.scalar("metrics/ate", stats[1], step=epoch)
                tf.summary.scalar("metrics/pehe", stats[2], step=epoch)

def test(config):
    pass

if __name__ == "__main__":
    params = parse_arguments()

    if params["dataset"] == "IHDP":
        dataset = IHDP_dataset(params)
    len_dataset = 0
    for _ in dataset:
        len_dataset +=1
    dataset = dataset.shuffle(len_dataset)



    if params["mode"] == "train":
        train(params, dataset, len_dataset)
    else:
        test(params)
