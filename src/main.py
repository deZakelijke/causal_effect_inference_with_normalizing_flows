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
    parser.add_argument("--dataset", type=str, default="IHDP", 
                        help="Dataset used (default: IHDP)")
    parser.add_argument("--debug", action="store_true", default=False, 
                        help="Turn on debugging mode. What it does now is turn off summary writer and print a lott of stuff.")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of training iterations (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Learning rate of hte optmiser (default: 1e-3)")
    parser.add_argument("--log_steps", type=int, default=10, 
                        help="Save/print log every n steps (default: 10)")
    parser.add_argument("--mode", type=str, default="train", 
                        help="Switch between train and test mode (default: train)")
    parser.add_argument("--model", type=str, default="cevae", 
                        help="The type of model used for predictions (default: cevae)")
    parser.add_argument("--model_dir", type=str, default="/home/mdegroot/logs/", 
                        help="The directory to save the model to (default: ~/logs/)")
    parser.add_argument("--nr_flows", type=int, default=4, 
                        help="Number of flows in the flow models (default: 4)")
    parser.add_argument("--separate_files", action="store_true", default=False, 
                        help="Switch to training the model on each data file separately instead of everything at once")

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

    if args.model == "cevae":
        args.nr_flows = 0

    for arg, value in vars(args).items():
        print(f"Argument: {arg:<16} {value:<5}")
    print()

    return vars(args)

def main(params):
    params["x_bin_size"] = 19
    params["x_cont_size"] = 6
    params["z_size"] = 16

    timestamp = str(int(time.time()))[2:] 
    logdir = f"{params['model_dir']}{params['model']}/{params['dataset']}/{params['learning_rate']}/{timestamp}"
    if not params["debug"]:
        writer = tf.summary.create_file_writer(logdir)
    else:
        writer = None


    if params["separate_files"]:
        for i in range(10):
            dataset = eval(f"{params['dataset']}_dataset")(params, separate_files=True, file_index=i)
            len_dataset = tf.data.experimental.cardinality(dataset)
            dataset = dataset.shuffle(len_dataset)
            train(params, dataset, len_dataset, writer, i)
    else:
        dataset = eval(f"{params['dataset']}_dataset")(params)
        len_dataset = tf.data.experimental(dataset)
        dataset = dataset.shuffle(len_dataset)

        train(params, dataset, len_dataset, writer)


def train(params, dataset, len_dataset, writer, train_iteration=0):
    if params["model"] == "cevae":
        model = CEVAE(params)
    if params["model"] == "cenf":
        model = CENF(params)

    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    len_epoch = tf.data.experimental.cardinality(dataset.batch(params["batch_size"]))
    global_train_step = train_iteration * params["epochs"] * tf.data.experimental.cardinality(dataset.batch(params["batch_size"]))
    global_log_step = train_iteration * params["epochs"]

    if params["debug"]:
        for epoch in range(5):
            print(f"Epoch: {epoch}")
            avg_loss = 0
            step_start = global_train_step + epoch * len_epoch
            for step, features in dataset.batch(params["batch_size"]).enumerate(step_start):
                loss_value, grads = model.grad(features, step, params)
                avg_loss += loss_value
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                #break
            print(f"Epoch: {epoch}, average loss: {avg_loss / tf.dtypes.cast(len_epoch, tf.float64)}")
            stats = calc_stats(model, dataset, params)
            print(f"Average ite: {stats[0]:.4f}, abs ate: {stats[1]:.4f}, pehe; {stats[2]:.4f}")
            print("Epoch done")
        return


    with writer.as_default():
        for epoch in range(params["epochs"]):
            avg_loss = 0
            step_start = global_train_step + epoch * len_epoch
            for step, features in dataset.batch(params["batch_size"]).enumerate(step_start):
                loss_value, grads = model.grad(features, step, params)
                avg_loss += loss_value
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if epoch % params["log_steps"] == 0:
                print(f"Epoch: {epoch}, average loss: {avg_loss / tf.dtypes.cast(len_epoch, tf.float64)}")
                stats = calc_stats(model, dataset, params)
                print(f"Average ite: {stats[0]:.4f}, abs ate: {stats[1]:.4f}, pehe; {stats[2]:.4f}")
                tf.summary.scalar("metrics/loss", loss_value, step=epoch + global_log_step)
                tf.summary.scalar("metrics/ite", stats[0], step=epoch + global_log_step)
                tf.summary.scalar("metrics/ate", stats[1], step=epoch + global_log_step)
                tf.summary.scalar("metrics/pehe", stats[2], step=epoch + global_log_step)

        print(f"Epoch: {epoch}, loss: {loss_value}")
        stats = calc_stats(model, dataset, params)
        print(f"Average ite: {stats[0]:.4f}, abs ate: {stats[1]:.4f}, pehe; {stats[2]:.4f}")
        tf.summary.scalar("metrics/loss", loss_value, step=epoch + global_log_step)
        tf.summary.scalar("metrics/ite", stats[0], step=epoch + global_log_step)
        tf.summary.scalar("metrics/ate", stats[1], step=epoch + global_log_step)
        tf.summary.scalar("metrics/pehe", stats[2], step=epoch + global_log_step)


def test(params):
    pass

if __name__ == "__main__":
    params = parse_arguments()

    main(params)

