import argparse
import numpy as np
import logging
import os
import re
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
    """ Parse all arguments and check for faulty input. """
    parser = argparse.ArgumentParser(description="Causal effect Normalizing Flow trainer")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size (default=32)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Beta parameter to balance rate vs distortion in loss."\
                        "Higher than one means more weight to rate, lower than one means"\
                        "more weight to distortion. (default=1.0)")
    parser.add_argument("--dataset", type=str, default="IHDP", 
                        help="Dataset used (default: IHDP)")
    parser.add_argument("--debug", action="store_true", default=False, 
                        help="Turn on debugging mode. What it does now is turn off "\
                        "summary writer and print a lott of stuff.")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of training iterations (default: 100)")
    parser.add_argument("--experiment_name", type=str, 
                        help="Name of experiment used for results folder. Disabled in debug mode")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Learning rate of the optmiser (default: 1e-4)")
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
                        help="Switch to training the model on each data file "\
                        "separately instead of everything at once")

    args = parser.parse_args()

    if args.debug:
        tf.random.set_seed(0)
        args.experiment_name = ""
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        logging.getLogger("tensorflow").setLevel(logging.WARNING)
        if args.experiment_name is None:
            raise ValueError("Experiment name is required if debug mode is disabled")
        if not re.match(r'^[A-Za-z0-9_]+$', args.experiment_name):
            raise ValueError("Experment name may only contain alphanumerical and underscore")

    if not args.model in VALID_MODELS:
       raise NotImplementedError(f"Model {args.model} is not implemented")

    if not args.dataset in VALID_DATASETS:
       raise NotImplementedError(f"Dataset {args.dataset} is not implemented")

    if not args.mode == "train":
        raise NotImplementedError("Only training mode is implemented")

    if args.model == "cevae":
        args.nr_flows = 0

    for arg, value in vars(args).items():
        print(f"Argument: {arg:<16} {value:<5}")
    print()

    return vars(args)

def main(params):
    """ Main execution. Creates logging aand writer, and launches selected training. """
    params["x_bin_size"] = 19
    params["x_cont_size"] = 6
    params["z_size"] = 16

    timestamp = time.strftime("%Y:%m:%d/%X")
    if not params["debug"]:
        logdir = (f"{params['model_dir']}{params['model']}/{params['dataset']}/"
                  f"{params['learning_rate']}/{timestamp}/{params['experiment_name']}")
        writer = tf.summary.create_file_writer(logdir)
    else:
        writer = None


    if params["separate_files"]:
        for i in range(10):
            dataset, y_mean, y_std = eval(f"{params['dataset']}_dataset")(params, separate_files=True, file_index=i)
            len_dataset = tf.data.experimental.cardinality(dataset)
            dataset = dataset.shuffle(len_dataset)
            train(params, dataset, len_dataset, writer, y_mean, y_std, i)
    else:
        dataset, y_mean, y_std = eval(f"{params['dataset']}_dataset")(params)
        len_dataset = tf.data.experimental.cardinality(dataset)
        dataset = dataset.shuffle(len_dataset)

        train(params, dataset, len_dataset, writer, y_mean, y_std)


def train(params, dataset, len_dataset, writer, y_mean, y_std, train_iteration=0):
    """ Runs training of selected model. """
    if params["model"] == "cevae":
        model = CEVAE(params)
    if params["model"] == "cenf":
        model = CENF(params)

    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    len_epoch = tf.data.experimental.cardinality(dataset.batch(params["batch_size"]))
    global_train_step = train_iteration * params["epochs"] * tf.data.experimental.cardinality(dataset.batch(params["batch_size"]))
    global_log_step = train_iteration * params["epochs"]

    if params["debug"]:
        for epoch in range(3):
            print(f"Epoch: {epoch}")
            avg_loss = 0
            step_start = global_train_step + epoch * len_epoch
            for step, features in dataset.batch(params["batch_size"]).enumerate(step_start):
                loss_value, grads = model.grad(features, step, params)
                avg_loss += loss_value
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                #break
            print(f"Epoch: {epoch}, average loss: {avg_loss / tf.dtypes.cast(len_epoch, tf.float64)}")
            stats = calc_stats(model, dataset, y_mean, y_std, params)
            print(f"Average ite: {stats[0]:.4f}, abs ate: {stats[1]:.4f}, pehe; {stats[2]:.4f}, y_error factual: {stats[3][0]:.4f}, y_error counterfactual {stats[3][1]:.4f}")
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
                print(f"Epoch: {epoch}, average loss: {(avg_loss / tf.dtypes.cast(len_epoch, tf.float64)):.4f}")
                stats = calc_stats(model, dataset, y_mean, y_std, params)
                print(f"Average ite: {stats[0]:.4f}, abs ate: {stats[1]:.4f}, pehe; {stats[2]:.4f}, "\
                      f"y_error factual: {stats[3][0]:.4f}, y_error counterfactual {stats[3][1]:.4f}")
 
                l_step = (epoch + global_log_step) // params['log_steps']
                tf.summary.scalar("metrics/loss", avg_loss / tf.dtypes.cast(len_epoch, tf.float64), step=l_step)
                tf.summary.scalar("metrics/ite", stats[0], step=l_step)
                tf.summary.scalar("metrics/ate", stats[1], step=l_step)
                tf.summary.scalar("metrics/pehe", stats[2], step=l_step)
                tf.summary.scalar("metrics/y_factual", stats[3][0], step=l_step)
                tf.summary.scalar("metrics/y_counterfactual", stats[3][1], step=l_step)

        print(f"Epoch: {epoch}, average loss: {(avg_loss / tf.dtypes.cast(len_epoch, tf.float64)):.4f}")
        stats = calc_stats(model, dataset, y_mean, y_std, params)
        print(f"Average ite: {stats[0]:.4f}, abs ate: {stats[1]:.4f}, pehe; {stats[2]:.4f}, "\
              f"y_error factual: {stats[3][0]:.4f}, y_error counterfactual {stats[3][1]:.4f}")
 
        l_step = (epoch + global_log_step + 1) // params['log_steps']
        tf.summary.scalar("metrics/loss", loss_value, step=l_step)
        tf.summary.scalar("metrics/ite", stats[0], step=l_step)
        tf.summary.scalar("metrics/ate", stats[1], step=l_step)
        tf.summary.scalar("metrics/pehe", stats[2], step=l_step)
        tf.summary.scalar("metrics/y_factual", stats[3][0], step=l_step)
        tf.summary.scalar("metrics/y_counterfactual", stats[3][1], step=l_step)



def test(params):
    pass

if __name__ == "__main__":
    params = parse_arguments()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    #tf.config.experimental_run_functions_eagerly(True)
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, False)

    main(params)

