import argparse
import numpy as np
import logging
import os
import re
import sys
import tensorflow as tf
import time

from causal_inference_worker import CIWorker
from contextlib import nullcontext
from dataset import IHDP_dataset, TWINS_dataset
from evaluation import calc_stats
from tf.config.experimental import set_virtual_device_configuration

VALID_MODELS = ["cevae", "cenf"]
VALID_DATASETS = ["IHDP", "TWINS"]
DATASET_DISTRIBUTION_DICT = {"IHDP": {'x': ['M', 'N'], 't': 'B', 'y': 'N'},
                             "TWINS": {'x': ['M', 'N'], 't': 'B', 'y': 'B'}}


tf.keras.backend.set_floatx('float64')


def parse_arguments():
    """ Parse all arguments and check for faulty input.

    Parses all input arguments of the program. Several argumets, such as model
    and dataset are checked for validity. Raises error if invald arguments are
    passed to the program.

    Raises
    ------
    NotImplementedError
        If an invalid model, mode or dataset are passed.

    ValueError
        If the name of the experiment contains invalid characters or if no
        exeriment name is provided.

    Return
    ------
    params : dict
        A dictionary containing all parameters are passed. Keys are the
        aruments and the values are the values of the parsed arguments or
        their default value.
    """

    parser = argparse.ArgumentParser(description="Causal effect Normalizing Flow trainer")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default=32)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Beta parameter to balance rate vs distortion in loss."
                        "Higher than one means more weight to rate, lower than one means"
                        "more weight to distortion. (default=1.0)")
    parser.add_argument("--dataset", type=str, default="IHDP",
                        help="Dataset used (default: IHDP)")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Turn on debugging mode. What it does now is turn off "
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
    parser.add_argument("--model_dir", type=str, default="/home/mgroot/logs/",
                        help="The directory to save the model to (default: ~/logs/)")
    parser.add_argument("--nr_flows", type=int, default=4,
                        help="Number of flows in the flow models (default: 4)")
    parser.add_argument("--separate_files", action="store_true", default=False,
                        help="Switch to training the model on each data file "
                        "separately instead of everything at once")

    args = parser.parse_args()

    if args.debug:
        tf.random.set_seed(0)
        args.experiment_name = ""
        args.log_steps = 1
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        logging.getLogger("tensorflow").setLevel(logging.WARNING)
        if args.experiment_name is None:
            raise ValueError("Experiment name is required if debug mode is "
                             "disabled")
        if not re.match(r'^[A-Za-z0-9_]+$', args.experiment_name):
            raise ValueError("Experment name may only contain alphanumerical "
                             "and underscore")

    if args.model not in VALID_MODELS:
        raise NotImplementedError(f"Model {args.model} is not implemented")

    if args.dataset not in VALID_DATASETS:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented")

    if not args.mode == "train":
        raise NotImplementedError("Only training mode is implemented")

    if args.model == "cevae":
        args.nr_flows = 0

    for arg, value in vars(args).items():
        print(f"Argument: {arg:<16} {value:<5}")
    print()

    args.dataset_distributions = DATASET_DISTRIBUTION_DICT[args.dataset]

    return vars(args)


def print_stats(stats, index):
    """ Prints the statistics and puts them in Tensorboard.

    Prints the tuple of statistics in a readable format and passes them to the
    Tensorboard writer if one is active in the current context.

    Parameters
    ----------
    stats : (float, float, float (float, float))
        The statistic that should be printed.
    index : int
        Index used for the graph in tensorboard. Each next call of print_stats
        should have the consecutive index.
    """

    print(f"Average ite: {stats[0]:.4f}, abs ate: {stats[1]:.4f}, "
          f"pehe: {stats[2]:.4f}, y_error factual: {stats[3][0]:.4f}, "
          f"y_error counterfactual {stats[3][1]:.4f}")

    tf.summary.scalar("metrics/ite", stats[0], step=index)
    tf.summary.scalar("metrics/ate", stats[1], step=index)
    tf.summary.scalar("metrics/pehe", stats[2], step=index)
    tf.summary.scalar("metrics/y_factual", stats[3][0], step=index)
    tf.summary.scalar("metrics/y_counterfactual", stats[3][1], step=index)


def train(params, writer, train_iteration=0):
    """ Runs training of selected model.

    Loads a dataset and trains a new mode lfor the specified number of epochs.
    All settings should be listed in the params dictionary.

    Parameters
    ----------
    params : dict
        Dictionary of all settings needed to define the training procedure.
        Use the parse_arguments() function to generate a dictionary with the
        required fileds

    writer : tensorflow.summary.writer
        The writer opbject to which all results and logs are written.
        Can be None

    train_iteration : int, optional
        Incrementor that should count the repetitions of times train()
        is called.
        Used to correcly log a series of trainings in the writer.

    Returns
    -------
    stats : tuple
        Tuple of the statistics that were calculated after the last epoch.
    """

    dataset, metadata = eval(f"{params['dataset']}_dataset")\
                            (params, separate_files=params['separate_files'],
                             file_index=train_iteration)
    scaling_data = metadata[0]
    category_sizes = metadata[1]

    len_dataset = tf.data.experimental.cardinality(dataset)
    dataset = dataset.shuffle(len_dataset)

    model = CIWorker(params, category_sizes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    len_epoch = tf.data.experimental.cardinality(dataset.batch(params["batch_size"]))
    global_train_step = train_iteration * params["epochs"] * \
        tf.data.experimental.cardinality(dataset.batch(params["batch_size"]))
    global_log_step = train_iteration * params["epochs"]

    with writer.as_default() if writer is not None else nullcontext():
        for epoch in range(params["epochs"]):
            if params['debug']:
                print(f"Epoch: {epoch}")
            avg_loss = 0
            step_start = global_train_step + epoch * len_epoch
            for step, features in dataset.batch(params["batch_size"]).enumerate(step_start):
                loss_value, grads = model.grad(features, step, params)
                avg_loss += loss_value
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if epoch % params["log_steps"] == 0:
                print(f"Epoch: {epoch}, average loss: "
                      f"{(avg_loss / tf.dtypes.cast(len_epoch, tf.float64)):.4f}")
                stats = calc_stats(model, dataset, scaling_data, params)
                l_step = (epoch + global_log_step) // params['log_steps']
                print_stats(stats, l_step)
                tf.summary.scalar("metrics/loss", loss_value, step=l_step)

        print(f"Epoch: {epoch}, average loss: {(avg_loss / "
              f"tf.dtypes.cast(len_epoch, tf.float64)):.4f}")
        stats = calc_stats(model, dataset, scaling_data, params)
        l_step = (epoch + global_log_step + 1) // params['log_steps']
        print_stats(stats, l_step)
        tf.summary.scalar("metrics/loss", loss_value, step=l_step)
        return stats


def test(params):
    raise NotImplementedError("Test mode not implemented yet.")


def main(params):
    """ Main execution. Creates logging and writer, and launches selected training. """
    if params['dataset'] == "IHDP":
        params["x_bin_size"] = 19
        params["x_cat_size"] = 0 + 19
        params["x_cont_size"] = 6
    if params['dataset'] == "TWINS":
        params["x_bin_size"] = 0
        params["x_cat_size"] = 3
        params["x_cont_size"] = 0

    params["z_size"] = 16
    repetitions = 10

    timestamp = time.strftime("%Y:%m:%d/%X")
    if not params["debug"]:
        logdir = (f"{params['model_dir']}{params['model']}/{params['dataset']}/"
                  f"{params['learning_rate']}/{timestamp}/{params['experiment_name']}")
        writer = tf.summary.create_file_writer(logdir)
    else:
        writer = None

    if params["separate_files"]:
        total_stats = []
        for i in range(repetitions):
            stats = train(params, writer, i)
            total_stats.append(stats)
        total_stats = np.array(total_stats)
        print("Final average results")
        if not params['debug']:
            with writer.as_default():
                print_stats(total_stats.mean(0),
                            params['epochs'] * repetitions // params['log_steps'] + 1)
    else:
        train(params, writer)


if __name__ == "__main__":
    params = parse_arguments()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(gpus)
    if gpus:
        for gpu in gpus:
            set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

    main(params)
