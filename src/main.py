import argparse
import numpy as np
import logging
import os
import re
import shutil
import sys
import tensorflow as tf
import time

from causal_real_nvp import CRNVP
from cenf import CENF
from cevae import CEVAE
from contextlib import nullcontext
from dataset import IHDP, IHDP_LARGE, TWINS, SHAPES, SPACE, SPACE_NO_GRAV
from evaluation import calc_stats
from normalizing_causal_flow import NCF
from planar_flow import PlanarFlow
from radial_flow import RadialFlow
from sylvester_flow import SylvesterFlow
from tar_net import TARNET

VALID_DATASETS = ["IHDP", "IHDP_LARGE", "TWINS", "SPACE", "SPACE_NO_GRAV"]
VALID_MODELS = ["CEVAE", "CRNVP", "NCF", "PlanarFlow", "RadialFlow",
                "SylvesterFlow", "TARNET"]
VALID_FLOWS = ["AffineCoupling", "NLSCoupling"]

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

    parser = argparse.ArgumentParser(description="Causal effect Normalizing "
                                                 "Flow trainer")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default=32)")
    parser.add_argument("--dataset", type=str, default="IHDP",
                        help="Dataset used. "
                        f"Available datasets are: {VALID_DATASETS} "
                        "(default: IHDP)")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Turn on debugging mode. What it does now is turn"
                        " off summary writer and print a lott of stuff.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training iterations (default: 100)")
    parser.add_argument("--experiment_name", type=str, default="",
                        help="Name of experiment used for results folder. "
                        "Disabled in debug mode")
    parser.add_argument("--feature_maps", type=int, default=200,
                        help="Number of nodes in hidden fully connected layers"
                        " or number of filters in convolutional layers. "
                        "(default: 200)")
    parser.add_argument("--flow_type", type=str, default="affine_coupling",
                        help="Type of flow functions in pure flow model")
    parser.add_argument("--gclip", action="store_true", default=False,
                        help="Turn on gradient clipping to a maximum of |1|")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate of the optmiser (default: 1e-4)")
    parser.add_argument("--log_steps", type=int, default=10,
                        help="Save/print log every n steps (default: 10)")
    parser.add_argument("--mode", type=str, default="train",
                        help="Switch between train and test mode "
                        "(default: train)")
    parser.add_argument("--model", type=str, default="CEVAE",
                        help="The type of model used for predictions. "
                        f"Available models are: {VALID_MODELS} "
                        "(default: CEVAE)")
    parser.add_argument("--model_dir", type=str,
                        default="/home/mdegroot/logs/",
                        help="The directory to save the model to "
                        "(default: ~/logs/)")
    parser.add_argument("--n_flows", type=int, default=2,
                        help="Number of flows in the flow models (default: 2)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of samples to draw when performing an"
                        "intervention, if sampling is used (default: 100)")
    parser.add_argument("--path_dataset", type=str, default="",
                        help="Pass the path of the dataset folder if not "
                        "using default path.")
    parser.add_argument("--separate_files", action="store_true", default=False,
                        help="Switch to training the model on each data file "
                        "separately instead of everything at once")

    args = parser.parse_args()

    if args.debug:
        tf.random.set_seed(0)
        args.log_steps = 1
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        if args.experiment_name is None or\
           args.experiment_name == "":
            raise ValueError("Experiment name is required if debug mode is "
                             "disabled")
        if not re.match(r'^[A-Za-z0-9_]+$', args.experiment_name):
            raise ValueError("Experment name may only contain alphanumerical "
                             "and underscore")

    args.model = args.model.upper()
    if args.model.lower() not in [i.lower() for i in VALID_MODELS]:
        raise NotImplementedError(f"Model {args.model} is not implemented")
    else:
        args.model = [i for i in VALID_MODELS if
                      i.lower() == args.model.lower()][0]

    if args.model == 'NCF':
        if args.flow_type not in VALID_FLOWS:
            raise NotImplementedError("Flow type invalid. Must be one of: "
                                      f"{VALID_FLOWS}")
    else:
        args.flow_type = ''

    args.dataset = args.dataset.upper()
    if args.dataset not in VALID_DATASETS:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented")

    if not args.mode == "train":
        # raise NotImplementedError("Only training mode is implemented")
        pass

    if args.model == "cevae" or args.model == "tarnet":
        args.n_flows = 0

    for arg, value in vars(args).items():
        print(f"Argument: {arg:<22} {value:<5}")
    print()

    return vars(args)


def print_stats(stats, index, training=False, loss=None):
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
    training : bool
        Flag to swithc between training and testing  logs. Uses different
        tags in tensorflow.
    """

    if training:
        prefix = "train"
    else:
        prefix = "test"

    print(f"\nDataset: {prefix}\nAverage ite: {stats[0]:.4f}, "
          f"abs ate: {stats[1]:.4f}, "
          f"pehe: {stats[2]:.4f},\n"
          f"y_error factual: {stats[3][0]:.4f}, "
          f"y_error counterfactual {stats[3][1]:.4f}\n")

    tf.summary.scalar(f"metrics/{prefix}/ite", stats[0], step=index)
    tf.summary.scalar(f"metrics/{prefix}/ate", stats[1], step=index)
    tf.summary.scalar(f"metrics/{prefix}/pehe", stats[2], step=index)
    tf.summary.scalar(f"metrics/{prefix}/y_factual", stats[3][0], step=index)
    tf.summary.scalar(f"metrics/{prefix}/y_counterfactual", stats[3][1],
                      step=index)

    if loss is not None:
        ite_ratio = stats[0] / loss
        ate_ratio = stats[1] / loss
        pehe_ratio = stats[2] / loss
        print(f"Ratios of stats vs loss on {prefix} set:\n"
              f"ITE ratio: {ite_ratio}\n"
              f"ATE ratio: {ate_ratio}\nPEHE ratio: {pehe_ratio}")
        tf.summary.scalar(f"ratios/{prefix}/ite_ratio", ite_ratio, step=index)
        tf.summary.scalar(f"ratios/{prefix}/ate_ratio", ate_ratio, step=index)
        tf.summary.scalar(f"ratios/{prefix}/pehe_ratio", pehe_ratio, step=index)


def grad(model, features, gradient_clipping, step, debug):
    with tf.GradientTape() as tape:
        x = tf.concat([features[0], features[1]], -1)
        t = features[2]
        y = features[4]
        output = model(x, t, y, step, training=True)
        loss = model.loss(features, *output, step)
    if debug:
        print(f"Forward pass complete, step: {step}")
    gradients = tape.gradient(loss, model.trainable_variables)
    if gradient_clipping:
        gradients = [(tf.clip_by_value(grad, -1.0, 1.0)) for grad in gradients]
    return loss, gradients


def set_inputs(model, dataset):
    data = dataset.__iter__().next()
    model._set_inputs(tf.concat([data[0], data[1]], -1), data[2], data[4])


def load_weights(model_dir, model_name, dataset, learning_rate, model):
    path = (f"{model_dir}{model_name}/{dataset}/{learning_rate}/")
    experiment_dir = [dir_name for dir_name in os.walk(path) if
                      dir_name[1] and
                      dir_name[1][0] == params['experiment_name']][0]
    experiment_dir = os.path.join(experiment_dir[0], experiment_dir[1][0])
    experiment_name = os.path.join(experiment_dir, "model_0")
    model.load_weights(experiment_name)


def train(params, writer, logdir, train_iteration=0):
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

    logdir : str
        Directory in which to save the model

    train_iteration : int, optional
        Incrementor that should count the repetitions of times train()
        is called.
        Used to correcly log a series of trainings in the writer.

    Returns
    -------
    stats : tuple
        Tuple of the statistics that were calculated after the last epoch.
    """

    debug = params['debug']
    cardinality = tf.data.experimental.cardinality
    data_loader = eval(params['dataset'])
    if params["path_dataset"] is "":
        data = data_loader(params, separate_files=params['separate_files'],
                           file_index=train_iteration)
    else:
        data = data_loader(params, path_data=params['path_dataset'],
                           separate_files=params['separate_files'],
                           file_index=train_iteration)
    train_dataset, test_dataset = data
    scaling_data = params['scaling_data']

    len_dataset = cardinality(train_dataset)
    len_dataset = cardinality(test_dataset)

    model = eval(params['model'])(**params)
    # model.encoder.annealing_factor = 1.1
    set_inputs(model, train_dataset)

    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    len_epoch = cardinality(train_dataset.batch(params["batch_size"]))
    global_log_step = train_iteration * params["epochs"]
    global_train_step = global_log_step * len_epoch

    with writer.as_default() if writer is not None else nullcontext():
        for epoch in range(params["epochs"]):
            train_dataset = train_dataset.shuffle(len_dataset)
            if params['debug']:
                print(f"Epoch: {epoch}")
            avg_loss = 0
            step_start = global_train_step + epoch * len_epoch
            for step, features in train_dataset.batch(params["batch_size"])\
                    .enumerate(step_start):
                step = tf.constant(step)
                loss_value, grads = grad(model, features, params['gclip'],
                                         step, debug)
                avg_loss += loss_value
                optimizer.apply_gradients(zip(grads,
                                              model.trainable_variables))
            avg_loss /= tf.cast(len_epoch, tf.float64)

            if epoch % params["log_steps"] == 0:
                test_dataset = test_dataset.shuffle(len_dataset)
                print(f"Epoch: {epoch}, average training loss: {avg_loss:.4f}")
                l_step = (epoch + global_log_step) // params['log_steps']
                tf.summary.scalar("metrics/train/loss", avg_loss,
                                  step=l_step)

                stats = calc_stats(model, train_dataset, scaling_data, params)
                print_stats(stats, l_step, training=True, loss=loss_value)

                stats = calc_stats(model, test_dataset, scaling_data, params)
                print_stats(stats, l_step, training=False, loss=loss_value)
                if not params["debug"]:
                    model.save_weights(f"{logdir}/model_{train_iteration}")

        print(f"Epoch: {epoch}, average loss: {avg_loss:.4f}")
        l_step = (epoch + global_log_step + 1) // params['log_steps']
        tf.summary.scalar("metrics/train/loss", loss_value, step=l_step)

        stats_train = calc_stats(model, train_dataset, scaling_data, params)
        print_stats(stats_train, l_step, training=True)

        stats_test = calc_stats(model, test_dataset, scaling_data, params)
        print_stats(stats_test, l_step, training=False)

        model.summary()
        return stats_train, stats_test


def test(params, writer, logdir):
    """
    We select an experiment based on an earlier experiment name, which sort of implies
    that those have to be unique for each model and dataset combination. We just assume
    that that is the case. We do have to add something to differentiate the dataset
    because we need the training dataset to identify the trained model and a test dataset
    for the test model.

    How do we want to log this? It doesn't really yield a graph because the dataset is
    constant. We could still make a graph but over different batches or something?


    Parameters
    ----------
    params : dict
        Dictionary of all settings needed to define the training procedure.
        Use the parse_arguments() function to generate a dictionary with the
        required fileds

    writer : tensorflow.summary.writer
        The writer opbject to which all results and logs are written.
        Can be None

    logdir : str
        Directory in which to save the model

    Returns
    -------
    stats : tuple
        Tuple of the statistics that were calculated after the last epoch.
    """
    debug = params['debug']
    cardinality = tf.data.experimental.cardinality
    data_loader = eval(params['dataset'])
    if params["path_dataset"] is "":
        dataset, _ = data_loader(params, ratio=0.00001, test=True,
                                 separate_files=params['separate_files'])
    else:
        dataset, _ = data_loader(params, ratio=0.00001, test=True,
                                 path_data=params['path_dataset'],
                                 separate_files=params['separate_files'])
    
    scaling_data = params['scaling_data']

    len_dataset = cardinality(dataset)
    len_epoch = cardinality(dataset.batch(params["batch_size"]))

    model = eval(params['model'])(**params)
    set_inputs(model, dataset)
    dataset_name =  params['dataset']
    if params['model'] == "NCF":
        model_name = f"{params['model']}/{params['flow_type']}"
    else:
        model_name = params['model']
    load_weights(params['model_dir'], model_name, dataset_name,
                 params['learning_rate'], model)
    print("Testing dataset")    
    with writer.as_default() if writer is not None else nullcontext():
        dataset = dataset.shuffle(len_dataset)
        stats = calc_stats(model, dataset, scaling_data, params)
        print_stats(stats, 0, training=False)


def main(params):
    """ Main execution.

    Creates logging and writer, and launches selected training.
    """
    repetitions = 100 if params["dataset"] == "IHDP_LARGE" else 10

    timestamp = time.strftime("%Y:%m:%d/%X")
    if not params["debug"]:
        if params['flow_type']:
            flow_type_dir = params['flow_type'] + '/'
        else:
            flow_type_dir = None
        if params['mode'] == 'test':
            experiment_name = f"test/{params['experiment_name']}"
        else:
            experiment_name = params['experiment_name']
        logdir = (f"{params['model_dir']}"
                  f"{params['model']}/"
                  f"{flow_type_dir or ''}"
                  f"{params['dataset']}/"
                  f"{params['learning_rate']}/"
                  f"{timestamp}/"
                  f"{experiment_name}")
        writer = tf.summary.create_file_writer(logdir)
    else:
        logdir = None
        writer = None

    if params['mode'] == 'test':
        test(params, writer, logdir)

    elif params['mode'] == 'train':
        if params["separate_files"]:
            total_stats_train = []
            total_stats_test = []
            for i in range(repetitions):
                stats_train, stats_test = train(params, writer, logdir, i)
                total_stats_train.append(stats_train)
                total_stats_test.append(stats_test)
            total_stats_train = np.array(total_stats_train)
            total_stats_test = np.array(total_stats_test)
            print("Final average results")
            if not params['debug']:
                with writer.as_default():
                    print_stats(total_stats_train.mean(0),
                                params['epochs'] * repetitions //
                                params['log_steps'] + 1, training=True)
                    print_stats(total_stats_test.mean(0),
                                params['epochs'] * repetitions //
                                params['log_steps'] + 1, training=False)
        else:
            train(params, writer, logdir)


if __name__ == "__main__":

    params = parse_arguments()
    set_vdc = tf.config.experimental.set_virtual_device_configuration
    vdc = tf.config.experimental.VirtualDeviceConfiguration
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # set_vdc(gpu, [vdc(memory_limit=9000)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs, ",
                  len(logical_gpus), "Logical GPUs")
        with tf.device('device:GPU:0'):
            main(params)
    else:
        main(params)
