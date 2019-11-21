import argparse
import numpy as np
import sys
import tensorflow as tf
from cevae import train_cevae

VALID_MODELS = ["cevae"]
VALID_DATASETS = ["IHDP"]

tf.keras.backend.set_floatx('float64')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Causal effect Normalizing Flow trainer")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default=16")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate of hte optmiser (default: 1e-4)")
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
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save/print log every n steps (default: 10)")

    args = parser.parse_args()

    if not args.model in VALID_MODELS:
       raise NotImplementedError(f"Model {args.model} is not implemented")

    if not args.dataset in VALID_DATASETS:
       raise NotImplementedError(f"Dataset {args.dataset} is not implemented")

    for arg, value in vars(args).items():
        print(f"Argument: {arg:<16} {value:<5}")
    print()

    return vars(args)

def train(config):
    if config["model"] == "cevae":
        train_cevae(config)

def test(config):
    pass

if __name__ == "__main__":
    params = parse_arguments()

    if params["mode"] == "train":
        train(params)
    else:
        test(params)
