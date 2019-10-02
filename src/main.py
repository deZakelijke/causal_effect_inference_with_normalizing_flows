import argparse
import numpy as np
import sys
import torch
from cevae import train_cevae


def parse_arguments():
    parser = argparse.ArgumentParser(description="Causal effect Normalizing Flow trainer")
    parser.add_argument("--mode", type=str, default="test",
                        help="Switch between train and test mode (default=test)")

    args = parser.parse_args()

    for arg, value in vars(args).items():
        print(f"Argument: {arg}, value: {value}")
    print()

    return args

def train(config):
    train_cevae(config)

def test(config):
    pass

if __name__ == "__main__":
    config = parse_arguments()

    if config.mode == "train":
        train(config)
    else:
        test(config)
