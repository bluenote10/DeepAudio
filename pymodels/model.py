#!/usr/bin/env python
"""
Pytorch model
"""

from __future__ import print_function

import argparse
import os
import sys
import numpy as np

import torch
import torch.autograd
import torch.nn.functional as F


def get_batch(X, Y, i, batch_size):
    x = X[:, i:i+batch_size].transpose(0, 1)
    y = Y[:, i:i+batch_size].transpose(0, 1)
    return x, y


def init_model(model_path, num_keys):
    model = torch.nn.Linear(in_features=num_keys, out_features=num_keys)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    return model


def store_model(model_path, model):
    torch.save(model.state_dict(), model_path)


def train(model_path):

    X = np.load("X.npy")
    Y = np.load("Y.npy")
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    num_keys = X.size(0)

    # Define model
    model = init_model(model_path, num_keys)

    i = 0
    batch_size = 32
    for _ in range(10000):
        # Get data
        batch_x, batch_y = get_batch(X, Y, i, batch_size)
        i += batch_size
        if i > X.size(1):
            i = 0

        # Reset gradients
        model.zero_grad()

        # Forward pass
        output = F.smooth_l1_loss(model(batch_x), batch_y)
        loss = output.item()
        sys.stdout.write("loss = {}\r".format(loss))

        # Backward pass
        output.backward()

        # Apply gradients
        for param in model.parameters():
            param.data.add_(-0.1 * param.grad.data)

        # Stop criterion
        if loss < 1e-3:
            break

    batch_x = X.transpose(0, 1)
    batch_y = Y.transpose(0, 1)
    pred = model(batch_x)
    loss = F.smooth_l1_loss(pred, batch_y)
    print("loss = {}".format(loss.item()))

    pred = pred.transpose(0, 1)
    np.save("P.npy", pred.detach().transpose(0, 1).numpy())

    store_model(model_path, model)


def predict(model_path):
    X = np.load("X.npy")
    X = torch.from_numpy(X)
    num_keys = X.size(0)

    # Define model
    model = init_model(model_path, num_keys)

    batch_x = X.transpose(0, 1)
    pred = model(batch_x)

    np.save("P.npy", pred.detach().transpose(0, 1).numpy())


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predict-only",
        action='store_true',
        help="Run prediction only",
    )
    parser.add_argument(
        "--model",
        help="Model path",
        required=True,
    )
    args = parser.parse_args(args)
    return args


def main():
    args = parse_args()

    if not args.predict_only:
        train(args.model)

    predict(args.model)


if __name__ == "__main__":
    main()

