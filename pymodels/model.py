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
import torch.optim as optim
import torch.nn.functional as F


use_cuda = torch.cuda.is_available()

if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
    device = torch.device('cuda')
else:
    dtype = torch.FloatTensor
    ltype = torch.LongTensor
    device = torch.device('cpu')


def get_batch(X, Y, i, batch_size):
    x = X[:, i:i+batch_size].transpose(0, 1)
    y = Y[:, i:i+batch_size].transpose(0, 1)
    return x, y


def init_model_single_layer(model_path, num_keys):
    model = torch.nn.Linear(in_features=num_keys, out_features=num_keys)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    return model


def init_model_two_layers(model_path, num_keys, H=30):
    model = torch.nn.Sequential(
        #torch.nn.Linear(num_keys, H),
        #torch.nn.ReLU(),
        #torch.nn.Linear(H, num_keys),
        #torch.nn.Sigmoid(),
        torch.nn.Linear(num_keys, num_keys),
        torch.nn.Sigmoid(),
    ).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    #if use_cuda:
    #    print("move model to gpu")
    #    #model.cuda()

    return model


def store_model(model_path, model):
    torch.save(model.state_dict(), model_path)


def train(model_path, layers=2):

    X = np.load("X.npy")
    Y = np.load("Y.npy")
    # limit target for cross entropy usage
    Y[X < 0] = 0
    Y[X > 1] = 1

    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    N = X.size(1)
    num_keys = X.size(0)

    perm = torch.randperm(N)
    X = X[:, perm]
    Y = Y[:, perm]

    # Define model
    if layers == 1:
        model = init_model_single_layer(model_path, num_keys)
    elif layers == 2:
        model = init_model_two_layers(model_path, num_keys)

    #loss_fn = F.smooth_l1_loss
    #loss_fn = torch.nn.MSELoss(size_average=False)
    #loss_fn = torch.nn.BCEWithLogitsLoss() # combines sigmoid with binary cross entropy
    loss_fn = torch.nn.BCELoss()

    i = 0
    batch_size = 32
    optimizer = optim.Adam(model.parameters())

    for _ in range(10000):
        # Get data
        batch_x, batch_y = get_batch(X, Y, i, batch_size)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        i += batch_size
        if i > X.size(1):
            i = 0

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        output = loss_fn(model(batch_x), batch_y)
        loss = output.item()
        sys.stdout.write("loss = {}\r".format(loss))

        # Backward pass
        output.backward()

        # Apply gradients
        optimizer.step()

        # Stop criterion
        #if loss < 1e-3:
        #    break

    batch_x = X.transpose(0, 1)
    batch_y = Y.transpose(0, 1)
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    pred = model(batch_x)
    loss = loss_fn(pred, batch_y)
    print("loss = {}".format(loss.item()))

    pred = pred.transpose(0, 1)
    np.save("P.npy", pred.cpu().detach().transpose(0, 1).numpy())

    store_model(model_path, model)


def predict(model_path, layers=2):
    X = np.load("X.npy")
    X = torch.from_numpy(X)
    num_keys = X.size(0)

    # Define model
    if layers == 1:
        model = init_model_single_layer(model_path, num_keys)
    elif layers == 2:
        model = init_model_two_layers(model_path, num_keys)

    batch_x = X.transpose(0, 1)
    batch_x = batch_x.to(device)
    pred = model(batch_x)

    np.save("P.npy", pred.cpu().detach().transpose(0, 1).numpy())


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

