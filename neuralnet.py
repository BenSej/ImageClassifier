# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques for the fall 2021 semester
# Modified by Kaiwen Hong for the Spring 2022 semester

"""
This is the main entry point for MP2. You should only modify code
within this file and neuralnet.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNet(nn.Module):

    def __init__(self, lrate, loss_fn, in_size, out_size):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {device} device')

        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @param l(x,y) an () tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 2 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size

        """

        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(in_size, 32),
            nn.ReLU(),
            nn.Linear(32, out_size)
        )
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lrate)

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        return self.model(x)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """

        self.optimizer.zero_grad()
        loss = self.loss_fn(self.forward(x), y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def fit(train_set, train_labels, dev_set, n_iter, batch_size=100):
    """ Fit a neural net. Use the full batch size.

    @param train_set: an (N, out_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of epoches of training
    @param batch_size: size of each batch to train on. (default 100)

    NOTE: This method _must_ work for arbitrary M and N.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """

    losses = []
    net = NeuralNet(.025, nn.CrossEntropyLoss(), len(train_set[0]), 2)
    train_set = (train_set - train_set.mean()) / train_set.std()
    for i in range(5):
        j = 0
        while j < 75:
            batch = train_set[batch_size * j: batch_size * (j + 1)]
            labels = train_labels[batch_size * j: batch_size * (j + 1)]
            losses.append(net.step(batch, labels))
            j += 1
    dev_set = (dev_set - train_set.mean()) / train_set.std()
    dev_labels = np.argmax(net(dev_set).detach(), axis=1)
    return losses, dev_labels, net
