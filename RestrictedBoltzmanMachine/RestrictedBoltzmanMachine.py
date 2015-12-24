# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 16:28:02 2015
Last update on Mon Dec 22 2015

@author: michielstock

Implementation of a restricted boltzman machine, uses an approximate scheme
for training
"""

import numpy as np
from random import shuffle

sigmoid = lambda x : 1 / (1 + np.exp(-x))

class RestrictedBoltzmanMachine:
    """
    Basic restricted boltzman machine
    """

    def __init__(self, n_visible, n_hidden):
        self._bias_visible = np.random.randn(n_visible, 1) / 100
        self._bias_hidden = np.random.randn(n_hidden, 1) / 100
        self._weights = np.random.randn(n_visible, n_hidden) / 100
        self.n_visible = n_visible
        self.n_hidden = n_hidden

    def visible_given_hidden(self, hidden, sample=True):
        """
        Samples the visible units, given hidden variables
        if sample is true: return binary samples, else return probabilities
        """
        probs = sigmoid(np.dot(self._weights, hidden.T).T
                                        + self._bias_visible.T)
        if sample:
            return np.random.binomial(1, probs)
        else:
            return probs

    def hidden_given_visible(self, visible, sample=True):
        """
        Samples the hidden units, given visible variables
        if sample is true: return binary samples, else return probabilities
        """
        probs = sigmoid(np.dot(visible, self._weights) + self._bias_hidden.T)
        if sample:
            return np.random.binomial(1, probs)
        else:
            return probs

    def sample_fantasies(self, visible_seed, sampling_steps=2):
        """
        Create some fantasy particles using (approximate) Gibs sampling
        """
        visible = visible_seed.copy()
        for i in range(sampling_steps):
            hidden = self.hidden_given_visible(visible)
            visible = self.visible_given_hidden(hidden)
            return visible, hidden

    def update_weights(self, visible, learning_rate, sampling_steps):
        """
        Updates weigths
        """
        lr = learning_rate / visible.shape[0]
        hidden = self.hidden_given_visible(visible)
        fantasy_vis, fantasy_hid = self.sample_fantasies(visible,
                                                         sampling_steps)
        self._weights += lr * (np.dot(visible.T, hidden) -
                                      np.dot(fantasy_vis.T, fantasy_hid))
        self._bias_visible += lr * (visible.sum(0)
                                    - fantasy_vis.sum(0)).reshape((-1,1))
        self._bias_hidden += lr * (hidden.sum(0)
                                    - fantasy_hid.sum(0)).reshape((-1,1))
    
    def train_stochastic_gradient_ascent(self, X, learning_rate=0.01,
                                         sampling_steps=1, iterations=1000,
                                         minibatch_size=10):
        n_instances = len(X)
        instances = range(n_instances)
        mse_reconstr = []
        for iteration in range(iterations):
            shuffle(instances)
            start = 0
            while start < n_instances:
                minibatch = X[start:start+minibatch_size]
                self.update_weights(minibatch,
                                    learning_rate, sampling_steps)
                start += minibatch_size
                vis_reconstr, _ = self.sample_fantasies(minibatch, 1)
                mse_reconstr.append(np.mean((vis_reconstr - minibatch)**2))
        return mse_reconstr
        
    def train_C1(self, X, learning_rate=0.01, iterations=1000,
                 minibatch_size=10, momentum=0.9):
        """
        Trains the restricted boltzman machine using Hinton's recommended
        method making use of a single Gibbs step.
        Uses momentum to speed up learning
        """
        n_instances = len(X)
        instances = range(n_instances)
        mse_reconstr = []
        dW = 0
        da = 0
        db = 0
        for iteration in range(iterations):
            shuffle(instances)
            start = 0
            while start < n_instances:
                visible = X[start:start+minibatch_size]
                lr = learning_rate / visible.shape[0]  # correct for sample
                # size
                # sample corresponding hidden
                hidden = self.hidden_given_visible(visible)
                # get probabilities visible
                reconstr = self.visible_given_hidden(hidden, sample=False)
                hidden_CD2 = self.hidden_given_visible(reconstr, sample=False)
                # new directions
                dW = (1 - momentum) * lr * (np.dot(visible.T, hidden) -\
                        np.dot(reconstr.T, hidden)) + momentum * dW
                da = (1 - momentum) * lr * (visible.sum(0) -\
                        reconstr.sum(0)).reshape(-1,1) + momentum * da
                db = (1 - momentum) * lr * (hidden.sum(0) -\
                        hidden_CD2.sum(0)).reshape(-1,1) + momentum * db
                # update weights
                self._weights += dW
                self._bias_visible += da
                self._bias_hidden += db
                # update start
                start += minibatch_size
                # error
                mse_reconstr.append(np.mean((reconstr - visible)**2))
        return mse_reconstr
        
        
if __name__ == '__main__':

    rbm = RestrictedBoltzmanMachine(100, 10)
    pattern = np.kron(np.random.binomial(1, 0.5, (100, 10)), np.ones((1, 10)))
    error = rbm.train_C1(pattern, learning_rate=0.001, iterations=1000,
                              sampling_steps=1)
                                                 
    print rbm.sample_fantasies(np.random.binomial(1, 0.5,(5, 100)))
