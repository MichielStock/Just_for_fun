# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 16:28:02 2015

@author: michielstock

Implementation of a restricted boltzman machine
"""

import numpy as np

sigmoid = lambda x : 1 / (1 + np.exp(-x))

class RestrictedBoltzmanMachine:
    """
    Basic restricted boltzman machine
    """

    def __init__(self, n_visible, n_hidden):
        self._bias_visible = np.random.randn(n_visible, 1) / 10
        self._bias_hidden = np.random.randn(n_hidden, 1) / 10
        self._weights = np.random.randn(n_visible, n_hidden) / 10
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
    def visible_given_hidden(self, hidden, sample=True):
        """
        Samples the visible units, given hidden variables
        if sample is true: return binary samples, else return probabilities
        """
        probs = sigmoid(np.dot(self._weights, hidden) + self._bias_visible)
        if sample:
            return np.random.binomial(1, probs)
        else:
            return probs
