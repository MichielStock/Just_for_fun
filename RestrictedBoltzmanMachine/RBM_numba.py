# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 13:05:28 2015

@author: michielstock
"""

import numpy as np
from random import shuffle
import numba

@numba.jit
def sample_fantasies_numba(rbm, visible_seed, sampling_steps=2):
    """
    Create some fantasy particles using (approximate) Gibs sampling
    """
    visible = visible_seed.copy()
    for i in range(sampling_steps):
        hidden = rbm.hidden_given_visible(visible)
        visible = rbm.visible_given_hidden(hidden)
    return visible, hidden