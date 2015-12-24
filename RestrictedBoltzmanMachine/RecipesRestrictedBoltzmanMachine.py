# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 2015
Last update on -

@author: michielstock

Special case of the restricted boltzman machine for working with the recipes data
"""

import numpy as np
from random import shuffle

from RestrictedBoltzmanMachine import RestrictedBoltzmanMachine, sigmoid

class RecipeRestrictedBoltzmanMachine(RestrictedBoltzmanMachine):
    """
    Special version of the vanilla RBM suitable for recipe recommendation
    """
    def __init__(self, ingredients, regions, categories, n_hidden):
        self.ingredients = ingredients
        self.regions = regions
        n_visible = len(ingredients) + len(regions)
        self._bias_visible = np.random.randn(n_visible, 1) / 100
        self._bias_hidden = np.random.randn(n_hidden, 1) / 100
        self._weights = np.random.randn(n_visible, n_hidden) / 100
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.ingredient_hash = {ingr : i for i, ingr in enumerate(ingredients)}
        self.region_hash = {reg : i + len(ingredients) for i, reg in enumerate(regions)}
        self.categories = categories
    
    def recommend_ingredients(self, recipe, top_size=5, region=None, category=None):
        """
        For a given recipe, recommend ingredients that might match
        Optionally, region of the recipe can be fixed as can the category
        to select ingredients from
        """
        n_ingredients = len(self.ingredients)
        # calculate hidden activations
        # without regions
        activation_hidden = self._bias_hidden +\
                    self._weights[[self.ingredient_hash[ingr] for\
                                   ingr in recipe]].sum(0).reshape(-1,1)
        # if regions are included: update activations
        if region is not None:
            activation_hidden += self._weights[[self.region_hash[region]]].reshape(-1,1)
        # transform to probabilities
        p_hidden = sigmoid(activation_hidden)
        # use hidden actions to reconstruct recipe
        p_visible_reconstructed = list(sigmoid((np.dot(self._weights, p_hidden) +\
                                                self._bias_visible).ravel()))
        # recommend ingredients
        recommendations = [(p, ingr, cat) for p, ingr, cat\
                            in zip(p_visible_reconstructed[:n_ingredients],
                                self.ingredients, self.categories) if ingr not in recipe] 
        if category is not None:
            recommendations = [tup for tup in recommendations if tup[2] == category]
        recommendations.sort()
        return recommendations[::-1][:top_size]
        
            