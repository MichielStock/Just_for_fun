# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 2015
Last update on Tue Dec 29 2015

@author: michielstock
michielfmstock@gamil.com

Special case of the restricted boltzmann machine for working with
the recipes data

Implementation of a special class of the restricted boltzmann machine for
recommending recipes. Allows for saving the model.

The main part trains a RBM on the recipe dataset and saves this
"""

import numpy as np
import pandas as pd

from RestrictedBoltzmannMachine import RestrictedBoltzmannMachine, sigmoid


class RecipeRestrictedBoltzmannMachine(RestrictedBoltzmannMachine):
    """
    Special version of the vanilla RBM suitable for recipe recommendation
    """
    def __init__(self, ingredients, regions, categories, n_hidden):
        self.ingredients = list(ingredients)
        self.regions = list(regions)
        self.categories = list(categories)
        n_visible = len(ingredients) + len(regions)
        self._bias_visible = np.random.randn(n_visible, 1) / 100
        self._bias_hidden = np.random.randn(n_hidden, 1) / 100
        self._weights = np.random.randn(n_visible, n_hidden) / 100
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.ingredient_hash = {ingr: i for i, ingr in enumerate(ingredients)}
        self.region_hash = {reg: i + len(ingredients) for i,
                            reg in enumerate(regions)}

    def recommend_ingredients(self, recipe, top_size=5, region=None,
                              category=None):
        """
        For a given recipe, recommend ingredients that might match
        Optionally, region of the recipe can be fixed as can the category
        to select ingredients from
        """
        n_ingredients = len(self.ingredients)
        # calculate hidden activations
        # without regions
        activation_hidden = self._bias_hidden +\
            self._weights[[self.ingredient_hash[ingr] for
                            ingr in recipe]].sum(0).reshape(-1, 1)
        # if regions are included: update activations
        if region is not None:
            activation_hidden += self._weights[[
                    self.region_hash[region]]].reshape(-1, 1)
        # transform to probabilities
        p_hidden = sigmoid(activation_hidden)
        # use hidden actions to reconstruct recipe
        p_visible_reconstructed = list(sigmoid((np.dot(self._weights,
                                    p_hidden) + self._bias_visible).ravel()))
        # recommend ingredients
        recommendations = [(p, ingr, cat) for p, ingr, cat
                            in zip(p_visible_reconstructed[:n_ingredients],
                                self.ingredients, self.categories) if
                                ingr not in recipe]
        if category is not None:
            recommendations = [tup for tup in recommendations if
                        tup[2] == category]
        recommendations.sort()
        return recommendations[::-1][:top_size]

    def save(self, directory):
        """
        Saves the model parameters to a given directory
        Model can be loaded with the class
            'RecipeRestrictedBoltzmannMachinePretrained'
        """
        pd.DataFrame(self._weights, index=self.ingredients + self.regions
                    ).to_csv('{0}_{1}'.format(directory, 'weigths'))
        pd.DataFrame(self._bias_visible, index=self.ingredients + self.regions
                    ).to_csv('{0}_{1}'.format(directory, 'bias_visible'))
        pd.DataFrame(self._bias_hidden).to_csv(
                        '{0}_{1}'.format(directory, 'bias_hidden'))
        pd.DataFrame(self.categories, index=self.ingredients).to_csv(
                        '{0}_{1}'.format(directory, 'categories'))


class RecipeRestrictedBoltzmannMachinePretrained(
        RecipeRestrictedBoltzmannMachine):
    """
    Subclass of the recipe restricted boltzmann machine which can be loaded
    from a previously trained model
    """

    def __init__(self, directory):
        """
        Loads the recipe RBM from a repository containing the trained
        model
        """
        # load the data
        weights_dataframe = pd.DataFrame.from_csv('{0}_{1}'.format(directory,
                                                  'weigths'))
        bias_hidden_dataframe = pd.DataFrame.from_csv(
                            '{0}_{1}'.format(directory, 'bias_hidden'))
        bias_visible_dataframe = pd.DataFrame.from_csv(
                            '{0}_{1}'.format(directory, 'bias_visible'))
        categories_dataframe = pd.DataFrame.from_csv(
                            '{0}_{1}'.format(directory, 'categories'))
        # setup the RBM
        self.ingredients = list(weights_dataframe.index[:-11])
        self.regions = list(weights_dataframe.index[:-11])
        self.categories = list(categories_dataframe['0'].tolist())
        self.n_visible, self.n_hidden = weights_dataframe.shape
        self._bias_visible = bias_visible_dataframe.values
        self._bias_hidden = bias_hidden_dataframe.values
        self._weights = weights_dataframe.values
        self.ingredient_hash = {ingr: i for i, ingr in
                                                enumerate(self.ingredients)}
        self.region_hash = {reg: i + len(self.ingredients) for i,
                            reg in enumerate(self.regions)}
                            
def pretty_print_recommendation(recommendations):
    """
    Prints the recmmended ingredients provided by the recipes RBM
    """
    for p, ingr, cat in recommendations:
        print('{0} ({1}) : {2}'.format(ingr, cat, p))

if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt

    # load the data
    print('LOADING DATA')
    print('_' * 50)
    print('')

    recipes = pd.DataFrame.from_csv('Recipes_with_origin.csv')
    print(recipes.head())

    categories = pd.DataFrame.from_csv('categories.csv')
    print(categories.head())

    print('')

    ingredients = recipes.columns[:-11]
    regions = recipes.columns[-11:]

    # initializing and training the model
    print('TRAINING THE MODEL')
    print('_' * 50)
    print('')

    rbm = RecipeRestrictedBoltzmannMachine(ingredients, regions, n_hidden=250,
                                          categories=list(categories.category))

    # train in some different phases
    error = rbm.train_C1(recipes.values, learning_rate=0.1,
                                            iterations=200, minibatch_size=20)
    print('first training step finished')
    error += rbm.train_C1(recipes.values, learning_rate=0.05,
                                          iterations=200, minibatch_size=20)
    print('second training step finished')
    error += rbm.train_C1(recipes.values, learning_rate=0.01,
                                          iterations=200, minibatch_size=20,
                                          momentum=0.6)
    print('third training step finished')

    # plot learning and parameters
    plt.plot(error)
    plt.title('Reconstruction error')
    plt.xlabel('iteration')
    plt.ylabel('MSE')

    plt.imshow(rbm._weights, interpolation='nearest')

    # initializing and training the model
    print('SAVING THE MODEL')
    print('_' * 50)
    print()

    rbm.save('Recipe_parameters/')

    print('TESTING THE MODEL')
    print('_' * 50)
    print('')

    print("yogurt, cucumber, mint")
    pretty_print_recommendation(rbm.recommend_ingredients(['yogurt',
                                'cucumber', 'mint'], top_size=10))
    print('')

    print("meat, tomato, basil (recommend spice)")
    pretty_print_recommendation(rbm.recommend_ingredients(['meat', 'tomato',
                                     'basil'], top_size=10, category='spice'))
    print('')

    print("bean, beef, potato (make South Asian)")
    pretty_print_recommendation(rbm.recommend_ingredients(['bean', 'beef',
                                'potato'], top_size=10, region='SouthAsian'))
