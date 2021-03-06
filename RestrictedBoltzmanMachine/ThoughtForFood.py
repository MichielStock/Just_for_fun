# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 2015
Last update on -

@author: michielstock
michielfmstock@gamil.com

Script to use a pre-trained restriced Boltzmann machine for recipes via
the terminal
"""

from RecipesRestrictedBoltzmannMachine import\
    RecipeRestrictedBoltzmannMachinePretrained, pretty_print_recommendation
import sys
import re

# read arguments in list, recall that first argument is just the file name
arguments = sys.argv

if len(arguments) == 1:  # no arguments
    print('how to use...')
    # to be completed

else:
    input_string = ' '.join(arguments[1:])
    # load model
    recipe_rbm = RecipeRestrictedBoltzmannMachinePretrained(
                                                        'Recipe_parameters/')
    n_ingr_arg = re.findall('-[nN] [0-9]+', input_string)
    region_arg = re.findall('-[rR] \S+', input_string)
    category_arg = re.findall('-[cC] [^-]+', input_string)
    ingr_arg = re.findall('-[iI] [^-]+', input_string)

    if len(n_ingr_arg) > 0:
        n_ingr = int(n_ingr_arg[0][3:])
    else:
        n_ingr = 5
    
    if len(region_arg) > 0:
        region = region_arg[0][3:]
    else:
        region = None

    if len(category_arg) > 0:
        category = category_arg[0][3:].rstrip()
    else:
        category = None
        
    if len(ingr_arg) > 0:
        ingredients = ingr_arg[0][3:].rstrip().split(',')
    else:
        print('No ingredients provided!')
        raise KeyError
        
    recommendations = recipe_rbm.recommend_ingredients(ingredients,
                                    top_size=n_ingr, region=region,
                                    category=category, sample=True)
    pretty_print_recommendation(recommendations)
