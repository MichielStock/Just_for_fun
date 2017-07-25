"""
Created on Thursday 20 July 2017
Last update: Friday 21 July 2017

@author: Michiel Stock
michielfmstock@gmail.com

Downloads and parses the urine metabolomics dataset from the Human Metabolomics
Database (http://www.hmdb.ca/).
"""

import xmltodict
import json

def unlist(potential_list):
    """
    Turns possible nested list in a list
    """
    real_list = []
    for el in list(potential_list):
        if type(el) is list:
            real_list += el
        else:
            real_list.append(el)
    return real_list

# load the metabolites dataset as a dictonary
with open('Data/urine_metabolites.xml') as fh:
    urine_metabolites_db = xmltodict.parse(fh.read())['hmdb']

# clean the dataset in a nice format
urine_metabolites_data = {}

# simple properties to collect:
simple_fields = set([
    'description',
    'chemical_formula',
    'smiles',
    'average_molecular_weight'])

for metabolite in urine_metabolites_db['metabolite']:
    metabolite_data = {}
    # get name
    name = metabolite['name']
    # simple properties
    for k, v in metabolite.items():
        if k in simple_fields:
            metabolite_data[k] = v
    # chemical taxonomy of the metabolite
    if 'taxonomy' in metabolite:
        for k in ['kingdom', 'class', 'sub_class']:
            metabolite_data[k] = metabolite['taxonomy'][k]
            # some ontological information
    if 'ontology' in metabolite:
        biofunctions = metabolite['ontology']['biofunctions']
        if biofunctions is not None:
            metabolite_data['biofunctions'] = list(set(unlist(biofunctions.values())))
    # pathways
    if metabolite['pathways'] is not None:
        pathways = []
        for pw_dict in unlist(metabolite['pathways'].values()):
            pathways.append(pw_dict['name'])
        metabolite_data['pathways'] = pathways
    # diseases
    if metabolite['diseases'] is not None:
        diseases = []
        for dis_dict in unlist(metabolite['diseases'].values()):
            diseases.append(dis_dict['name'])
        metabolite_data['diseases'] = diseases

    urine_metabolites_data[name] = metabolite_data


# save the dataset
with open('Data/urine_metabolome.json', 'w') as fp:
    json.dump(obj=urine_metabolites_data, fp=fp, indent=True, ensure_ascii=True)
