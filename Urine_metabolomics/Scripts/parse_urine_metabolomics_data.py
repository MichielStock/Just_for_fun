"""
Created on Thursday 20 July 2017
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Downloads and parses the urine metabolomics dataset from the Human Metabolomics
Database (http://www.hmdb.ca/).
"""

import xmltodict

# load the metabolites dataset as a dictonary
with open('Data/urine_metabolites.xml') as fh:
    urine_metabolites_db = xmltodict.parse(fh.read())['hmdb']

# clean the dataset in a nice format
urine_metabolites_data = {}

# simple propeties to collect:
simple_fields = set([
    'description',
    'chemical_formula',
    'smiles',
    'average_molecular_weight'])

for metabolite in urine_metabolites_db['metabolite']:
    propeties = {}
    name = metabolite['name']
    urine_metabolites_data[name] = propeties
    for k, v in metabolite.items():
        if k in simple_fields:
            propeties[k] = v
    if 'taxonomy' in metabolite:
        for k in ['kingdom', 'class', 'sub_class']
        urine_metabolites_data[k] = metabolite[k]
