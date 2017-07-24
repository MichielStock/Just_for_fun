"""
Created on Friday 21 July 2017
Last update: Mon 24 July 2017

@author: Michiel Stock
michielfmstock@gmail.com

Usines chemoinformatics to generate fingerprints and images
of the urine metabolites
"""

import json
from rdkit import Chem
import pandas as pd
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors

fingerprints = {}
descriptors = {}
metabolites = []

descriptors_to_compute = {
    'logP' : Descriptors.MolLogP,
    'Mol. Wt' : Descriptors.MolWt,
    'NHOH count' : Descriptors.NHOHCount,
    'Num. H acceptors' : Descriptors.NumHAcceptors,
    'Num. H donors' : Descriptors.NumHDonors,
    'Ring count' : Descriptors.RingCount,
}

descriptors = {k : [] for k in descriptors_to_compute.keys()}

with open('Data/urine_metabolome.json', 'r') as fp:
    urine_metabolites_data = json.load(fp)

for metabolite, metabolite_data in urine_metabolites_data.items():
    smiles = metabolite_data['smiles']
    # generate Morgan fingerprints
    try:
        molecule = Chem.MolFromSmiles(smiles)
        fp = MACCSkeys.GenMACCSKeys(molecule).ToBitString()
        fingerprints[metabolite] = list(map(int, fp))
        # draw molecule
        Draw.MolToFile(molecule,'Figures/Molecules/{}.svg'.format(metabolite))

        for descr, fun in descriptors_to_compute.items():
            value = fun(molecule)
            descriptors[descr].append(value)
        metabolites.append(metabolite)  # save names
    except:
        pass

# save the fingerprints
pd.DataFrame(fingerprints).T.to_csv('Data/metabolite_fingerprints.csv')
pd.DataFrame(descriptors, index=metabolites).to_csv('Data/metabolite_descriptors.csv')
