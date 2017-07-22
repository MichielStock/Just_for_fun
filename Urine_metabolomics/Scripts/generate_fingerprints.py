"""
Created on Friday 21 July 2017
Last update: -

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

fingerprints = {}

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
    except:
        pass

# save the fingerprints
pd.DataFrame(fingerprints).T.to_csv('Data/metabolite_fingerprints.csv')
