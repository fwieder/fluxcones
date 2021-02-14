"""
Created on Tue Jan 12 13:59:06 2021

@author: fred
"""



import cobra
import numpy as np




'''
Create model from sbml file. input is the path to the sbml file
'''
def read_sbml(path_to_file):
    model = cobra.io.read_sbml_model(path_to_file)
    return model




'''
returns the stoichiometric Matrix of the model as np.array  
'''
def get_stoichiometry(model):
    S = cobra.util.array.create_stoichiometric_matrix(model)
    return S


'''
returns a {0,1}-vector with a 1 indicating reversible reactions from the model and 0 indicating irreversible reactions
'''
def get_reversibility(model):
    return np.array([rea.reversibility for rea in model.reactions]).astype(int)


def gen_model(path_to_file):
    sbml_model = cobra.io.read_sbml_model(path_to_file)
    class sbml:
        stoich = cobra.util.array.create_stoichiometric_matrix(sbml_model)
        rev = np.array([rea.reversibility for rea in sbml_model.reactions]).astype(int)
        name = sbml_model.name
    return sbml

class flux_model:
    def __init__(self, path_to_file):
        sbml_model = cobra.io.read_sbml_model(path_to_file)
        self.stoich = cobra.util.array.create_stoichiometric_matrix(sbml_model)
        self.rev = np.array([rea.reversibility for rea in sbml_model.reactions]).astype(int)
        self.name = sbml_model.name