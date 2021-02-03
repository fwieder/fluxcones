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
    model.reactions.sort()
    model.metabolites.sort()
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

