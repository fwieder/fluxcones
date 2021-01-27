"""
Created on Tue Jan 12 13:59:06 2021

@author: fred
"""
import cobra
import numpy as np

def read_sbml(file):
    model = cobra.io.read_sbml_model(file)
    model.reactions.sort()
    model.metabolites.sort()
    return model

def get_stoichiometry(model):
    S = cobra.util.array.create_stoichiometric_matrix(model)
    return S

def get_reversibility(model):
    return np.array([rea.reversibility for rea in model.reactions]).astype(int)
