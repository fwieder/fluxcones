"""
Created on Tue Jan 12 13:59:06 2021

@author: fred
"""
import cobra
import numpy as np
import time

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

def printProgressBar(iteration, total, starttime=0, decimals=1, length=50, fill='█', printEnd="\r"):
   
    if iteration > 0 :
        timeLeft = (total-iteration)*(time.perf_counter() - starttime)/iteration 
        suffix= "   Time left: %3dm %2ds" % (timeLeft//60, timeLeft%60)
    else:
        suffix = ""
        
    percent =  ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r|%s| %s%% %s' % (bar, percent,suffix), end = printEnd)
    if iteration == total:
        print()