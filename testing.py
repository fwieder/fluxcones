"""
Created on Sat Jan 16 14:53:05 2021

@author: fred
"""

'''
This is just a test-file for testing the functions with small models of which the results are known

'''




import numpy as np
import time
import sys

from functions import *
from util import printProgressBar,model_paths

'''
models = [flux_model("./Biomodels/bigg_models/" + model_paths[i] + ".xml") for i in range(len(model_paths))]

for i,model in enumerate(models):
    print(model_paths[i])

    print("Shape of stoichiometric matrix: ", np.shape(model.stoich))
    print("Number of reversible reactions: ", np.count_nonzero(model.rev))
    print("Dimension of the reversible metabolic space: ", model.lin_dim)

sys.exit()
'''





model = flux_model("./Biomodels/bigg/" + model_paths[0] + ".xml")

model.name = "Sulfur"
model.stoich = np.genfromtxt("./Biomodels/kegg/Sulfur/kegg920_stoichiometry")
model.rev = list(np.genfromtxt("./Biomodels/kegg/Sulfur/kegg920_reversibility").astype(int))
model.irr = (np.ones(len(model.rev)) - model.rev).astype(int)


efvs = get_efvs(model,"efmtool")
print(len(efvs))
sys.exit()

print(model.name)
print("Shape of stoichiometric matrix: ", np.shape(model.stoich))
print("Number of reversible reactions: ", np.count_nonzero(model.rev))
print("Dimension of the reversible metabolic space: ", model.lin_dim)



print("Finding EFMs in MMBs")

finding_start_time = time.time()
mmb_efms = get_efms_in_mmbs(model)
finding_time = time.time() - finding_start_time

print("")
print("EFMs in MMBs found in %3dm %2ds" %(finding_time//60,finding_time%60))

    
