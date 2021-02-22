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

from functions import get_efvs,get_mmbs,filter_efms,flux_model,efms_in_mmb, is_efm
from util import printProgressBar,model_paths


models = [flux_model("./Biomodels/bigg_models/" + model_paths[i] + ".xml") for i in range(len(model_paths))]



'''
model.name = "Sulfur"
model.stoich = np.genfromtxt("./Biomodels/models_halim/Sulfur/kegg920_stoichiometry")
model.rev = np.genfromtxt("./Biomodels/models_halim/Sulfur/kegg920_reversibility").astype(int)
model.irr = (np.ones(len(model.rev)) - model.rev).astype(int)
'''

for model in models:
    
    mmb_start_time = time.time()
    mmbs = get_mmbs(model.stoich,model.rev)
    mmb_comp_time = time.time() - mmb_start_time
    print(len(get_mmbs(model.stoich,model.rev)))
sys.exit()
print(model.name)
print("Shape of stoichiometric matrix: ", np.shape(model.stoich))
print("Number of reversible reactions: ", np.count_nonzero(model.rev))
print("Dimension of the reversible metabolic space: ", model.lin_dim)


#delete Biomass-reaction (12) of e_coli

'''
model.stoich = np.delete(model.stoich,12,1)
model.rev = np.delete(model.rev, 12,0)
model.irr = np.delete(model.irr, 12,0)
'''

'''
efv_start_time = time.time()
efvs = get_efvs(model.stoich,model.rev,"efmtool")
efv_comp_time =  time.time() - efv_start_time

print(np.shape(efvs)[0], "EFMs calculated in %3dm %2ds" % (efv_comp_time//60,efv_comp_time%60))
'''

mmb_start_time = time.time()
mmbs = get_mmbs(model.stoich,model.rev)
mmb_comp_time = time.time() - mmb_start_time

print(len(mmbs), "MMBs calculated in %3dm %2ds" % (mmb_comp_time//60,mmb_comp_time%60))
'''
print("initiating EFM filtering")
filter_start_time = time.time()   

mmb_efms, int_efms, frev_efms = filter_efms(efvs,mmbs,model.rev)

filter_comp_time = time.time()-filter_start_time

print("")

print("EFMs filtered in %3dm %2ds" % (filter_comp_time//60,filter_comp_time%60))
'''

mmb_efms_alt = []

print("Finding EFMs in MMBs")
finding_start_time = time.time()
start = time.perf_counter()
for ind,mmb in enumerate(mmbs):
    mmb_efms_alt.append(efms_in_mmb(mmb,model))
    printProgressBar(ind,len(mmbs),starttime = start)
finding_time = time.time() - finding_start_time
print("")
print("EFMs in MMBs found in %3dm %2ds" %(finding_time//60,finding_time%60))

    
