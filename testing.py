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

from functions import get_efvs,get_mmbs,filter_efms,flux_model
from sbml_import import read_sbml, get_reversibility, get_stoichiometry




S = np.genfromtxt("./Biomodels/small_examples//illusnet/illusnet_stoichiometry")
rev = np.genfromtxt("./Biomodels/small_examples/illusnet/illusnet_reversibility").astype(int)


model = flux_model("./Biomodels/bioModels_MEPs/e_coli_core.xml")

print(model.name)
print("Shape of stoichiometric matrix: ", np.shape(model.stoich))
print("Number of reversible reactions: ", np.count_nonzero(model.rev))



efv_start_time = time.time()
efvs = get_efvs(model.stoich,model.rev,"efmtool")
efv_comp_time =  time.time() - efv_start_time

print(np.shape(efvs)[0], "EFMs calculated in %3dm %2ds" % (efv_comp_time//60,efv_comp_time%60))


mmb_start_time = time.time()
mmbs = get_mmbs(model.stoich,model.rev)
mmb_comp_time = time.time() - mmb_start_time

print(len(mmbs), "MMBs calculated in%3dm %2ds" % (mmb_comp_time//60,mmb_comp_time%60))

print("initiating EFM filtering")
        
mmb_efms, int_efms, frev_efms = filter_efms(efvs,mmbs,model.rev)
