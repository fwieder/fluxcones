"""
Created on Sat Jan 16 14:53:05 2021

@author: fred
"""
import numpy as np
import time

from functions import get_efvs,get_mmbs,sort_efms
from sbml_import import read_sbml, get_reversibility, get_stoichiometry




S = np.genfromtxt("C:/Users/User/Desktop/Biomodels/small_examples/illusnet_stoichiometry")
rev = np.genfromtxt("C:/Users/User/Desktop/Biomodels/small_examples/illusnet_reversibility").astype(int)



model = read_sbml("C:/Users/User/Desktop/Biomodels/bioModels_MEPs/e_coli_core.xml")
S = get_stoichiometry(model)
rev = get_reversibility(model)


efv_start_time = time.time()
efvs = get_efvs(S,rev,"efmtool")
efv_comp_time =  time.time() - efv_start_time

print(np.shape(efvs)[0], "EFMs calculated in", np.round(efv_comp_time//60,2), "min" , np.round(efv_comp_time%3600,2),"s")
 
mmb_start_time = time.time()
mmbs = get_mmbs(S,rev)
mmb_comp_time = time.time() - mmb_start_time

print(len(mmbs), "MMBs calculated in", np.round(mmb_comp_time//60,2), "min" , np.round(mmb_comp_time%3600,2),"s")

print("initiating EFM filtering")
        
mmb_efms, int_efms, frev_efms = sort_efms(efvs,mmbs,rev)
