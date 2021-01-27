"""
Created on Sat Jan 16 14:53:05 2021

@author: fred
"""
import numpy as np
from functions import get_gens,get_efms,get_mmbs, sort_efms
import efmtool
from sbml_import import read_sbml, get_reversibility, get_stoichiometry
import time


'''
S = np.genfromtxt("/home/fred/Work/metabolic_networks/small_examples/illusnet_stoichiometry")
rev = np.genfromtxt("/home/fred/Work/metabolic_networks/small_examples/illusnet_reversibility").astype(int)

'''

model = read_sbml("C:/Users/User/Desktop/Biomodels/bioModels_MEPs/e_coli_core.xml")
S = get_stoichiometry(model)
rev = get_reversibility(model)



efms = get_efms(S,rev,"efmtool")
print("efms done",np.shape(efms))



'''
mmbs = get_mmbs(S,rev)
print("mmbs done",np.shape(mmbs))


sorting = sort_efms(efms,mmbs,rev)
print("sorting done")

print(len(sorting[0]), "frev_efms:" , sorting[0])
print(len(sorting[1]), "mpf_efms" , sorting[1])
print(len(sorting[2]), "int_efms" , sorting[2])


start_time = time.time()

efms_cdd = get_efms(S2,rev2,"cdd")
efms_efmtool = get_efms(S2,rev2,"efmtool")

comp_time = time.time() - start_time


print("EFMs from efmtool:")
print(np.shape(efms_efmtool))

print("EFMs from cdd:")
print(np.shape(efms_cdd))

print("Computation time:" , int(comp_time//60), "minutes and", round(comp_time%60,1) , "seconds.")
'''