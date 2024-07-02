# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:12:55 2023

@author: fred
"""

import numpy as np

from flux_cones import flux_cone,supp
from itertools import product,combinations
from algorithms import MILP_shortest_decomp
import tqdm
from collections import Counter


num_metabs = 2
num_reacs = 6
value_list = [-1,0,1]
cols = [np.array(col) for col in product(value_list,repeat=num_metabs)]

#print(len(cols))


stoichs = [np.array(i).reshape(num_metabs,num_reacs) for i in combinations(cols,num_reacs)]
#print(len(stoichs))


revs = [np.array(i) for i in product([0,1],repeat = num_reacs) if len(supp(np.array(i)))>1]

#print(len(stoichs), "stoichiometric matrices")
#print(len(revs),"reversibility vectors")
#print("total exapmles:", len(stoichs)*len(revs))




data = np.zeros((len(stoichs),len(revs)))
data = []

model_ids = [(i,j) for i,j in product(range(len(stoichs)),range(len(revs)))]
print(len(model_ids), "models")
#models = [flux_cone(stoich,rev) for stoich,rev in product(stoichs,revs)]
deg_3_counter=0

def conjecture_check(model_id):
    model = flux_cone(stoichs[model_id[0]],revs[model_id[1]])
    model.get_efms("cdd")
    global deg_3_counter
    
    if len(model.efms) < 3:
        return (True,len(model.efms))
    else:
        for i,efm in enumerate(model.efms):
            if model.degree(efm) > 2:
                deg_3_counter +=1
                if len(model.two_gens(efm))==0:
                    coeffs = MILP_shortest_decomp(efm,np.delete(model.efms,i,axis=0))
                    if coeffs[0] != None:
                        if len(supp(coeffs)) > 2:
                            print(model.stoich,model.rev)
                            print(coeffs)
                            return (False,len(model.efms))
                            
                
                
        return (True,len(model.efms))



    
if __name__ == "__main__":
    
    data = []
    
    for model_id in tqdm.tqdm(model_ids):
        res = conjecture_check(model_id)
        if res[0] == True:
            data.append(res[1])
        if res[0] == False:
            print("Conjecture disproven!")
            break
    print(" ")
    print(deg_3_counter,"EFMs decomposed")
    print(np.count_nonzero(data), "models with EFMs")
    print(np.max(data), "largest number of EFMs")
    print(Counter(data))
    print("$(" + str(num_metabs) + "," + str(num_reacs) + ")$ & $" + str(set(value_list)) + "$ & " + str(np.count_nonzero(data)) + " & " + str(deg_3_counter) + " & " + str(np.max(data)) + "\\\\ ") 
