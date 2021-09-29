# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 13:43:52 2021

@author: Frederik Wieder
"""


import numpy as np
from flux_class_vecs import flux_cone, supp
from util import printProgressBar

tol = 1e-14
digits = 14

model = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")


irr = supp(model.irr)

def irr_supp(vector):
    return list(np.intersect1d(supp(vector),irr))


   
if __name__ == "__main__":
    model.get_efvs("efmtool")
    model.get_geometry()
   
    def two_gens(vector):
        efm = supp(vector)
        rev = supp(model.rev)
        rev_zeros = []
        for reaction_index , reaction_value  in enumerate(vector):
            if np.round(reaction_value,digits) == 0 and reaction_index in rev:
                rev_zeros.append(reaction_index)        
        candidate_inds = []
        
        for ind,efv_0 in enumerate(model.efvs):
            if set(irr_supp(efv_0)) <= set(irr_supp(vector)):
                candidate_inds.append(ind)
        
        
        candidates = model.efvs[candidate_inds]
       
        for rev_zero_ind in rev_zeros:
            pos = candidates[np.where(candidates[:,rev_zero_ind] > tol)]
            neg = candidates[np.where(candidates[:,rev_zero_ind] < -tol)]
            
            if len(pos) > 0  and len(neg) > 0:
                for pos_efv in pos:
                    for neg_efv in neg:
                        new_vec = pos_efv[rev_zero_ind]*np.round(neg_efv,digits) - neg_efv[rev_zero_ind]*np.round(pos_efv,digits)
                        if set(supp(new_vec)) == set(efm):
                                
                            return (pos_efv,neg_efv)                    
    
        return False


    
   
    gen_efms = [supp(efv) for efv in model.generators]
    efms = [supp(efv) for efv in model.efvs]
    print(len(efms), "EFVs")
    import time
    start = time.perf_counter()
    f = open("./test.txt", "r")
    already_checked = int(f.read())
    f.close()
    for i in range(already_checked, len(model.efvs)):
        efv = model.efvs[i]
        printProgressBar(i, len(model.efvs)-already_checked, starttime=start)
        if i % 100 == 0:
            print("backup after" , i , "efms checked")
            f = open("./test.txt", "w")
            f.write(str(i))
            f.close()
        if two_gens(efv) == False:
            if not supp(efv) in gen_efms:
                print("found counterexample")
                print(i)
                break
        