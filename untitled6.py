# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 11:38:06 2021

@author: Frederik Wieder
"""

from flux_class_vecs import flux_cone,supp
import numpy as np

tol = 1e-10
digits = 10


S = np.array([[2,1,1,0,0,-1,0,0],[0,1,1,0,0,0,-1,0],[0,-1,1,0,0,0,0,-1],[-1,0,-1,1,0,0,0,0],[-1,-1,-1,0,1,0,0,0]])
rev = np.array([1,1,0,1,1,1,1,1])

model = flux_cone("test",S,rev)

irr = supp(model.irr)
def irr_supp(vector):
    return list(np.intersect1d(supp(vector),irr))


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

if __name__ == "__main__":
    model.get_efvs("cdd")
    model.get_geometry()
    print(model.efvs)
    
    for efv in model.efvs:
        print(two_gens(efv))
    
    
    