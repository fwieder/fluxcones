# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:10:16 2021

@author: Frederik Wieder
"""

import numpy as np
from flux_class_vecs import flux_cone, supp
from util import printProgressBar

tol = 1e-12
digits = 12

model = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")
#model.delete_reaction(12)

modelname = "Glycolysis/kegg1"

model = flux_cone.from_kegg("./Biomodels/kegg/" + modelname)
ext = np.genfromtxt("./Biomodels/kegg/" + modelname + "_externality")
ind = np.nonzero(ext)[0][0]
    

    
model.stoich = np.c_[model.stoich[:,:ind],model.stoich[:,np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind]]
    
model.rev = np.append(model.rev[:ind],model.rev[np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind],axis=0)
    
model.irr = np.append(model.irr[:ind],model.irr[np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind],axis=0)


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

    for i,efv in enumerate(model.efvs):
        printProgressBar(i, len(model.efvs), starttime=start)
        
        if two_gens(efv) == False:
            if not supp(efv) in gen_efms:
                print("found counterexample")
                print(i)
                break
        
    