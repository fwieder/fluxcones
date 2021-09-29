# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:42:00 2021

@author: Frederik Wieder
"""


from flux_class_vecs import supp, flux_cone
import numpy as np
from sympy import Matrix
import tqdm
from multiprocessing import Pool
from collections import Counter

tol = 1e-14
digits = 14

""" Setup model """  
modelnames = ["Glycolysis/kegg1","Pyruvate/kegg62","Butanoate/kegg65", "Fructose and mannose/kegg51","Galactose/kegg52","Nitrogen/kegg910","Pentose and glucuronate/kegg4","PPP/kegg3","Propanoate/kegg64","Starch and sucrose/kegg5","Sulfur/kegg920","TCA/kegg2"]  
modelname = modelnames[5]

model = flux_cone.from_kegg("./Biomodels/kegg/" + modelname)
ext = np.genfromtxt("./Biomodels/kegg/" + modelname + "_externality")

ind = np.nonzero(ext)[0][0]
        
model.stoich = np.c_[model.stoich[:,:ind],model.stoich[:,np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind]]
    
model.rev = np.append(model.rev[:ind],model.rev[np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind],axis=0)
    
model.irr = np.append(model.irr[:ind],model.irr[np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind],axis=0)

irr = supp(model.irr)

gens = model.get_geometry()[0]

all_efvs = model.get_efvs("cdd")

gen_efms = [supp(gen) for gen in gens]

non_gen_inds = []
for i,efv in enumerate(all_efvs):
    if supp(efv) not in gen_efms:
        non_gen_inds.append(i)

efvs = all_efvs[non_gen_inds]

  
def two_gens(vector):
    efm = supp(vector)
    rev = supp(model.rev)
    rev_zeros = []
    for reaction_index , reaction_value  in enumerate(vector):
        if np.round(reaction_value,digits) == 0 and reaction_index in rev:
            rev_zeros.append(reaction_index)        
    candidate_inds = []
    
    for ind,efv_0 in enumerate(all_efvs):
        if set(irr_supp(efv_0)) <= set(irr_supp(vector)) and not supp(efv_0) == supp(vector):
            candidate_inds.append(ind)
    
    
    candidates = all_efvs[candidate_inds]
    
    two_gen_tuples = []

    for rev_zero_ind in rev_zeros:
        pos = candidates[np.where(candidates[:,rev_zero_ind] > tol)]
        neg = candidates[np.where(candidates[:,rev_zero_ind] < -tol)]
        
        if len(pos) > 0  and len(neg) > 0:
            for pos_efv in pos:
                for neg_efv in neg:
                    new_vec = pos_efv[rev_zero_ind]*neg_efv - neg_efv[rev_zero_ind]*pos_efv
                    if set(supp(new_vec)) == set(efm):
                        two_gen_tuples.append([pos_efv,neg_efv])                    
                    
    return two_gen_tuples

def irr_supp(vector):
    return list(np.intersect1d(supp(vector),irr))


if __name__ == "__main__":

    print(modelname)

    two_gen_pairs = tqdm.tqdm(map(two_gens,efvs), total= len(efvs))
    
        
    lens = [len(pairs) for pairs in two_gen_pairs]
    print(sorted(Counter(lens).items()))