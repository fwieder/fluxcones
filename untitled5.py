# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 18:41:20 2021

@author: Frederik Wieder
"""

from flux_class_vecs import supp, flux_cone
import numpy as np
from sympy import Matrix
import tqdm
from multiprocessing import Pool
from collections import Counter
from itertools import combinations  

modelnames = ["Glycolysis/kegg1","Pyruvate/kegg62","Butanoate/kegg65", "Fructose and mannose/kegg51","Galactose/kegg52","Nitrogen/kegg910","Pentose and glucuronate/kegg4","PPP/kegg3","Propanoate/kegg64","Starch and sucrose/kegg5","Sulfur/kegg920","TCA/kegg2"]  
modelname = modelnames[6]


model = flux_cone.from_kegg("./Biomodels/kegg/" + modelname)
ext = np.genfromtxt("./Biomodels/kegg/" + modelname + "_externality")

ind = np.nonzero(ext)[0][0]
    
    
    
model.stoich = np.c_[model.stoich[:,:ind],model.stoich[:,np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind]]
    
model.rev = np.append(model.rev[:ind],model.rev[np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind],axis=0)
    
model.irr = np.append(model.irr[:ind],model.irr[np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind],axis=0)

model =flux_cone.from_sbml("./Biomodels/bigg/iCN718.xml")
print("iCN718.xml")
print(model.get_lin_dim())

if __name__ == "__main__":
    model.get_frev_efvs()
    frev_efms = list(set([tuple(supp(efv)) for efv in model.frev_efvs]))
    frev_efms = [list(efm) for efm in frev_efms]
    print(modelname)
    print(len(frev_efms))
        
    def circuit_axiom(frev_efm1,frev_efm2):
        if len(np.intersect1d(frev_efm1,frev_efm2)) == 0:
            return True
        true_for = []
        for reaction in np.intersect1d(frev_efm1,frev_efm2):
            for efm in frev_efms:
                if set(efm) < set(np.setdiff1d(np.union1d(frev_efm1,frev_efm2),reaction)):
                    true_for.append(reaction)
                    break
        if all(true_for == np.intersect1d(frev_efm1,frev_efm2)):
            return True
        return False
    
    for comb in combinations(frev_efms,2):
        if not circuit_axiom(comb[0], comb[1]):
            print("False")
    print("Circuit Axiom holds for", modelname, "with", len(frev_efms), "reversible efms")
    