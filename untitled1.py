#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:18:33 2021

@author: fred
"""

from flux_class_vecs import flux_cone
from collections import Counter
import numpy as np
import sys

modelnames = ["Butanoate/kegg65", "Fructose and mannose/kegg51","Galactose/kegg52","Nitrogen/kegg910","Pentose and glucuronate/kegg4","PPP/kegg3","Propanoate/kegg64","Starch and sucrose/kegg5","Sulfur/kegg920","TCA/kegg2"]



for modelname in [modelnames[0]]:
    
    model = flux_cone.from_kegg("./Biomodels/kegg/" + modelname)
    
    ext = np.genfromtxt("./Biomodels/kegg/" + modelname + "_externality")

    ind = np.nonzero(ext)[0][0]
    
    
    
    model.stoich = np.c_[model.stoich[:,:ind],model.stoich[:,np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind]]
    
    model.rev = np.append(model.rev[:ind],model.rev[np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind],axis=0)
    
    model.irr = np.append(model.irr[:ind],model.irr[np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind],axis=0)
   
    
    while(model.get_lin_dim() > 0):
        model.get_frev_efvs()
        frev_efms = [list(np.nonzero(efv)[0]) for efv in model.frev_efvs]
        reaction_counter = Counter([reac for efm in frev_efms for reac in efm])
        tosplit = reaction_counter.most_common(1)[0][0]
        model.split_rev(tosplit)
    
    model = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")
    model.delete_reaction(12)
    
    
    if __name__ == "__main__":
        print("")
        print(modelname)
        print("lin dim" , model.get_lin_dim())
        print(len(model.get_efvs("efmtool")), "found with efmtool")
        #print(len(model.get_efvs("cdd")), "found with cdd")
        print(len(model.efvs), "total efvs (" , int(len(model.get_frev_efvs())/2) ,"frev efms counted twice)")
        efms = set([tuple(np.nonzero(efv)[0]) for efv in model.efvs])
        print(len(efms), "total efms")
        
        model.get_geometry()
    
       
        rev_cancels = model.generators
        
        rev = np.nonzero(model.rev)[0]        
        
        for efv in rev_cancels:
            if set(np.nonzero(efv)[0]) < set(rev):
                rev_cancels = np.r_[rev_cancels,-efv.reshape(1,len(efv))]
        
        
        old = rev_cancels
        for it in range(20):
            new = model.rev_cancels(old)
            print("Iteration", it, ":", len(new), " efms found.")
            if len(new) == len(efms) + model.lin_dim or len(new) == len(old):
                print("No more new efms found")
                break
                            
            old = new
        
        print("")
        print("")
        
        for it in range(20):
            print("Iteration" , it , ":", len(set([tuple(np.nonzero(efv)[0]) for efv in rev_cancels])) ,"efms")
            new_rev_cancels = model.rev_cancellations(rev_cancels)
            if len(set([tuple(np.nonzero(efv)[0]) for efv in new_rev_cancels])) == len(efms) or len(set([tuple(np.nonzero(efv)[0]) for efv in new_rev_cancels])) == len(set([tuple(np.nonzero(efv)[0]) for efv in rev_cancels])):
                print(len(set([tuple(np.nonzero(efv)[0]) for efv in new_rev_cancels])) , "efms found in Iteration" , it+1)
                break
            rev_cancels = new_rev_cancels
        
        print("")
        print("")
        print("")
        print("")