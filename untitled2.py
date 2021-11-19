# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 13:31:27 2021

@author: Frederik Wieder
"""

from flux_class_vecs import flux_cone
import numpy as np
import sys

if __name__ == "__main__":
    model = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")
    model.delete_reaction(12)
    
    model = flux_cone.from_kegg("./Biomodels/kegg/Nitrogen/kegg910")
    
    model.get_efvs("efmtool")
    print(len(model.efvs), "efms found")
    
    model.get_geometry()
    rev_cancels = model.generators
    print(len(model.generators), "extreme rays")
    rev = np.nonzero(model.rev)[0]
    
    for efv in rev_cancels:
        if set(np.nonzero(efv)[0]) < set(rev):
            rev_cancels = np.r_[rev_cancels,-efv.reshape(1,len(efv))]
    
    
    
    old = rev_cancels
    for it in range(20):
        print("Iteration", it, ":", len(old), " efms found.")
        new = model.rev_cancels(old)
        
        if len(new) == len(model.efvs):
            print("all efms found with simple approach")
            break
        if len(new) == len(old):
            print("No more new efms found with simple approach")
            break
                            
        old = new