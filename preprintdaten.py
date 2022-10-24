# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 08:57:36 2022

@author: Frederik Wieder
"""


from flux_class_vecs import supp, flux_cone
import numpy as np
from collections import Counter
import sys,time
from util import printProgressBar
from multiprocessing import Pool

modelnames = ["Pyruvate/kegg62","PPP/kegg3"]
#modelnames = ["Glycolysis/kegg1","Pyruvate/kegg62","Butanoate/kegg65", "Fructose and mannose/kegg51","Galactose/kegg52","Nitrogen/kegg910","Pentose and glucuronate/kegg4","PPP/kegg3","Propanoate/kegg64","Starch and sucrose/kegg5","Sulfur/kegg920","TCA/kegg2"]
models1 = []

for modelname in modelnames:
    
    model = flux_cone.from_kegg("./Biomodels/kegg/" + modelname)
    model.name = modelname
    ext = np.genfromtxt("./Biomodels/kegg/" + modelname + "_externality")
    
    ind = np.nonzero(ext)[0][0]
        
        
        
    model.stoich = np.c_[model.stoich[:,:ind],model.stoich[:,np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind]]
        
    model.rev = np.append(model.rev[:ind],model.rev[np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind],axis=0)
        
    model.irr = np.append(model.irr[:ind],model.irr[np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind],axis=0)
    
    
    
    #model = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")
    #model.delete_reaction(12)
    

    irr = supp(model.irr)
    model.nonegs = np.eye(len(model.stoich[0]))[irr]
    model.S = np.r_[model.stoich,model.nonegs]
    
    models1.append(model)


coli = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")
coli.delete_reaction(12)
models1.append(coli)

models = [models1[2],models1[1],models1[0]]

model = models[0]
print(model.name)

if __name__ == "__main__":
    model.get_geometry()
    model.get_efvs()
    
    print("rank efvs", np.linalg.matrix_rank(model.efvs))
    print("rank gens", np.linalg.matrix_rank(model.generators))

"""
if __name__ == "__main__": 
    for model in models:
        
        
        
        
        model.get_geometry()
        model.get_efvs("efmtool")
        model.get_mmbs()
        print(model.name)
        print("m,n", np.shape(model.stoich))
        print("irr:", len(supp(model.irr)))
        print("rev:", len(supp(model.rev)))
        print("rank(S)", np.linalg.matrix_rank(model.stoich))
        print("rank(S_irr):",np.linalg.matrix_rank(model.stoich[:,supp(model.irr)]))
        print("dim(L)",model.get_lin_dim())
        
        q = max([len(model.irr_supp(efv)) for efv in model.efvs])
        t = model.get_lin_dim()
        r = np.linalg.matrix_rank(model.stoich[:,supp(model.irr)]) + 1
        
        print("q",q)
        print("r",r)
        
        degs = [model.check_dim(efv) for efv in model.efvs]
        maxdeg = max(degs)
        print("Maxdeg", maxdeg)
        print("t+q", t+q)
        print("t+r", t+r)
        print("dim(C)",model.get_cone_dim())
        print("|EFMs|", len(model.efvs))
        print("")
        print("")
        
        print("n-rank(S)", np.shape(model.stoich)[1] - np.linalg.matrix_rank(model.stoich))
        print("dim(C)",model.get_cone_dim())
        print("dim(L)",model.get_lin_dim())        
        blocked_rev = model.blocked_rev_reactions()
        blocked_irr = model.blocked_irr_reactions()
        model.make_irredundant()
        print("|Facets|", len(supp(model.irr)))
        print("|blocked irr|", len(blocked_irr))
        print("|blocked rev|", len(blocked_rev))
        print("|EFMs|", len(model.efvs))
        print("|MMBs|", len(model.mmbs))
        print("")
        print("")
        
        print("rank(S_irr):",np.linalg.matrix_rank(model.stoich[:,supp(model.irr)]))
        print("lin.dim:", model.get_lin_dim())
        model.get_efvs("efmtool")
        q = max([len(model.irr_supp(efv)) for efv in model.efvs])
        t = model.get_lin_dim()
        r = np.linalg.matrix_rank(model.stoich[:,supp(model.irr)]) + 1
        degs = [model.check_dim(efv) for efv in model.efvs]
        maxdeg = max(degs)
        print("q",q)
        
        print("r",r)
        
        
        print("Maxdeg", maxdeg)
        print("t+q", t+q)
        print("t+r", t+r)
        """