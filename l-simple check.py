# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:38:07 2021

@author: Frederik Wieder
"""

from flux_class_vecs import supp, flux_cone, zero
import numpy as np
from sympy import Matrix
import tqdm
from multiprocessing import Pool
from collections import Counter
import matplotlib as mpl
import matplotlib.pyplot as plt  
from matplotlib.ticker import MaxNLocator
modelnames = ["Glycolysis/kegg1","Pyruvate/kegg62","Butanoate/kegg65", "Fructose and mannose/kegg51","Galactose/kegg52","Nitrogen/kegg910","Pentose and glucuronate/kegg4","PPP/kegg3","Propanoate/kegg64","Starch and sucrose/kegg5","Sulfur/kegg920","TCA/kegg2"]  

models = []

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
    #model.make_irredundant()
    models.append(model)






if __name__ == "__main__":
    models = models[2:]
    for model in models:
        print(model.name)
        model.get_geometry()
        cone_dim = model.get_cone_dim()
        efvs = model.get_efvs("efmtool")
       
        model.efv_dims = [model.check_dim(efv) for efv in efvs]
        k = max(model.efv_dims)
        irr_len = len(supp(model.irr))
        d_irr = np.linalg.matrix_rank(model.stoich[:,supp(model.irr)])
        m,n = np.shape(model.stoich)
        l_cands = [len(zero(efv[supp(model.irr)])) - model.check_dim(efv) + cone_dim for efv in efvs]
        l = max(l_cands)
      
        print("n = ",n)
        print("m = ",m)
        print("d_irr = ",d_irr)
        print("irr_len = ", irr_len)
        print("l = ", l)
        print("k = ",k)
        print("old_bound = ", model.get_lin_dim() + d_irr + 1)
        print("add_bound = ", n-m+l-irr_len+d_irr-1)
        print("")
        print("cone_dim = ", cone_dim )
        