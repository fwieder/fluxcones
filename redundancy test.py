# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:04:31 2021

@author: Frederik Wieder
"""

from flux_class_vecs import flux_cone, supp
from scipy.optimize import linprog
import numpy as np
from multiprocessing import Pool
import tqdm

modelnames = ["Glycolysis/kegg1","Pyruvate/kegg62","Butanoate/kegg65", "Fructose and mannose/kegg51","Galactose/kegg52","Nitrogen/kegg910","Pentose and glucuronate/kegg4","PPP/kegg3","Propanoate/kegg64","Starch and sucrose/kegg5","Sulfur/kegg920","TCA/kegg2"]  
models = []
for modelname in modelnames:
        

    model = flux_cone.from_kegg("./Biomodels/kegg/"+modelname)
    model.name = modelname
    ext = np.genfromtxt("./Biomodels/kegg/" + modelname + "_externality")

    ind = np.nonzero(ext)[0][0]
    
    
    
    model.stoich = np.c_[model.stoich[:,:ind],model.stoich[:,np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind]]
    
    model.rev = np.append(model.rev[:ind],model.rev[np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind],axis=0)
    
    model.irr = np.append(model.irr[:ind],model.irr[np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind],axis=0)
    irr = supp(model.irr)
    model.nonegs = np.eye(len(model.stoich[0]))[irr]
    model.S = np.r_[model.stoich,model.nonegs]
    model.make_irredundant()
    models.append(model)
    
if __name__ == "__main__":
    for model in models:
        print(model.name)
        print(np.linalg.matrix_rank(model.stoich))
        print(np.shape(model.stoich))