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

modelname = modelnames[9]

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

def make_irredundant(model):
    redundants = "a"
    
    while len(redundants) > 0:
        if redundants != "a":
            model.make_rev(redundants[0])
        redundants = []
        for index in supp(model.irr):
            c = -np.eye(len(model.stoich.T))[index]
            A_ub = np.eye(len(model.stoich.T))[np.setdiff1d(supp(model.irr),index)]
            A_eq = model.stoich
            b_ub = np.zeros(len(A_ub))
            b_eq = np.zeros(len(A_eq))
            bounds = (None,None)
            if abs(linprog(c,A_ub,b_ub,A_eq,b_eq,bounds).fun) < .1:
                redundants.append(index)
                
            
if __name__ == "__main__":
    print(model.name)
    #model.make_irredundant()
    model.get_geometry()
    model.get_efvs("efmtool")
    
    with Pool(8) as p:
        model.dim_efvs = list(tqdm.tqdm(p.imap(model.check_dim,model.efvs), total = len(model.efvs)))
    irr_supp_lens = [len(model.irr_supp(efv)) for efv in model.efvs]
    dif  = np.array(model.dim_efvs) - np.array(irr_supp_lens)
    
    print(max(dif))
    print(min(dif))
    print(model.get_lin_dim())