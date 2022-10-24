# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:38:07 2021
@author: Frederik Wieder
"""

from flux_class_vecs import supp, flux_cone
import numpy as np

import warnings

warnings.filterwarnings('ignore')

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
    model.make_irredundant()
    models.append(model)
    


if __name__ == "__main__":
    for model in models:
        print(model.name)
        model.get_mmbs()
        mmb_lens = [len(mmb) for mmb in model.mmbs]
        gamma = max(mmb_lens)
        print("gamma = ", gamma)
        model.get_efvs()
        dims_irr = [(model.check_dim(efv),len(model.irr_supp(efv))) for efv in model.efvs]
        t = model.get_lin_dim()
        print("testing conjecture")
        for deg_irr in dims_irr:
            if deg_irr[0]-t-1+gamma<deg_irr[1]:
                print("False: ", deg_irr)
                break
        print("conjecture true for ", model.name)
    