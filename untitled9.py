# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 12:38:07 2021

@author: Frederik Wieder
"""

from flux_class_vecs import supp, flux_cone
import numpy as np
from sympy import Matrix
import tqdm
from multiprocessing import Pool
from collections import Counter
  

modelnames = ["Glycolysis/kegg1","Pyruvate/kegg62","Butanoate/kegg65", "Fructose and mannose/kegg51","Galactose/kegg52","Nitrogen/kegg910","Pentose and glucuronate/kegg4","PPP/kegg3","Propanoate/kegg64","Starch and sucrose/kegg5","Sulfur/kegg920","TCA/kegg2"]  
modelname = modelnames[0]

model = flux_cone.from_kegg("./Biomodels/kegg/" + modelname)
ext = np.genfromtxt("./Biomodels/kegg/" + modelname + "_externality")

ind = np.nonzero(ext)[0][0]
    
    
    
model.stoich = np.c_[model.stoich[:,:ind],model.stoich[:,np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind]]
    
model.rev = np.append(model.rev[:ind],model.rev[np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind],axis=0)
    
model.irr = np.append(model.irr[:ind],model.irr[np.unique(model.stoich[:,ind:],axis=1,return_index=True)[1]+ind],axis=0)



model = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")
model.delete_reaction(12)



irr = supp(model.irr)

nonegs = np.eye(len(model.stoich[0]))[irr]



def irr_supp(vector):
    return list(np.intersect1d(supp(vector),irr))

def irr_eqs(vector):
    return(list(set(supp(model.irr))-set(irr_supp(vector))))

def zero(vector):
    return(list(set(np.arange(len(vector)))-set(supp(vector))))

A = np.r_[model.stoich,-model.stoich]

S = np.r_[A,nonegs]


def check(vector):
    return len(vector) - np.linalg.matrix_rank(S[zero(np.dot(S,vector))])

if __name__ == "__main__":
    print(modelname)
    print("lin_dim:" , model.get_lin_dim())
    #model.efvs = np.load("./e_coli_no_bio_efvs.npy")
    #model.generators = np.load("./e_coli_no_bio_gens.npy")
    model.get_efvs("efmtool")
    model.get_geometry()
    model.get_efvs_in_mmbs()
    model.get_frev_efvs()
    mmb_efvs_total = sum([len(mmb_list) for mmb_list in model.mmb_efvs])
    
    print("Found", len(model.frev_efvs), "EFMs in the lineality space")
    print("Found", mmb_efvs_total, "EFMs in minimal proper faces")
    
    print("Dimensions of faces all efvs are in:")
    with Pool(12) as p:
        dims_efvs = list(tqdm.tqdm(p.imap(check,model.efvs),total= len(model.efvs)))
    print(sorted(Counter(dims_efvs).items()))
    
    print("Cone dim:", model.get_cone_dim())