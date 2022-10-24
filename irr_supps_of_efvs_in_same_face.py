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
import matplotlib as mpl
import matplotlib.pyplot as plt  
from matplotlib.ticker import MaxNLocator
import sys

modelnames = ["Pyruvate/kegg62","PPP/kegg3"]

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
    
    models.append(model)

def irr_supp(vector):
    return list(np.intersect1d(supp(vector),irr))

def irr_eqs(vector):
    return(list(set(supp(model.irr))-set(irr_supp(vector))))

def zero(vector):
    return(list(set(np.arange(len(vector)))-set(supp(vector))))


coli = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")
coli.delete_reaction(12)
models.append(coli)

if __name__ == "__main__":
    model = models[0]
    model.make_irredundant()
    
    model.get_geometry()
    model.get_efvs("efmtool")
    model.get_mmbs()
    mmb_lens = [len(mmb) for mmb in model.mmbs]
    
    
    dim16_efvs = []
    dim17_efvs = []
    dim18_efvs = []
    dim19_efvs = []
    dim20_efvs = []
    dim21_efvs = []
    dim22_efvs = []
    dim23_efvs = []
    dim24_efvs = []
    
    for efv in model.efvs:
        dim = model.check_dim(efv)
        
        if dim == 16:
            dim16_efvs.append(efv)
        if dim == 17:
            dim17_efvs.append(efv)
        if dim == 18:
            dim18_efvs.append(efv)
        if dim == 19:
            dim19_efvs.append(efv)
        if dim == 20:
            dim20_efvs.append(efv)
        if dim == 21:
            dim21_efvs.append(efv)
        if dim == 22:
            dim22_efvs.append(efv)
        if dim == 23:
            dim23_efvs.append(efv)
        if dim == 24:
            dim24_efvs.append(efv)
    
    
    dim16_irr_supps = [tuple(model.irr_supp(efv)) for efv in dim16_efvs]
    dim17_irr_supps = [tuple(model.irr_supp(efv)) for efv in dim17_efvs]
    dim18_irr_supps = [tuple(model.irr_supp(efv)) for efv in dim18_efvs]
    dim19_irr_supps = [tuple(model.irr_supp(efv)) for efv in dim19_efvs]
    dim20_irr_supps = [tuple(model.irr_supp(efv)) for efv in dim20_efvs]
    dim21_irr_supps = [tuple(model.irr_supp(efv)) for efv in dim21_efvs]
    dim22_irr_supps = [tuple(model.irr_supp(efv)) for efv in dim22_efvs]
    dim23_irr_supps = [tuple(model.irr_supp(efv)) for efv in dim23_efvs]
    dim24_irr_supps = [tuple(model.irr_supp(efv)) for efv in dim24_efvs]
    
    counter16 = Counter(e for e in dim16_irr_supps)
    counter17 = Counter(e for e in dim17_irr_supps)
    counter18 = Counter(e for e in dim18_irr_supps)
    counter19 = Counter(e for e in dim19_irr_supps)
    counter20 = Counter(e for e in dim20_irr_supps)
    counter21 = Counter(e for e in dim21_irr_supps)
    counter22 = Counter(e for e in dim22_irr_supps)
    counter23 = Counter(e for e in dim23_irr_supps)
    counter24 = Counter(e for e in dim24_irr_supps)
    
    print(model.name)
    print("[(<efms in a face>,<number of faces with that many efms>)] , <number of faces of dimension k that contain EFMs>, <total number of efms in faces of dimension k>")
    print("16-faces")
    print(sorted(Counter(counter16.values()).items()),len(counter16),len(dim16_efvs))
    print("17-faces")
    print(sorted(Counter(counter17.values()).items()),len(counter17),len(dim17_efvs))
    print("18-faces")
    print(sorted(Counter(counter18.values()).items()),len(counter18),len(dim18_efvs))
    print("19-faces")
    print(sorted(Counter(counter19.values()).items()),len(counter19),len(dim19_efvs))
    print("20-faces")
    print(sorted(Counter(counter20.values()).items()),len(counter20),len(dim20_efvs))
    print("21-faces")
    print(sorted(Counter(counter21.values()).items()),len(counter21),len(dim21_efvs))
    print("22-faces")
    print(sorted(Counter(counter22.values()).items()),len(counter22),len(dim22_efvs))
    print("23-faces")
    print(sorted(Counter(counter23.values()).items()),len(counter23),len(dim23_efvs))
    print("24-faces")
    print(sorted(Counter(counter24.values()).items()),len(counter24),len(dim24_efvs))
    
    
    
    dim16_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim16_efvs]
    dim17_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim17_efvs]
    dim18_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim18_efvs]
    dim19_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim19_efvs]
    dim20_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim20_efvs]
    dim21_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim21_efvs]
    dim22_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim22_efvs]
    dim23_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim23_efvs]
    dim24_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim24_efvs]
    
    
    print(model.name)
    print("Cardinalities of irr_supps for:")
    print("16-face efms:", sorted(Counter(dim16_efv_irr_supp_lens).items()))
    print("17-face efms:", sorted(Counter(dim17_efv_irr_supp_lens).items()))
    print("18-face efms:", sorted(Counter(dim18_efv_irr_supp_lens).items()))
    print("19-face efms:", sorted(Counter(dim19_efv_irr_supp_lens).items()))
    print("20-face efms:", sorted(Counter(dim20_efv_irr_supp_lens).items()))
    print("21-face efms:", sorted(Counter(dim21_efv_irr_supp_lens).items()))
    print("22-face efms:", sorted(Counter(dim22_efv_irr_supp_lens).items()))
    print("23-face efms:", sorted(Counter(dim23_efv_irr_supp_lens).items()))
    print("24-face efms:", sorted(Counter(dim24_efv_irr_supp_lens).items()))

    print("")
    print("MMB sizes:", sorted(Counter(mmb_lens).items()))
    
    
    
    model = models[1]
    model.make_irredundant()
    
    model.get_geometry()
    model.get_efvs("efmtool")
    model.get_mmbs()
    mmb_lens = [len(mmb) for mmb in model.mmbs]
    
    dim8_efvs = []
    dim9_efvs = []
    dim10_efvs = []
    dim11_efvs = []
    dim12_efvs = []
    dim13_efvs = []
    dim14_efvs = []
    
    for efv in model.efvs:
        dim = model.check_dim(efv)
        
        if dim == 8:
            dim8_efvs.append(efv)
        if dim == 9:
            dim9_efvs.append(efv)
        if dim == 10:
            dim10_efvs.append(efv)
        if dim == 11:
            dim11_efvs.append(efv)
        if dim == 12:
            dim12_efvs.append(efv)
        if dim == 13:
            dim13_efvs.append(efv)
        if dim == 14:
            dim14_efvs.append(efv)
    
    
    
    dim8_irr_supps = [tuple(model.irr_supp(efv)) for efv in dim8_efvs]
    dim9_irr_supps = [tuple(model.irr_supp(efv)) for efv in dim9_efvs]
    dim10_irr_supps = [tuple(model.irr_supp(efv)) for efv in dim10_efvs]
    dim11_irr_supps = [tuple(model.irr_supp(efv)) for efv in dim11_efvs]
    dim12_irr_supps = [tuple(model.irr_supp(efv)) for efv in dim12_efvs]
    dim13_irr_supps = [tuple(model.irr_supp(efv)) for efv in dim13_efvs]
    dim14_irr_supps = [tuple(model.irr_supp(efv)) for efv in dim14_efvs]
    
    counter8 = Counter(e for e in dim8_irr_supps)
    counter9 = Counter(e for e in dim9_irr_supps)
    counter10 = Counter(e for e in dim10_irr_supps)
    counter11 = Counter(e for e in dim11_irr_supps)
    counter12 = Counter(e for e in dim12_irr_supps)
    counter13 = Counter(e for e in dim13_irr_supps)
    counter14 = Counter(e for e in dim14_irr_supps)
    
    print(model.name)
    print("[(<efms in a face>,<number of faces with that many efms>)] , <number of faces of dimension k that contain EFMs>, <total number of efms in faces of dimension k>")
    print("8-faces")
    print(sorted(Counter(counter8.values()).items()),len(counter8),len(dim8_efvs))
    print("9-faces")
    print(sorted(Counter(counter9.values()).items()),len(counter9),len(dim9_efvs))
    print("10-faces")
    print(sorted(Counter(counter10.values()).items()),len(counter10),len(dim10_efvs))
    print("11-faces")
    print(sorted(Counter(counter11.values()).items()),len(counter11),len(dim11_efvs))
    print("12-faces")
    print(sorted(Counter(counter12.values()).items()),len(counter12),len(dim12_efvs))
    print("13-faces")
    print(sorted(Counter(counter13.values()).items()),len(counter13),len(dim13_efvs))
    print("14-faces")
    print(sorted(Counter(counter14.values()).items()),len(counter14),len(dim14_efvs))
    
    
    
    dim8_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim8_efvs]
    dim9_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim9_efvs]
    dim10_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim10_efvs]
    dim11_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim11_efvs]
    dim12_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim12_efvs]
    dim13_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim13_efvs]
    dim14_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim14_efvs]
    
    print(model.name)
    print("Cardinalities of irr_supps for:")
    print("8-face efms:", sorted(Counter(dim8_efv_irr_supp_lens).items()))
    print("9-face efms:", sorted(Counter(dim9_efv_irr_supp_lens).items()))
    print("10-face efms:", sorted(Counter(dim10_efv_irr_supp_lens).items()))
    print("11-face efms:", sorted(Counter(dim11_efv_irr_supp_lens).items()))
    print("12-face efms:", sorted(Counter(dim12_efv_irr_supp_lens).items()))
    print("13-face efms:", sorted(Counter(dim13_efv_irr_supp_lens).items()))
    print("14-face efms:", sorted(Counter(dim14_efv_irr_supp_lens).items()))
    print("")
    print("MMB sizes:", sorted(Counter(mmb_lens).items()))
    
    model = models[2]
    model.make_irredundant()
    
    model.get_geometry()
    model.get_efvs("efmtool")
    model.get_mmbs()
    mmb_lens = [len(mmb) for mmb in model.mmbs]
    
    dim1_efvs = []
    dim2_efvs = []
    dim3_efvs = []
    dim4_efvs = []
    dim5_efvs = []
    dim6_efvs = []
    
    for efv in model.efvs:
        dim = model.check_dim(efv)
        
        if dim == 1:
            dim1_efvs.append(efv)
        if dim == 2:
            dim2_efvs.append(efv)
        if dim == 3:
            dim3_efvs.append(efv)
        if dim == 4:
            dim4_efvs.append(efv)
        if dim == 5:
            dim5_efvs.append(efv)
        if dim == 6:
            dim6_efvs.append(efv)
    
    
    dim1_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim1_efvs]
    dim2_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim2_efvs]
    dim3_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim3_efvs]
    dim4_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim4_efvs]
    dim5_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim5_efvs]
    dim6_efv_irr_supp_lens = [len(model.irr_supp(efv)) for efv in dim6_efvs]
    print(model.name)
    print("Cardinalities of irr_supps for:")
    print("1-face efms:", sorted(Counter(dim1_efv_irr_supp_lens).items()))
    print("2-face efms:", sorted(Counter(dim2_efv_irr_supp_lens).items()))
    print("3-face efms:", sorted(Counter(dim3_efv_irr_supp_lens).items()))
    print("4-face efms:", sorted(Counter(dim4_efv_irr_supp_lens).items()))
    print("5-face efms:", sorted(Counter(dim5_efv_irr_supp_lens).items()))
    print("6-face efms:", sorted(Counter(dim6_efv_irr_supp_lens).items()))
    print("")
    print("MMB sizes:", sorted(Counter(mmb_lens).items()))