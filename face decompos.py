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

if __name__ == "__main__": 
    for model in models:
        print(model.name)
        print("m,n", np.shape(model.stoich))
        model.get_geometry()
        print("irr:", len(supp(model.irr)))
        print("rev:", len(supp(model.rev)))
        print("rank(S)", np.linalg.matrix_rank(model.stoich))
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
        for reaction in model.blocked_irr_reactions():
            model.delete_reaction(reaction)
        model.get_geometry()
        gen_efms = [supp(efv) for efv in model.generators]
        min_efms = []
        for efm in gen_efms:
            if not set(efm) < set(supp(model.rev)):
                min_efms.append(efm)
        min_efvs = []
        for efv in model.generators:
            if supp(efv) in min_efms:
                min_efvs.append(efv)
        gens = np.array(min_efvs)
        
        model.get_mmbs()
        mmb_lens = [len(mmb) for mmb in model.mmbs]
        max_len = max(mmb_lens)
        print(model.name)
        print("lin_dim:", model.get_lin_dim())
        print("pointed_dim:", np.linalg.matrix_rank(gens))
        print("cone_dim", model.get_cone_dim())
        print("max_mmb_len:", max_len)
        print("Bound:" ,len(supp(model.irr))-np.linalg.matrix_rank(gens)+1)
    
    if __name__ == "__main__":
        model = models[0]
        print(model.name)
        model.make_irredundant()
        
        model.get_geometry()
        model.get_efvs("efmtool")
        print(len(model.efvs), "efms")  
        model.get_mmbs()
        mmb_lens = [len(mmb) for mmb in model.mmbs]
        
        dim_efvs = [[] for i in range(model.get_cone_dim())]
        
        for efv in model.efvs:
            dim_efvs[model.check_dim(efv)].append(efv)
            
        counter = [[] for i in range(len(dim_efvs))]
        
        for i,face in enumerate(dim_efvs):
            counter[i].append([model.irr_supp(efv) for efv in face])
        
        import itertools
        for i,k in enumerate(counter):
            k[0].sort()
            counter[i] = list(k for k,_ in itertools.groupby(k[0]))
        
    
    #%%
        k_faces_dims = [[] for i in range(len(counter))]
        
        for i in range(len(counter)):
            print("")
            print("Handling", len(counter[i]), str(i) + "-faces")
            start = time.perf_counter()
            for index,support in enumerate(list(counter[i])):
                printProgressBar(index, len(counter[i]),starttime = start)
                face = model.face_by_irr_supp(support,"efmtool")
                k_faces_dims[i].append(sorted(Counter([model.check_dim(efv) for efv in face]).items()))
            
    #%%
        
        for k in range(len(k_faces_dims)):
            print(str(k)+"-faces:" , Counter(tuple(e) for e in k_faces_dims[k]))    
    """