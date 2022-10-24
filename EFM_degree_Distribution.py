

from flux_class_vecs import supp, flux_cone
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt  
from matplotlib.ticker import MaxNLocator
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

coli = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")

coli.delete_reaction(12)
models.append(coli)

if __name__ == "__main__":
    
    for model in models:
        model.get_efvs("efmtool")
    
        model.dim_efvs = [model.check_dim(efv) for efv in model.efvs]
        model.efv_dim_counter = np.array(sorted(Counter(model.dim_efvs).items()))
        
        fig = plt.figure()
        
        plt.bar(model.efv_dim_counter[:,0],model.efv_dim_counter[:,1])
        plt.title(model.name)
        plt.xlabel("Degree of EFM")
        plt.ylabel("Number of EFMs")