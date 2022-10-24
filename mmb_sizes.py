from flux_class_vecs import supp, flux_cone
import numpy as np
from sympy import Matrix
import tqdm
from multiprocessing import Pool
from collections import Counter
import matplotlib as mpl
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
patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
if __name__ == "__main__":
    for model in models:
        print(model.name)
        model.get_mmbs()
        print(len(model.mmbs))
    import sys
    sys.exit()
    
    
    model = models[1]
    model.redundant_mmbs = model.get_mmbs()
    model.red_mmb_lens = [len(mmb) for mmb in model.redundant_mmbs]
    model.make_irredundant()
    model.irredundant_mmbs = model.get_mmbs()
    model.irr_mmb_lens = [len(mmb) for mmb in model.irredundant_mmbs]
    
    red_counter = [0 for i in range(max(model.red_mmb_lens))]
    for pair in Counter(model.red_mmb_lens).items():
        red_counter[pair[0]-1] = pair[1]
    
    irr_counter = [0 for i in range(len(red_counter))]
    
    for index,mmb in enumerate(sorted(Counter(model.irr_mmb_lens).items())):
        irr_counter[index] = mmb[1]
    
    X = np.arange(max(model.red_mmb_lens))
    
    fig = plt.figure()
    
    ax = fig.add_axes([0,0,1,1])
    
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.bar(X + 1 -.125, red_counter, color = 'b', width = 0.25)
    ax.bar(X + 1 +.125, irr_counter, color = 'r',hatch = patterns[0],edgecolor = "black", width = 0.25)
    
    ax.legend(labels=['redundant', 'irredudant'])
    plt.title(model.name)
    plt.xlabel("Cardinality of MMB")
    plt.ylabel("Number of MMBs")
    
  
    