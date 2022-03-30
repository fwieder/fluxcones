from flux_class_vecs import supp, flux_cone
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

    models.append(model)


if __name__ == "__main__":
    models = models
    for model in models:
        model.redundant_mmbs = model.get_mmbs()
        model.red_mmb_lens = [len(mmb) for mmb in model.redundant_mmbs]
        model.make_irredundant()
        model.irredundant_mmbs = model.get_mmbs()
        model.irr_mmb_lens = [len(mmb) for mmb in model.irredundant_mmbs]
   
    mpl.rcParams['figure.dpi'] = 600

    data = [[model.red_mmb_lens,model.irr_mmb_lens] for model in models]
    columns = [ 'redundant description' ,'irredudant description']
    rows = list(model.name for model in models)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    the_table = plt.table(cellText=data,rowLabels=rows,colLabels=columns,loc = 'center')
    the_table.scale(1,1.5)
    plt.box(on=None)
    plt.show()