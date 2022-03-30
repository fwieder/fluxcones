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

def irr_supp(vector):
    return list(np.intersect1d(supp(vector),irr))

def irr_eqs(vector):
    return(list(set(supp(model.irr))-set(irr_supp(vector))))

def zero(vector):
    return(list(set(np.arange(len(vector)))-set(supp(vector))))






if __name__ == "__main__":
    models = models
    for model in models:
        print(model.name)
        #model.efvs = np.load("./e_coli_no_bio_efvs.npy")
        #model.generators = np.load("./e_coli_no_bio_gens.npy")
        #model.efvs = np.load("./e_coli_no_bio_efvs.npy")
        #model.generators = np.load("e_coli_no_bio_gens.npy")
        model.get_geometry()
        model.get_efvs("efmtool")
        
        gens = model.generators
        model.cone_dim = np.linalg.matrix_rank(model.generators)
        print("cone_dim:", np.linalg.matrix_rank(model.generators))
        efvs = model.efvs
    
        print("lin_dim:" , model.get_lin_dim())
        model.S_irr_dim = np.linalg.matrix_rank(model.stoich[:,supp(model.irr)])
        print("rank(S_Irr) : ", model.S_irr_dim)
        model.max_face_dim = model.lin_dim + model.S_irr_dim + 1
        print("Max face dim:", model.max_face_dim)
        
        print("Dimensions of faces all efvs are in with numpy.linalg:")

        model.dim_efvs = [model.check_dim(efv) for efv in model.efvs]
        model.efv_dim_counter = sorted(Counter(model.dim_efvs).items())
        model.percents = [tuple([dim[0],round(dim[1]/len(model.efvs)*100,2)]) for dim in model.efv_dim_counter]
        model.actual_max = model.efv_dim_counter[-1][0]
        print("Highest dimensional face containing EFMS:" , model.efv_dim_counter[-1][0])
        
        print("")
        
   
    mpl.rcParams['figure.dpi'] = 600

    data = [[np.shape(model.stoich),(len(supp(model.irr)),len(supp(model.rev))),model.cone_dim,model.lin_dim,model.S_irr_dim,model.max_face_dim,model.actual_max] for model in models]
    columns = [ 'shape S' ,'# (irr,rev)' , 'Cone_dim' , 'lin_dim' , 'S_irr_dim' , 'face bound' , 'actual max face dim']
    rows = list(model.name for model in models)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    the_table = plt.table(cellText=data,rowLabels=rows,colLabels=columns,loc = 'center')
    the_table.scale(1,1.5)
    plt.box(on=None)
    plt.show()