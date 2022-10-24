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
    for model in models:
        model.make_irredundant()
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
        model.get_efvs("efmtool")
        print(len(model.efvs), "efms")
        model.get_geometry()
        model.faces_by_dim = [[] for x in range(model.get_cone_dim()+1)]
        for efv in model.efvs:
            model.faces_by_dim[model.check_dim(efv)].append(efv)
        lens = [len(face) for face in model.faces_by_dim]
        model.irr_supps_lens_by_face = [[] for x in range(model.get_cone_dim()+1)]
        for i,face in enumerate(model.faces_by_dim):
            if len(face) > 0:
                for efv in face:
                    model.irr_supps_lens_by_face[i].append(len(model.irr_supp(efv)))
        model.irr_supps_counters = [sorted(Counter(face).items()) for face in model.irr_supps_lens_by_face]
        for i,x in enumerate(model.irr_supps_counters):
            if x != []:
                print(str(i)+"-faces: " ,x, " irr_supp_len_bounds = " ,(i-model.get_lin_dim(), i + len(supp(model.irr)) - model.get_cone_dim()))
        print("new face dim bound:" , model.get_cone_dim() - len(supp(model.irr))+1 + model.get_lin_dim())