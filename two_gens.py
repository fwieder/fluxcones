from multiprocessing import Pool
from flux_class_vecs import flux_cone,supp
import numpy as np
import tqdm,sys,time

tol = 1e-14
digits = 14
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
    
    models.append(model)



model = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")

model.efvs = np.load("./e_coli_efvs.npy")
model.generators = np.load("./e_coli_gens.npy")


if __name__ == "__main__":
    
    print(model.name)
    #model.get_efvs("efmtool")
    #model.get_geometry()
    gen_efms = [supp(gen) for gen in model.generators]
    
    non_gen_inds = []
    for i,efv in enumerate(model.efvs):
        if supp(efv) not in gen_efms:
            non_gen_inds.append(i)
    
    efvs = model.efvs[non_gen_inds]

    
    
    with Pool(12) as p:
        two_gen_pairs = list(tqdm.tqdm(p.imap(model.two_gens,efvs),total= len(efvs)))
        p.close()
        
    from collections import Counter
    lens = [len(pair) for pair in two_gen_pairs]
    print(sorted(Counter(lens).items()))
    #np.save("./two_gens_results/two_gens_all", two_gen_pairs)
    