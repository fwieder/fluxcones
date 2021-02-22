from multiprocessing import Pool
from functions import get_mmbs, flux_model, get_efvs
import numpy as np
from util import model_paths

import time



model = flux_model("./Biomodels/bigg_models/" + model_paths[2] + ".xml")

print("model imported")

def efms_in_mmb(mmb):
    
    face_indices = model.rev.copy()
    face_indices[mmb] = 1
    
    S = model.stoich[:,np.nonzero(face_indices)[0]]
    rev = model.rev[np.nonzero(face_indices)[0]]
    
    res = get_efvs(S,rev,"cdd")
    
    efvs_in_mmb = np.zeros([np.shape(res)[0],np.shape(model.stoich)[1]])
    efvs_in_mmb[:,np.nonzero(face_indices)[0]] = res
    
    
    efms = [list(np.nonzero(np.round(efvs_in_mmb[i],5))[0]) for i in range(len(efvs_in_mmb))]
   
    efms_in_mmb =[]
    
    for efm in efms:
        if not set(efm).issubset(set(np.nonzero(model.rev)[0])):
            efms_in_mmb.append(efm)
    
  
    return efms_in_mmb

def get_efms_in_mmbs(model):
    
    mmb_start_time = time.time()
    mmbs = get_mmbs(model.stoich,model.rev)
    mmb_comp_time = time.time() - mmb_start_time
    print(len(mmbs), "MMBs calculated in %3dm %2ds" % (mmb_comp_time//60,mmb_comp_time%60))
    
    comp_start = time.time()
    with Pool(8) as p:
        mmb_efms = p.map(efms_in_mmb, mmbs)
    finding_time = time.time() - comp_start
    print("EFMs in MMBs found in %3dm %2ds" %(finding_time//60,finding_time%60))
    return mmb_efms

if __name__ == '__main__':
    
    mmb_efms = get_efms_in_mmbs(model)
    