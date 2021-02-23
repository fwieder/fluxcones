from multiprocessing import Pool
from functions import get_mmbs, flux_model, get_efvs
import numpy as np
from util import model_paths
import tqdm
import time

'''
Parallel computation of EFMs in MMBs. Pool(8) in "get_efms_in_mmbs" sets the amount of used processors to 8
'''


model = flux_model("./Biomodels/bigg_models/" + model_paths[0] + ".xml")





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

def get_efms_in_mmbs(model,mmbs):
    
    with Pool(proces) as p:
        mmb_efms = list(tqdm.tqdm(p.imap(efms_in_mmb, mmbs), total = len(mmbs)))
    p.close()
    return mmb_efms

if __name__ == '__main__':
    
    mmb_start_time = time.time()
    mmbs = get_mmbs(model.stoich,model.rev)
    mmb_comp_time = time.time() - mmb_start_time
    print(len(mmbs), "MMBs calculated in %3dm %2ds" % (mmb_comp_time//60,mmb_comp_time%60))
    
    for proces in range(1,16):
        print("Withh proces = " , proces)
        mmb_efms = get_efms_in_mmbs(model,mmbs)
    