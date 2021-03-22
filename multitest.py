from multiprocessing import Pool
from functions import get_mmbs, flux_model, get_efvs, write_results,get_frev_efms
from util import model_paths
import numpy as np
import tqdm,time,sys

'''
Parallel computation of EFMs in MMBs. proces = 8 in "get_efms_in_mmbs" sets the amount of parallel computations to a default value of 8
'''

model_name = model_paths[1]
model = flux_model("./Biomodels/bigg/" + model_name + ".xml")





def efms_in_mmb(mmb,model = model):
    
    face_indices = model.rev.copy()
    face_indices[mmb] = 1
    
    face = type('min_face', (object,), {})()
    face.stoich = model.stoich[:,np.nonzero(face_indices)[0]]
    face.rev = model.rev[np.nonzero(face_indices)[0]]
    
    res = get_efvs(face,"efmtool")
    
    efvs_in_mmb = np.zeros([np.shape(res)[0],np.shape(model.stoich)[1]])
    efvs_in_mmb[:,np.nonzero(face_indices)[0]] = res
    
    
    efms = [list(np.nonzero(np.round(efvs_in_mmb[i],5))[0]) for i in range(len(efvs_in_mmb))]
   
    efms_in_mmb =[]
    
    for efm in efms:
        if not set(efm).issubset(set(np.nonzero(model.rev)[0])):
            efms_in_mmb.append(efm)
  
    efms_in_mmb.sort()
    return(efms_in_mmb)

def get_efms_in_mmbs(model, proces = 1,mmbs = None):
    if mmbs == None:
        mmb_start_time = time.time()
        mmbs = get_mmbs(model)
        mmb_comp_time = time.time() - mmb_start_time
        print(len(mmbs), "MMBs calculated in %3dm %2ds" % (mmb_comp_time//60,mmb_comp_time%60))
    
    with Pool(proces) as p:
        mmb_efms = list(tqdm.tqdm(p.imap(efms_in_mmb,mmbs), total = len(mmbs)))
    p.close()
    return mmb_efms


if __name__ == '__main__':
    mmb_efms = get_efms_in_mmbs(model)
    