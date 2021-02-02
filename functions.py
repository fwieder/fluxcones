import numpy as np
import cdd
import efmtool
from util import printProgressBar
import time

def get_gens(S,rev, algo = "cdd"):
    if algo == "cdd":
        irr = (np.ones(len(rev)) - rev).astype(int)
    
        nonegs = np.eye(len(rev))[np.nonzero(irr)[0]]
       
        mat = cdd.Matrix(nonegs)
        mat.extend(S,linear = True)
        poly = cdd.Polyhedron(mat)
        gens = poly.get_generators()
    return(gens)

def get_efvs(S,rev, algo = "cdd"):
    if algo == "cdd":
        original_shape = np.shape(S)
        rev = np.nonzero(rev)[0]
        S_split = np.c_[S,-S[:,rev]]
        res = np.array(get_gens(S_split,np.zeros(len(S_split[0]))))
        orig = res[:,:original_shape[1]]
        torem = np.zeros(np.shape(orig))
        splits = res[:,original_shape[1]:]
        for i,j in enumerate(rev):
            torem[:,j] = splits[:,i]
        unsplit = orig - torem
        tokeep = []
        for index,vector in enumerate(unsplit):
            if np.count_nonzero(np.round(vector,5)) > 0:
                tokeep.append(index)
        efvs = unsplit[tokeep]
        return(efvs)
    
    if algo == "efmtool":
        reaction_names = np.arange(np.shape(S)[1]).astype(str)
        metabolite_names = np.arange(np.shape(S)[0]).astype(str)
        efvs = efmtool.calculate_efms(S,list(rev),reaction_names,metabolite_names).T
        return(efvs)
    
    

    
def get_mmbs(S,rev):
    irr = np.nonzero(np.ones(len(rev)) - rev)[0]
    res = np.array(get_gens(S,rev))
    mmbs = []
    for vector in res:
        mmb = []
        for index in irr:
            if abs(vector[index]) > 1e-5:
                mmb.append(index)
        if mmb != []:
            mmbs.append(mmb)
    mmbs.sort()
    return(mmbs)

def sort_efms(efms,mmbs,rev):
    rev = np.nonzero(rev)[0]
    efm_reactions = []
    for efm in efms:
        reactions = list(np.nonzero(np.round(efm,5))[0])
        efm_reactions.append(reactions)

    frev_reactions = []
    norev_reactions = []

    for efm in efm_reactions:
        if set(efm) < set(rev):
            if efm not in frev_reactions:
                frev_reactions.append(efm)
        else:
            norev_reactions.append(efm)
    
    
    mmb_efms = [[] for n in range(len(mmbs))]
    int_efms = []
    
    start = time.perf_counter()
    print("Filtering EFMs")
    
    for ind,efm in enumerate(norev_reactions):
        matches = 0
        for index,mmb in enumerate(mmbs):
            if set(mmb).intersection(set(efm)) == set(mmb):
                matches +=1
                if matches > 1:
                    int_efms.append(efm)
                    break
                mmb_index = index
        if matches == 1:
            mmb_efms[mmb_index].append(efm)
        if ind % 100 == 0:
            printProgressBar(ind,len(norev_reactions),starttime = start)
            
    return mmb_efms, int_efms , frev_reactions