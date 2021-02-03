import numpy as np
import cdd
import efmtool
from util import printProgressBar
import time


'''
define tolerance for zero comparisions
'''
tol = 1e-5


'''
get_gens returns a V-representation of a steady-state fluxcone defined by stoich and rev (stoichiometric matrix and {0,1}-reversible-reactions-vector c.f. sbml_import)
algo determines which algorithm is used to compute the V-representation of the fluxcone
'''
def get_gens(stoich,rev, algo = "cdd"):
    
    if algo == "cdd":
        
        
        # nonegs is the matrix defining the inequalities for each irreversible reachtion
        irr = (np.ones(len(rev)) - rev).astype(int)
        nonegs = np.eye(len(rev))[np.nonzero(irr)[0]]
        
        # initiate Matrix for cdd
        mat = cdd.Matrix(nonegs)
        mat.extend(stoich,linear = True)
        
        # generate polytope and compute generators
        poly = cdd.Polyhedron(mat)
        gens = poly.get_generators()
    
    return(gens)

'''
get_efvs returns an np-array containing the elementary fluxvectors of a fluxcone defined by stoich and rev
'''

def get_efvs(stoich,rev, algo = "cdd"):
    if algo == "cdd":
        
        original_shape = np.shape(stoich)
        rev_indices = np.nonzero(rev)[0]
        
        # split reversible reactions by appending columns
        S_split = np.c_[stoich,-stoich[:,rev_indices]]
        
        # compute generators of pointed cone by splitting (all reactions irreversible)
        res = np.array(get_gens(S_split,np.zeros(len(S_split[0]))))
        
        # reverse splitting by combining both directions that resulted from splitting
        orig = res[:,:original_shape[1]]
        torem = np.zeros(np.shape(orig))
        splits = res[:,original_shape[1]:]
        for i,j in enumerate(rev_indices):
            torem[:,j] = splits[:,i]
        unsplit = orig - torem
        tokeep = []
        
        # ignore spurious cycles
        for index,vector in enumerate(unsplit):
            if np.count_nonzero(np.round(vector,5)) > 0:
                tokeep.append(index)
        efvs = unsplit[tokeep]
        
        return(efvs)
    
    if algo == "efmtool":
        # initiate reaction names and metabolite names from 0 to n resp. m because 
        # efmtool needs these lists of strings as input
        reaction_names = np.arange(np.shape(stoich)[1]).astype(str)
        metabolite_names = np.arange(np.shape(stoich)[0]).astype(str)
        
        efvs = efmtool.calculate_efms(stoich,list(rev),reaction_names,metabolite_names).T
        return(efvs)
    
    

'''
generates the MMBs of a metabolic network using cdd
'''
def get_mmbs(stoich,rev):
    
    # list indices of irreversible reactions
    irr_indices = np.nonzero(np.ones(len(rev)) - rev)[0]
    
    # compute v-representation using cdd (no splitting of reversible reactions)
    res = np.array(get_gens(stoich,rev))
    
    # compute MMBs from the v-representation
    mmbs = []
    for vector in res:
        mmb = []
        for index in irr_indices:
            if abs(vector[index]) > tol:
                mmb.append(index)
        if mmb != []:
            mmbs.append(mmb)
    return(mmbs)

'''
Output:
mmb_efms:   list of lists. Each list contains all efms in the minimal proper face corresponding to the MMB
int_efms:   list of efms that are not in a minimal proper face and not in the reversible metabolic space
frev_efms:  list of efms that are fully reversible  (in the reversible metabolic space) 
'''
def filter_efms(efvs,mmbs,rev):
    
    # transform vectors to supports
    rev_indices = np.nonzero(rev)[0]
    efms = []
    
    for efm in efvs:
        reactions = list(np.nonzero(np.round(efm,5))[0])
        efms.append(reactions)
        
    
    # determine fully reversible efms
    frev_efms = []
    nonrev_efms = []
    for efm in efms:
        if set(efm) < set(rev_indices):
            if efm not in frev_efms:
                frev_efms.append(efm)
        else:
            nonrev_efms.append(efm)
    
    # initiate list of empty lists for each MMB (mmb_efms) and list of interior EFMs (int_efms)
    mmb_efms = [[] for n in range(len(mmbs))]
    int_efms = []
    
    start = time.perf_counter()
    print("Filtering EFMs")
    
    # iterate over all efms
    for ind,efm in enumerate(nonrev_efms):
        # matches will be the amount of mmbs that are a subset of 
        matches = 0
        for index,mmb in enumerate(mmbs):
            if set(mmb).issubset(set(efm)):
                matches +=1
                if matches > 1:
                    int_efms.append(efm)
                    break
                mmb_index = index
        if matches == 1:
            mmb_efms[mmb_index].append(efm)
        if ind % 100 == 0:
            printProgressBar(ind,len(nonrev_efms),starttime = start)
            
    return mmb_efms, int_efms , frev_efms