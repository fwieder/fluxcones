import numpy as np
import cdd,efmtool,cobra,time
from util import printProgressBar


'''
define tolerance for zero comparisions
'''
tol = 1e-5




'''
class of model containg the relevant information of a metabolic network. Should be called with a path to sbml-file
'''
class flux_model:
    def __init__(self, path_to_file):
        sbml_model = cobra.io.read_sbml_model(path_to_file)
        
        self.name = sbml_model.name
        
        self.stoich = cobra.util.array.create_stoichiometric_matrix(sbml_model)
        
        self.rev = np.array([rea.reversibility for rea in sbml_model.reactions]).astype(int)
        
        self.irr = (np.ones(len(self.rev)) - self.rev).astype(int)
        
        self.lin_dim = len(np.nonzero(self.rev)[0]) - np.linalg.matrix_rank(self.stoich[:,np.nonzero(self.rev)[0]])



'''
returns a V-representation of a steady-state fluxcone defined by stoich and rev (stoichiometric matrix and {0,1}-reversible-reactions-vector c.f. sbml_import)
algo determines which algorithm is used to compute the V-representation of the fluxcone
'''
def get_gens(equations,vec_of_free_indices, algo = "cdd"):
    # so far only implemented for algo == cdd
    if algo == "cdd":
        # nonegs is the matrix defining the inequalities for each irreversible reachtion
        noneg_inds = (np.ones(len(vec_of_free_indices)) - vec_of_free_indices).astype(int)
        nonegs = np.eye(len(vec_of_free_indices))[np.nonzero(noneg_inds)[0]]
        
        
        # initiate Matrix for cdd
        if len(nonegs) > 0:
            mat = cdd.Matrix(nonegs,number_type = 'float')
            mat.extend(equations,linear = True)
        else:
            mat = cdd.Matrix(equations,linear = True)
        
        
        # generate polytope and compute generators
        poly = cdd.Polyhedron(mat)
        gens = poly.get_generators()
    
    return(gens)



'''
returns an np-array containing the elementary fluxvectors the input model
the input parameter "algo" is used to determine the Algorithm that is used to enumearte the elementary flux vectors
'''
def get_efvs(model, algo = "cdd"):
    if algo == "cdd":
        # Store information about original shape to be able to revert splitting of reversible reactions later
        original_shape = np.shape(model.stoich)
        rev_indices = np.nonzero(model.rev)[0]
        
        
        # split reversible reactions by appending columns
        S_split = np.c_[model.stoich,-model.stoich[:,rev_indices]]
        
        
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
        reaction_names = list(np.arange(len(model.stoich[0])).astype(str))
        metabolite_names = list(np.arange(len(model.stoich)).astype(str))
        efvs = efmtool.calculate_efms(model.stoich,model.rev,reaction_names,metabolite_names)
        
        
        return(efvs.T)



'''
returns a list containing the MMBs of a metabolic network using cdd
'''
def get_mmbs(model):
    # list indices of irreversible reactions
    irr_indices = np.nonzero(np.ones(len(model.rev)) - model.rev)[0]
    
    
    # compute v-representation using cdd (no splitting of reversible reactions)
    res = np.array(get_gens(model.stoich,model.rev))
    
    
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
Filters a set of computed elementary flux vectors to determine, in which part of the fluxcone the fluxvectors lie

Output:
mmb_efms:   list of lists. Each list contains all efms in the minimal proper face corresponding to the MMB
int_efms:   list of efms that are not in a minimal proper face and not in the reversible metabolic space
frev_efms:  list of efms that are fully reversible  (in the reversible metabolic space) 
'''


def filter_efms(efvs,mmbs,rev):
    # transform flux-vectors to flux-modes by determining their supports
    rev_indices = np.nonzero(rev)[0]
    efms = []
    for efv in efvs:
        reactions = list(np.nonzero(np.round(efv,5))[0])
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
    
        
        # matches is the number of mmbs that are a subset of the current efm 
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
    printProgressBar(ind,len(nonrev_efms),starttime = start)
    
    for efmlist in mmb_efms:
        efmlist.sort()
    
    return mmb_efms, int_efms , frev_efms



'''
returns True, iff fluxmode is the support of an elementary flux-vector of the model
'''
def is_efm(fluxmode,model):
    if np.linalg.matrix_rank(model.stoich[:,fluxmode]) == len(fluxmode) - 1:
        return True
    else:
        return False



'''
Finds all efms of the given model that are in the minimal proper face defined by the input mmb. 
Function is defined again with default model=model, to work with multiprocessing
'''
def efms_in_mmb(mmb,model):
    # initiate temporaray model that defines the cone of minimal proper face defined by the input-mmb
    face_indices = model.rev.copy()
    face_indices[mmb] = 1
    
    face = type('min_face', (object,), {})()
    face.stoich = model.stoich[:,np.nonzero(face_indices)[0]]
    face.rev = model.rev[np.nonzero(face_indices)[0]]
    
    
    # determine elementary flux vectors in the temporary cone
    res = get_efvs(face,"cdd")
    
    
    # transform to original shape to match reaction indices of original model
    efvs_in_mmb = np.zeros([np.shape(res)[0],np.shape(model.stoich)[1]])
    efvs_in_mmb[:,np.nonzero(face_indices)[0]] = res
    
    
    # transfrom to modes by determining supports
    efms = [list(np.nonzero(np.round(efvs_in_mmb[i],5))[0]) for i in range(len(efvs_in_mmb))]
   
    
    # "forget" fully reversible efms
    efms_in_mmb =[]
    for efm in efms:
        if not set(efm).issubset(set(np.nonzero(model.rev)[0])):
            efms_in_mmb.append(efm)
  
    efms_in_mmb.sort()
    return(efms_in_mmb)



'''
determine all mmbs of the model and then determine the efms in each of the mmbs.
This version is unparallelized and much slower than the newer version in the multitest file
'''
def get_efms_in_mmbs(model):
    mmbs = get_mmbs(model)
    mmb_efms = []
    start = time.perf_counter()
    for ind,mmb in enumerate(mmbs):
        mmb_efms.append(efms_in_mmb(mmb,model))
        printProgressBar(ind,len(mmbs),starttime = start)
    return mmb_efms



'''
Write mmb_efms into a txt file
'''
def write_results(model_name,mmb_efms):
    f = open("./Results/" + model_name + "_efms_in_mmbs.txt","w")
    f.write(str(mmb_efms))
    f.close()
    return True



'''
determine all fully reversible efms. They are in the linear subspace of the fluxcone
'''
def get_frev_efms(model):
    # initiate temporaray model that defines the reversible metabolic space
    
    rms = type('rms', (object,), {})()
    rms.stoich = model.stoich[:,np.nonzero(model.rev)[0]]
    rms.rev = np.ones(len(rms.stoich[0]))
    
    
    # determine elementary flux vectors in the temporary cone
    res = get_efvs(rms,"cdd")
    
    
    # transform to original shape to match reaction indices of original model
    frev_efvs = np.zeros([np.shape(res)[0],np.shape(model.stoich)[1]])
    frev_efvs[:,np.nonzero(model.rev)[0]] = res
    
    
    # transfrom to modes by determining supports
    frev_efms = []
    for efv in frev_efvs:
        efm = list(np.nonzero(np.round(efv,5))[0])
        if efm not in frev_efms:
            frev_efms.append(efm)
    frev_efms.sort()
    return(frev_efms)


