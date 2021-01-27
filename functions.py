import numpy as np
import cdd
import efmtool
from itertools import chain

def get_gens(S,rev, algo = "cdd"):
    if algo == "cdd":
        irr = (np.ones(len(rev)) - rev).astype(int)
    
        nonegs = np.eye(len(rev))[np.nonzero(irr)[0]]
       
        mat = cdd.Matrix(nonegs)
        mat.extend(S,linear = True)
        poly = cdd.Polyhedron(mat)
        gens = poly.get_generators()
        
    return(gens)

def get_efms(S,rev, algo = "cdd"):
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
        efms = unsplit[tokeep]
        return(efms)
    
    if algo == "efmtool":
        reaction_names = np.arange(np.shape(S)[1]).astype(str)
        metabolite_names = np.arange(np.shape(S)[0]).astype(str)
        efms = efmtool.calculate_efms(S,list(rev),reaction_names,metabolite_names).T
        return(efms)
    
    

    
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
    d = {}
    d2 = {}
    count = 0 
    
    for i in mmbs:
        
        count = count + 1
        set2 = set(i)
        d['set'+str(count)] = set2
        
        d['lists'+str(count)] = []
        first = []
        
        d2['match'+str(count)]  = []
        
        for a in norev_reactions:
        
            set1 = set(a)
            if d['set'+str(count)].issubset(set1) == True:
              
                first.append(a)    
        d['lists'+str(count)].append(first)  
        d2['match'+str(count)].append(d['lists'+str(count)])
        
    count = 0 
    count2 = -1
    d3 = {}
    all_sub_lists = []
    for i in d2.values():
        
        count = count + 1
        count2 = count2 + 1
        d3['final'+str(count)]  = []
    
        real = []
        for item in i:

            for each_item in item:
                           
                for each_each_item in each_item:
                    seta= set(each_each_item)
                    save = []
                    
                    
                    for i in mmbs:
                    
                        setb = set(i)
                        a=setb.issubset(seta)
    
                        save.append(a)
                        
                    index_to_remove = count2
                    new_save = save[:index_to_remove] + save[index_to_remove + 1:]
                    if True not in new_save:
                        real.append(each_each_item)
                        
            d3['final'+str(count)].append(real)
            
            all_sub_lists.append(real)
            
    mpf_efms = list(chain(*all_sub_lists))
    setA = set(map(tuple, mpf_efms))
    setB = set(map(tuple, norev_reactions))

    int_efms = [i for i in setB if i not in setA]
    
    return (frev_reactions,mpf_efms, int_efms)