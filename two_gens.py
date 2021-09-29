from multiprocessing import Pool
from flux_class_vecs import flux_cone,supp
import numpy as np
import tqdm,sys,time

tol = 1e-14
digits = 14

model = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")

irr = supp(model.irr)

all_efvs = np.load("./e_coli_efvs.npy")
gens = np.load("./e_coli_gens.npy")

gen_efms = [supp(gen) for gen in gens]

non_gen_inds = []
for i,efv in enumerate(all_efvs):
    if supp(efv) not in gen_efms:
        non_gen_inds.append(i)

efvs = all_efvs[non_gen_inds]


def irr_supp(vector):
    return list(np.intersect1d(supp(vector),irr))


    
def two_gens(vector):
    efm = supp(vector)
    rev = supp(model.rev)
    rev_zeros = []
    for reaction_index , reaction_value  in enumerate(vector):
        if np.round(reaction_value,digits) == 0 and reaction_index in rev:
            rev_zeros.append(reaction_index)        
    candidate_inds = []
    
    for ind,efv_0 in enumerate(all_efvs):
        if set(irr_supp(efv_0)) <= set(irr_supp(vector)) and not supp(efv_0) == supp(vector):
            candidate_inds.append(ind)
    
    
    candidates = all_efvs[candidate_inds]
    

    for rev_zero_ind in rev_zeros:
        pos = candidates[np.where(candidates[:,rev_zero_ind] > tol)]
        neg = candidates[np.where(candidates[:,rev_zero_ind] < -tol)]
        
        if len(pos) > 0  and len(neg) > 0:
            for pos_efv in pos:
                for neg_efv in neg:
                    new_vec = pos_efv[rev_zero_ind]*neg_efv - neg_efv[rev_zero_ind]*pos_efv
                    if set(supp(new_vec)) == set(efm):
                            
                        return (pos_efv,neg_efv)                    
                    
    return ([])



if __name__ == "__main__":

    
    i=5000
    
    k = i +5000    
    
    print("From",i,"to",k)
    
    efvs = efvs[i:k]
    
    with Pool(12) as p:
        two_gen_pairs = list(tqdm.tqdm(p.imap(two_gens,efvs),total= len(efvs)))
        p.close()
        
    from collections import Counter
    lens = [len(pair) for pair in two_gen_pairs]
    print(Counter(lens))
    np.save("./two_gens_results/two_gens_from" + str(i) + "to" + str(k) , two_gen_pairs)
    