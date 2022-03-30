# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 11:37:24 2021

@author: Frederik Wieder
"""
from flux_class_vecs import flux_cone,supp
        
import numpy as np
import random
import tqdm
import time

from multiprocessing import Pool
from util import printProgressBar
from collections import Counter

start_time = time.perf_counter()
total_iterations = 100

gen_lens = []

efv_lens = []
lin_dims = []
cone_dims = []
face_dims = []
irr_lens = []
metabs = 5
reacs = 9

def check_model(model):
    gens = model.generators
    
    gen_lens.append(len(gens))
    
    efvs = model.efvs
    efv_lens.append(len(efvs))
    cone_dims.append(model.get_cone_dim())
    
    
    if len(gens) == len(efvs):
        return True
    
    if len(gens) < len(efvs):
        gen_efms = [supp(gen) for gen in model.generators]
        non_gen_inds = []
        for i,efv in enumerate(model.efvs):
            if supp(efv) not in gen_efms:
                non_gen_inds.append(i)
            
    efvs = model.efvs[non_gen_inds] 
    if len(efvs) < 2500:
        two_gen_pairs = list(map(model.two_gens,efvs))
    else:
        with Pool(8) as p:
            two_gen_pairs = list(tqdm.tqdm(p.imap(model.two_gens,efvs),total= len(efvs)))
            p.close()
    lens = [len(pair) for pair in two_gen_pairs]
    counter = sorted(Counter(lens).items())
    if counter[0][0] == 2:
        return True
    
    return False
    
    
        
for iteration in range(total_iterations):

    if iteration%1 ==0:
        printProgressBar(iteration, total_iterations,start_time)
    def random_stoich(metabolites,reactions, sparsity = 0.8 , value_range = [1,-1]):
        stoich = np.zeros((metabolites,reactions))
        for m in range(metabolites):
            for n in range(reactions):
                if random.random() > sparsity:
                    stoich[m,n] = random.choice(value_range)
        return(stoich)
    
    stoich = random_stoich(metabs,reacs,0.2,[-2,-1,1,2])
    zero_column_indices = []
    for column_index,column in enumerate(stoich.T):
        if len(np.nonzero(column)[0]) == 0:
            zero_column_indices.append(column_index)
    non_zero_cols = list(set(np.arange(len(stoich[0])))-set(zero_column_indices))
    stoich = stoich[:,non_zero_cols]
            
    rev = np.random.choice([0,1],len(stoich[0]))
    while len(np.nonzero(rev)[0]) == 0 or len(np.nonzero(rev)[0]) == len(rev):
        rev = np.random.choice([0,1],len(stoich[0]))
    model = flux_cone("random mocdel", stoich, rev)
    model.get_geometry()
    model.get_efvs("cdd")
    model.dim_efvs = list(map(model.check_dim,model.efvs))
    model.efv_dim_counter = sorted(Counter(model.dim_efvs).items())
    face_dims.append(model.efv_dim_counter)
    model.get_lin_dim()
    lin_dims.append(model.lin_dim)
    irr_lens.append(len(supp(model.irr)))
    if __name__ == "__main__":  
        if not check_model(model):
            print("Maybe actual counterexample")
            import sys
            sys.exit()
            
max_efms = max(efv_lens)
max_gens = max(gen_lens)
print(" ")
print("Found no counterexamples in", total_iterations, "random models with", metabs,"metabolites and", reacs, "reactions.")
print("Amounts of EFMs:")
print(efv_lens)
print("Amounts of generators:")
print(gen_lens)
print("Cone dims:")
print(cone_dims)
print("Lineality space dimensions:")
print(lin_dims)
print("face dimensions:")
print(face_dims)         
print(irr_lens)       