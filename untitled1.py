#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:18:33 2021

@author: fred
"""

from flux_class_vecs import flux_cone

import numpy as np
import sys

tol = 5


model = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")

#model.delete_reaction(12)

#model = flux_cone.from_kegg("./Biomodels/small_examples/net-P4/net-P4")

adj = model.get_adjacency()

gens = model.get_generators()

rev=np.nonzero(model.rev)[0]

new_efvs = []
def num_of_cancels(vector1,vector2):
    if len(vector1) != len(vector2):
        print("error")
        return(0)
    
    vec1 = np.round(vector1,tol)[rev]
    vec2 = np.round(vector2,tol)[rev]
    
    cancels = 0
    
    no_dubs = []
    for ind in range(len(vec1)):
        if (vec1[ind] > 0 and vec2[ind] < 0) or (vec1[ind] < 0 and vec2[ind] > 0):
            if (vec1[ind],vec2[ind]) not in no_dubs:
                cancels +=1
                no_dubs.append((vec1[ind],vec2[ind]))
                if model.is_efv(abs(vec2[ind])*np.round(vector1,tol) + abs(vec1[ind])*np.round(vector2,tol)):
                    new_efvs.append(abs(vec2[ind])*np.round(vector1,tol) + abs(vec1[ind])*np.round(vector2,tol))
    return cancels






pairs = []
for ind1,adj_list in enumerate(model.adjacency):
            for ind2 in adj_list:
                if sorted((ind1,ind2)) not in pairs:
                    pairs.append(sorted((ind1,ind2)))

counter = 0
for pair in pairs:
    counter += num_of_cancels(gens[pair[0]],gens[pair[1]])

new_efms=[]
for efv in new_efvs:
    new_efms.append(list(np.nonzero(efv)[0]))

new_efms = np.unique(np.array(new_efms))
print(counter)
print(len(new_efms))
sys.exit()
model.get_mmbs()

print(len(model.mmbs))
print(len(model.mmbs)+len(new_efms))
face2_efms = model.get_efms_in_all_2faces()[0]
print(len(face2_efms))