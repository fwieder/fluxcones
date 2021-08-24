# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 13:47:17 2021

@author: Frederik Wieder
"""

from flux_class_vecs import flux_cone
import numpy as np

model = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")
model.delete_reaction(12)
#model = flux_cone.from_kegg("./Biomodels/kegg/Nitrogen/kegg910")
tol = 1e-10
class hypergraph:
    def __init__(self,model):
        
        self.vertices = np.arange(len(model.stoich))
        rev = set(np.nonzero(model.rev)[0])
        edges = []
        for index,reaction in enumerate(model.stoich.T):
            if index in rev:
                edge = ["rev"]
            else:
                edge = ["irr"]
            outs = tuple(np.where(reaction > tol)[0])
            ins = tuple(np.where(reaction < -tol)[0])
            
            edge.insert(0,[ins,outs])
            edges.append(edge)
        self.edges = edges
    
        
    def is_exchange(self,reaction_index):
        if len(self.edges[reaction_index][0][0]) == 0 or len(self.edges[reaction_index][0][1]) == 0:
            return True
        else:
            return False
    
    def is_hyper(self,reaction_index):
        if len(self.edges[reaction_index][0][0]) > 1 or len(self.edges[reaction_index][0][1]) > 1:
            return True
        else:
            return False
        
G = hypergraph(model)

internal = []
for i,edge in enumerate(G.edges):
    if not G.is_exchange(i):
        internal.append(i)
        
print(internal, len(internal))

model.get_efvs()
print(len(model.efvs))

efms = [tuple(np.nonzero(efv)[0]) for efv in model.efvs]

cyc_efms = []
for efm in efms:
    if set(efm) < set(internal):
        cyc_efms.append(efm)
        
        
print(len(cyc_efms))

