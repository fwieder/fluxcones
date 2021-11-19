# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 13:47:17 2021

@author: Frederik Wieder
"""

from flux_class_vecs import flux_cone,supp

import numpy as np
import random

model = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")
model.delete_reaction(12)
efvs = np.load("./e_coli_no_bio_efvs.npy")

class arc:
    def __init__(self,start_nodes,target_nodes, consume_rates=None, produce_rates=None):
        self.start_nodes = start_nodes
        self.target_nodes = target_nodes
        if consume_rates == None:
            self.consume_rates = np.ones(len(self.start_nodes))
        else:
            self.consume_rates = consume_rates
        if produce_rates == None:
            self.produce_rates = np.ones(len(self.target_nodes))
        else:
            self.produce_rates = produce_rates
        
        
        
class Hygraph:
    def __init__(self,amount_of_nodes,arcs = []):
        
        self.nodes = np.arange(amount_of_nodes)
        self.arcs = arcs
      
    def add_arc(self,arc):
        self.arcs.append(arc)
    
    def get_stoich(self):
        stoich = np.zeros((len(self.nodes),len(self.arcs)))
        for i,arc in enumerate(self.arcs):
            for j,neg in enumerate(arc.start_nodes):
                stoich[neg,i] = -arc.consume_rates[j] 
            for j,neg in enumerate(arc.target_nodes):
                stoich[neg,i] = arc.produce_rates[j]
            
        
        return stoich
    
def generate_random_network(amount_of_metabolites,amount_of_reactions,hyperarc_probability = 0.1):
    G = Hygraph(amount_of_metabolites)
    for reac in range(amount_of_reactions):
        
G = Hygraph(6)
G.add_arc(arc([1,2],[3],[1,2],[1]))
G.add_arc(arc([2,3],[4,5],[1,1],[2,2])) 
G.add_arc(arc([1],[2]))
G.add_arc(arc([5],[3],[3],[2]))
print(G.get_stoich())   