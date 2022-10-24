# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 13:47:17 2021

@author: Frederik Wieder
"""
from flux_class_vecs import flux_cone,supp
    
import numpy as np
import random
from multiprocessing import Pool
import tqdm
from scipy.optimize import linprog

class arc:
    def __init__(self,start_nodes,target_nodes, consume_rates=None, produce_rates=None):
        self.start_nodes = start_nodes
        self.target_nodes = target_nodes
        
        self.consume_rates = consume_rates
        self.produce_rates = produce_rates
        
        
        
class Hygraph:
    def __init__(self,amount_of_nodes,arcs = []):
        
        self.nodes = np.arange(amount_of_nodes)
        self.arcs = arcs
      
    def add_arc(self,arc):
        self.arcs.append(arc)
    
    def get_incidence(self):
        incidence = np.zeros((len(self.nodes),len(self.arcs)))
        for i,arc in enumerate(self.arcs):
            for j,neg in enumerate(arc.start_nodes):
                incidence[neg,i] = -arc.consume_rates[j] 
            for j,neg in enumerate(arc.target_nodes):
                incidence[neg,i] = arc.produce_rates[j]
            
        
        return incidence
    
def generate_random_network(amount_of_metabolites,amount_of_reactions,hyperarc_probability = 0.2,possible_weights = [1,2,3],weight_probabilities = [0.8,0.15,0.05]):
    G = Hygraph(amount_of_metabolites)
    number_of_ex_reacs = random.randint(2,amount_of_reactions-2)
    ex_metabs = np.random.choice(G.nodes,number_of_ex_reacs)
    for ex in ex_metabs:
        if random.random() < 0.5:
            start_nodes = [ex]
            target_nodes = []
            consume_rates = [np.random.choice(possible_weights,1,p =weight_probabilities)]
            produce_rates = [0]
        else:
            target_nodes = [ex]
            start_nodes = []
            consume_rates = [0]
            produce_rates = [np.random.choice(possible_weights,1,p = weight_probabilities)]
        G.add_arc(arc(start_nodes,target_nodes,consume_rates,produce_rates))
        
    
    for reac in range(amount_of_reactions-number_of_ex_reacs):
        start_nodes = [random.choice(G.nodes)]
        while random.random() < hyperarc_probability:
            if len(start_nodes) < amount_of_metabolites-1:
                start_nodes.append(random.choice(list(set(G.nodes)-set(start_nodes))))
        target_nodes = [random.choice(list(set(G.nodes)-set(start_nodes)))]
        while random.random() < hyperarc_probability:
            if len(target_nodes) < amount_of_metabolites - len(start_nodes):
                target_nodes.append(random.choice(list(set(G.nodes)-set(start_nodes)-set(target_nodes))))
        consume_rates = np.random.choice(possible_weights,len(start_nodes),p = weight_probabilities)
        produce_rates = np.random.choice(possible_weights,len(target_nodes),p = weight_probabilities)
        G.add_arc(arc(start_nodes,target_nodes,consume_rates,produce_rates))
        
    
    incidence = G.get_incidence()
    
    rev = np.random.choice([0,1],len(G.arcs))
    while len(supp(rev)) == 0  or len(supp(rev)) == len(rev):
        rev = np.random.choice([0,1],len(G.arcs))
    return flux_cone("Random network",incidence[~np.all(incidence==0,axis=1)],rev)
    
if __name__ == "__main__":
    
    model = generate_random_network(8,20)
    model.name = "random model"
    model.make_irredundant()
    model.get_mmbs()
    mmb_lens = [len(mmb) for mmb in model.mmbs]
    gamma = max(mmb_lens)
    if gamma < 2:
        print("simplicial")
        import sys
        sys.exit()
    model.get_efvs()
    print("number of efms: ", len(model.efvs))
    print("gamma = ", gamma)
    
    dims_irr = [(model.check_dim(efv),len(model.irr_supp(efv))) for efv in tqdm.tqdm(model.efvs)]
    t = model.get_lin_dim()
    print("")
    print("lin_dim =" , t)
    from operator import itemgetter
    print("Mag degree:", max(dims_irr,key=itemgetter(0)))
    print("max irrsupp: ", max(dims_irr,key=itemgetter(1)))
    print("testing conjecture")
    for deg_irr in dims_irr:
        if deg_irr[0]-t-1+gamma<deg_irr[1]:
            print("False: ", deg_irr)
            import sys
            sys.exit()
    print("conjecture true for ", model.name)
    