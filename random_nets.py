# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 13:47:17 2021

@author: Frederik Wieder
"""
or it in range(100):
    from flux_class_vecs import flux_cone,supp
    
    import numpy as np
    import random
    from multiprocessing import Pool
    import tqdm
    
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
                start_nodes.append(random.choice(list(set(G.nodes)-set(start_nodes))))
            target_nodes = [random.choice(list(set(G.nodes)-set(start_nodes)))]
            while random.random() < hyperarc_probability:
                target_nodes.append(random.choice(list(set(G.nodes)-set(start_nodes)-set(target_nodes))))
            consume_rates = np.random.choice(possible_weights,len(start_nodes),p = weight_probabilities)
            produce_rates = np.random.choice(possible_weights,len(target_nodes),p = weight_probabilities)
            G.add_arc(arc(start_nodes,target_nodes,consume_rates,produce_rates))
            
        
        incidence = G.get_incidence()
        rev = np.random.choice([0,1],len(G.arcs))
        
        return flux_cone("Random network",incidence[~np.all(incidence==0,axis=1)],rev)
        
    if __name__ == "__main__":  
        model = generate_random_network(15,25)
        gens = model.get_geometry()[0]
        efvs = model.get_efvs("cdd")
        
        if len(gens) == 0:
            print("no EFMs")
        if len(gens) < len(efvs):
            gen_efms = [supp(gen) for gen in model.generators]
            
            non_gen_inds = []
            for i,efv in enumerate(model.efvs):
                if supp(efv) not in gen_efms:
                    non_gen_inds.append(i)
            
            efvs = model.efvs[non_gen_inds]
        
            
            if len(efvs) < 1500:
                two_gen_pairs = list(tqdm.tqdm(map(model.two_gens,efvs),total= len(efvs)))
            else:
                with Pool(8) as p:
                    two_gen_pairs = list(tqdm.tqdm(p.imap(model.two_gens,efvs),total= len(efvs)))
                    p.close()
                    
            from collections import Counter
            lens = [len(pair) for pair in two_gen_pairs]
            counter = sorted(Counter(lens).items())
            print(counter)
            if counter[0][0] == 2:
                print("No counterexample")
            else:
                print("COUNTEREXAMPLE")
