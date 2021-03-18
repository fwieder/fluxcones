# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:11:00 2021

@author: wiede
"""
import numpy as np
import cobra,tqdm,time
from functions import get_gens,get_efvs
from multiprocessing import Pool

class flux_cone:
    ''' initiate class object with a model path, stoichiometric matrix and a {0,1}-vector for reversible reactions '''
    def __init__(self, model_path, stoichiometry, reversibility):
        
        self.path = model_path
        
        self.stoich = stoichiometry
        
        self.rev = reversibility 
        
        self.irr = (np.ones(len(self.rev)) - self.rev).astype(int)
        
    
    ''' create the fluxcone as flux_cone.from_sbml to use an sbml file as input '''
    @classmethod
    def from_sbml(cls,path_to_sbml):
        
        sbml_model = cobra.io.read_sbml_model(path_to_sbml)
        
        stoich = cobra.util.array.create_stoichiometric_matrix(sbml_model)
        
        rev = np.array([rea.reversibility for rea in sbml_model.reactions]).astype(int)
        
        return cls(path_to_sbml,stoich,rev)
    
    ''' create the fluxcone as flux_cone.from_kegg to use a path to a folder containing the stoichiometrix matrix and the vector of reversible reactions as input'''
    @classmethod
    def from_kegg(cls,path_to_kegg):
        
        stoich = np.genfromtxt(path_to_kegg + "_stoichiometry.txt")
        
        rev = np.genfromtxt(path_to_kegg + "_reversibility.txt")
        
        return cls(path_to_kegg,stoich,rev)

#################################################################################################################################################    
    # Callable methods for flux_cone objects:
#################################################################################################################################################    
    
    def get_lin_dim(self):    
        lin_dim = len(np.nonzero(self.rev)[0]) - np.linalg.matrix_rank(self.stoich[:,np.nonzero(self.rev)[0]])
        self.lin_dim = lin_dim
        return(lin_dim)
    
    
    ''' call is_efm with the support of a flux vector as input. The function returns True, if the support is an elementary flux mode'''
    
    
    def is_efm(self,fluxmode):
        if np.linalg.matrix_rank(self.stoich[:,fluxmode]) == len(fluxmode) - 1:
            return True
        else:
            return False
    ''' calculate the mmbs of the fluxcone '''
    
    
    def get_mmbs(self):
        
        # list of indices of irreversible reactions
        irr_indices = np.nonzero(np.ones(len(self.rev)) - self.rev)[0]
    
        # compute v-representation using cdd (no splitting of reversible reactions)
        res = np.array(get_gens(self.stoich,self.rev))
    
        # compute MMBs from the v-representation
        mmbs = []
        for vector in res:
            mmb = []
            for index in irr_indices:
                if abs(vector[index]) > 1e-5:
                    mmb.append(index)
                    if mmb != [] and mmb not in mmbs:
                        mmbs.append(mmb)
        
        self.mmbs = mmbs
        return(mmbs)
    ''' calculate the fully reversible efms of the fluxcone'''  
    
    
    def get_frev_efms(self):
    # initiate temporaray model that defines the reversible metabolic space
    
        rms = type('rms', (object,), {})()
        rms.stoich = self.stoich[:,np.nonzero(self.rev)[0]]
        rms.rev = np.ones(len(rms.stoich[0]))
    
    
        # determine elementary flux vectors in the temporary cone
        res = get_efvs(rms,"cdd")
        
        
        # transform to original shape to match reaction indices of original model
        frev_efvs = np.zeros([np.shape(res)[0],np.shape(self.stoich)[1]])
        frev_efvs[:,np.nonzero(self.rev)[0]] = res
        
    
        # transfrom to modes by determining supports
        frev_efms = []
        for efv in frev_efvs:
            efm = list(np.nonzero(np.round(efv,5))[0])
            if efm not in frev_efms:
                frev_efms.append(efm)
        
        frev_efms.sort()
        
        self.frev_efms = frev_efms
        return(frev_efms)
    
    ''' calculate all efms i a given mmb '''
    def efms_in_mmb(self,mmb):
    
        face_indices = self.rev.copy()
        face_indices[mmb] = 1
    
        face = type('min_face', (object,), {})()
        face.stoich = self.stoich[:,np.nonzero(face_indices)[0]]
        face.rev = self.rev[np.nonzero(face_indices)[0]]
    
        res = get_efvs(face,"cdd")
    
        efvs_in_mmb = np.zeros([np.shape(res)[0],np.shape(self.stoich)[1]])
        efvs_in_mmb[:,np.nonzero(face_indices)[0]] = res
    
    
        efms = [list(np.nonzero(np.round(efvs_in_mmb[i],5))[0]) for i in range(len(efvs_in_mmb))]
   
        efms_in_mmb =[]
    
        for efm in efms:
            if not set(efm).issubset(set(np.nonzero(self.rev)[0])):
                efms_in_mmb.append(efm)
                
        efms_in_mmb.sort()
        
        return(efms_in_mmb)
    
    ''' calculate all efms in all mmbs '''
    def get_efms_in_mmbs(self,mmbs = None, proces = 12):
        if  mmbs == None:
            mmb_start_time = time.time()
            mmbs = self.get_mmbs()
            mmb_comp_time = time.time() - mmb_start_time
            #print(len(mmbs), "MMBs calculated in %3dm %2ds" % (mmb_comp_time//60,mmb_comp_time%60))
        mmb_efms = list(map(self.efms_in_mmb, mmbs))
        '''
        with Pool(proces) as p:
            mmb_efms = list(tqdm.tqdm(p.imap(self.efms_in_mmb,mmbs,proces), total = len(mmbs)))
        p.close()
        '''
        self.mmb_efms = mmb_efms
        return mmb_efms
        
        
if __name__ == '__main__':
    
    #model1 = flux_cone.from_sbml("./Biomodels/bigg_models/e_coli_core.xml")
    model = flux_cone.from_kegg("./Biomodels/small_examples/covert/covert")

    model.get_efms_in_mmbs()
    
    lens = [len(model.mmb_efms[i]) for i in range(len(model.mmb_efms))]
    print(lens)
    
    indices = np.arange(len(model.rev))
    from itertools import combinations
    dist=[]
    counter = 0
    for i in range(len(model.rev)):
        for inds in combinations(indices,i):
            rev = np.zeros(len(model.rev))
            rev[list(inds)] = 1
            model.rev = rev
        
            model.get_efms_in_mmbs()
            lens = list(np.unique([len(model.mmb_efms[i]) for i in range(len(model.mmb_efms))]))
        
            if lens not in dist:
                dist.append(lens)
            counter +=1
            if counter%1000 ==0:
                    print(counter,"done")
    dist.sort()
    print(dist)
    