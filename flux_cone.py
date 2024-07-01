# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:58:32 2024

@author: fred
"""
# import needed packages

import numpy as np
import efmtool,cdd,cobra
from scipy.optimize import linprog

# set tolerance for zero comparision, change as needed

tolerance = 1e-10


#######################################################################################################################
# "Helper functions" 
#######################################################################################################################


# Support function returns a np.array containing the indices of all entries of a vector larger than the tolerance
def supp(vector,tol = tolerance):
    return np.where(abs(vector) > tol)[0]

# Zero function returns a np.array containing the indices of all entries of a vector smaller than the tolerance
def zero(vector,tol = tolerance):
    return np.where(abs(vector) < tol)[0]


import sys
sys.exit()

# get_efms is a wrapper for efmtool and CDD to compute EFMs,
# INPUT np.array stoich: stoichiometric matrix,
# np.array rev: (0,1) vector for reversibility of reactions
# returns np.array that contains EFMs as rows

def get_efms(stoich,rev, algo = "efmtool"):
    if algo == "cdd":
        # Store information about original shape to be able to revert splitting of reversible reactions later
        original_shape = np.shape(stoich)
        rev_indices = np.nonzero(rev)[0]
        
        
        # split reversible reactions by appending columns
        S_split = np.c_[stoich,-stoich[:,rev_indices]]
        
        
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
        
        
        # remove spurious cycles
        for index,vector in enumerate(unsplit):
            if len(supp(vector)) > 0:
                tokeep.append(index)
        efms = unsplit[tokeep]
        
        
        return(efms)
    
    if algo == "efmtool":
        
        ''' 
        initiate reaction names and metabolite names from 0 to n resp. m because 
        efmtool needs these lists of strings as input
        "normalize options:  [max, min, norm2, squared, none] 
        '''
        
        opts = dict({
        "kind": "stoichiometry",
        "arithmetic": "double",
        "zero": "1e-10",
        "compression": "default",
        "log": "console",
        "level": "OFF",
        "maxthreads": "-1",
        "normalize": "max",
        "adjacency-method": "pattern-tree-minzero",
        "rowordering": "MostZerosOrAbsLexMin"
        })
        
        reaction_names = list(np.arange(len(stoich[0])).astype(str))
        metabolite_names = list(np.arange(len(stoich)).astype(str))
        efms = efmtool.calculate_efms(stoich,rev,reaction_names,metabolite_names,opts)
        
        
        return(efms.T)
    
    

#######################################################################################################################
# The actucal flux_cone class
#######################################################################################################################

     
    ''' 
    initiate class object with a model path, stoichiometric matrix and a {0,1}-vector for reversible reactions
    '''
    
    def __init__(self, stoichiometry, reversibility,name = None):
        
        self.name = name
        
        self.stoich = stoichiometry
        
        self.rev = reversibility 
        
        self.irr = (np.ones(len(self.rev)) - self.rev).astype(int)
        
        self.nonegs = np.eye(len(self.stoich[0]))[supp(self.irr)]
     
        self.S = np.r_[self.stoich,self.nonegs]
    
    
    ''' create the fluxcone as flux_cone.from_sbml to use an sbml file as input '''
    
    @classmethod
    def from_sbml(cls,path_to_sbml,name = None):
        
        sbml_model = cobra.io.read_sbml_model(path_to_sbml)
        
        stoich = cobra.util.array.create_stoichiometric_matrix(sbml_model)
        
        rev = np.array([rea.reversibility for rea in sbml_model.reactions]).astype(int)
        
        return cls(stoich,rev,name)
    

#################################################################################################################################################    
# Callable methods for flux_cone objects:
#################################################################################################################################################    
  
    
    
    ''' compute the dimension of the lineality space of the cone '''
    
    def get_lin_dim(self):    
        lin_dim = len(supp(self.rev)) - np.linalg.matrix_rank(self.stoich[:,supp(self.rev)])
        self.lin_dim = lin_dim
        return(lin_dim)
    
    
    ''' Cone dim not working if model contains reduandancies'''
    
    def get_cone_dim(self):
        cone_dim = len(self.stoich[0]) - np.linalg.matrix_rank(self.stoich)
        return(cone_dim)
    
    
    ''' is_efm tests whether a given np.array is an EFM of our flux_cone object'''
    
    def is_efm(self,vector):
        # 0 is not an EFM
        if len(supp(vector)) == 0:
            return False
        # rank test        
        if np.linalg.matrix_rank(self.stoich[:,supp(vector)]) == len(supp(vector)) - 1:
            return True
        
        return False
    
    
    ''' compute the EFMs of the fluxcone '''
    
    
    def get_efms(self, algo = "efmtool"):
        if algo == "cdd":
            efms = get_efms(self.stoich,self.rev,algo = "cdd")
            self.efs = efms
            return efms
    
        if algo == "efmtool":
            efms = get_efms(self.stoich,self.rev,algo = "efmtool")
            self.efms = efms
            return efms
    
    def get_rev_efms(self):
        
        irr_zeros = supp(self.irr)
        
        S = np.r_[self.stoich,np.eye(len(self.irr))[irr_zeros]]
        rev_efms = get_efms(S, self.rev)
        # transform to original shape to match reaction indices of original model
        
        self.rev_efms = rev_efms
        return(rev_efms)
    
    ''' compute the MMBs of the fluxcone'''
    
    def get_mmbs(self):
        
        # compute v-representation using cdd (no splitting of reversible reactions)
        res = np.array(get_gens(self.stoich,self.rev))
    
        # compute MMBs from the v-representation
        mmbs = []
        for vector in res:
            mmb = []
            for index in self.irr:
                if abs(vector[index]) > 1e-5:
                    mmb.append(index)
                    if mmb != [] and mmb not in mmbs:
                        mmbs.append(mmb)
        
        self.mmbs = mmbs
        return(mmbs)
    
    def degree(self,vector):
        return len(vector) - np.linalg.matrix_rank(self.S[zero(np.dot(self.S,vector))])
    
    
    def irr_supp(self,vector,zero_tol = digit_tol):
        return list(np.intersect1d(supp(vector),supp(self.irr,zero_tol)))
    
    def rev_supp(self,vector):
        return list(np.intersect1d(supp(vector),supp(self.rev)))
    
    def make_irr(self,index):
        self.irr[index] = 1
        self.rev[index] = 0
        
    def make_rev(self,index):
        self.rev[index] = 1
        self.irr[index] = 0
    
    def make_irredundant(self):
        redundants = "a"
        
        while len(redundants) > 0:
            irr = supp(self.irr)
            import random
            random.shuffle(irr)
            if redundants != "a":
                self.make_rev(redundants[0])
            redundants = []
            for index in irr:
                c = -np.eye(len(self.stoich.T))[index]
                A_ub = np.eye(len(self.stoich.T))[np.setdiff1d(supp(self.irr),index)]
                A_eq = self.stoich
                b_ub = np.zeros(len(A_ub))
                b_eq = np.zeros(len(A_eq))
                bounds = (None,None)
                if abs(linprog(c,A_ub,b_ub,A_eq,b_eq,bounds).fun) < .1:
                    redundants.append(index)

    
    def blocked_irr_reactions(self):
        blocked = []
        for index in supp(self.irr):
            c = -np.eye(len(self.stoich.T))[index]
            A_ub = np.eye(len(self.stoich.T))[supp(self.irr)]
            A_eq = self.stoich
            b_ub = np.zeros(len(A_ub))
            b_eq = np.zeros(len(A_eq))
            bounds = (None,None)
            if abs(linprog(c,A_ub,b_ub,A_eq,b_eq,bounds).fun) < .001 and abs(linprog(-c,A_ub,b_ub,A_eq,b_eq,bounds).fun) < .001:
                    blocked.append(index)
        blocked.reverse()
        return(blocked)
    
    def blocked_rev_reactions(self):
        blocked = []
        for index in supp(self.rev):
            c = -np.eye(len(self.stoich.T))[index]
            A_ub = np.eye(len(self.stoich.T))[supp(self.irr)]
            A_eq = self.stoich
            b_ub = np.zeros(len(A_ub))
            b_eq = np.zeros(len(A_eq))
            bounds = (None,None)
            if abs(linprog(c,A_ub,b_ub,A_eq,b_eq,bounds).fun) < .001 and abs(linprog(-c,A_ub,b_ub,A_eq,b_eq,bounds).fun) < .001:
                    blocked.append(index)
        blocked.reverse()
        return(blocked)
    
    def is_in(self,vec,is_in_tol = digit_tol):
        if len(vec[self.irr_supp(vec,is_in_tol)])>0:
            if min(vec[self.irr_supp(vec,is_in_tol)]) < -1e-6:
                #return False
                print("negative irreversible reaction")
                return False
            
        if supp(np.dot(self.stoich,vec),is_in_tol) == []:
            return True
        print("S * v not equal 0")
        return False
        
    def face_candidates(self,vector):
        return self.efvs[np.where(np.all((np.round(self.efvs[:,np.setdiff1d(supp(self.irr),self.irr_supp(vector))],10) == 0), axis=1))]
    
    def two_gens(self,vector):
        
        #candidates = self.face_candidates(vector)
        candidates = self.efvs
        gen_pairs = []
        for rev_zero_ind in self.rev_zeros(vector):
            pos = candidates[np.where(candidates[:,rev_zero_ind] > tol)]
            neg = candidates[np.where(candidates[:,rev_zero_ind] < -tol)]
            if len(pos) > 0  and len(neg) > 0:
                for pos_efv in pos:
                    for neg_efv in neg:
                        new_vec = -neg_efv[rev_zero_ind]*pos_efv + pos_efv[rev_zero_ind]*neg_efv
                        new_vec = pos_efv - pos_efv[rev_zero_ind]/neg_efv[rev_zero_ind]*neg_efv
                        if abs_max(new_vec - vector) < tol:
                        #if all(np.round(new_vec,5) == np.round(vector,5)):
                                
                            return(pos_efv,( - pos_efv[rev_zero_ind]/neg_efv[rev_zero_ind],neg_efv)) #,-neg_efv[rev_zero_ind],pos_efv[rev_zero_ind])
                            #gen_pairs.append(((pos_efv,self.degree(pos_efv)),(neg_efv,self.degree(neg_efv))))
                            #return gen_pairs
        
        return gen_pairs
    
    def all_two_gens(self,vector):
        candidates = self.face_candidates(vector)
        #candidates = self.efvs
        gen_pairs = []
        for rev_zero_ind in self.rev_zeros(vector):
            pos = candidates[np.where(candidates[:,rev_zero_ind] > tol)]
            neg = candidates[np.where(candidates[:,rev_zero_ind] < -tol)]
            if len(pos) > 0  and len(neg) > 0:
                for pos_efv in pos:
                    for neg_efv in neg:
                        new_vec = pos_efv + pos_efv[rev_zero_ind]/neg_efv[rev_zero_ind]*neg_efv
                        if supp(new_vec) == supp(vector):
                            gen_pairs.append((pos_efv,neg_efv))
                            
        return gen_pairs
    
    
    def rev_zeros(self,vector):
        return list(np.intersect1d(zero(vector), supp(self.rev)))
    
    def change_direction(self,irr_index):
        self.stoich[:,irr_index] = -self.stoich[:,irr_index]