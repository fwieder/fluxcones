# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:58:32 2024

@author: fred
"""
# import needed packages

import numpy as np
import efmtool,cdd,cobra
from scipy.optimize import linprog
from helper_functions import *

#######################################################################################################################
# The actucal flux_cone class
#######################################################################################################################

class flux_cone:
     
    ''' 
    initiate class object with a model path, stoichiometric matrix and a {0,1}-vector for reversible reactions
    '''
    
    def __init__(self, stoichiometry, reversibility):
        
        # stote size of stoichiometric matrix
        self.num_metabs,self.num_reacs = np.shape(stoichiometry)
        
        self.stoich = stoichiometry
        
        self.rev = reversibility 
        
        # self.irr only depends on self.rev
        self.irr = (np.ones(self.num_reacs) - self.rev).astype(int)
        
        # non-negativity constraints defined by v_irr >= 0
        nonegs = np.eye(self.num_reacs)[supp(self.irr)]
        
        # outer description of the flux cone by C = { x | Sx >= 0}
        self.S = np.r_[self.stoich,nonegs]
    
    
    ''' create the fluxcone as flux_cone.from_sbml to use an sbml file as input '''
    
    @classmethod
    def from_sbml(cls,path_to_sbml):
        
        # read sbml-file
        sbml_model = cobra.io.read_sbml_model(path_to_sbml)
        
        # extract stoichiometric matrix
        stoich = cobra.util.array.create_stoichiometric_matrix(sbml_model)
        
        # extract reversibility vector
        rev = np.array([rea.reversibility for rea in sbml_model.reactions]).astype(int)
        
        # initialize class object from extracted parameters
        return cls(stoich,rev)
    

#################################################################################################################################################    
# Callable methods for flux_cone objects:
#################################################################################################################################################    
  
    
    
    ''' compute the dimension of the lineality space of the cone '''
    
    def get_lin_dim(self):    
        lin_dim = len(supp(self.rev)) - np.linalg.matrix_rank(self.stoich[:,supp(self.rev)])
        self.lin_dim = lin_dim
        return(lin_dim)
    
    
    ''' get_cone_dim might not work if description of model contains reduandancies'''
    
    def get_cone_dim(self):
        cone_dim = self.num_reacs - np.linalg.matrix_rank(self.stoich)
        return(cone_dim)
    
    ''' test whether a given np.array is a steady-state fluxvector of the flux_cone instance'''
    
    def is_in(self,vec):
        # test whether v_irr >= 0
        if len(vec[self.irr_supp(vec,tol)]) > 0:
            if min(vec[self.irr_supp(vec,tol)]) < 0:
                print("Not in cone, because there is an irreversible reaction with negative flux")
                return False
        # test whether S*v = 0
        if all(supp(np.dot(self.stoich,vec),tol) == np.array([])):
            return True
        
        else:    
            print("S*v not equal to 0")
            return False
    
    ''' test whether a given np.array is an EFM of the flux_cone instance by applying the rank test'''
    
    def is_efm(self,vector):
        # 0 is not an EFM by defintion
        if len(supp(vector)) == 0:
            return False
        
        # rank test        
        if np.linalg.matrix_rank(self.stoich[:,supp(vector)]) == len(supp(vector)) - 1:
            return True
        
        return False
    
    
    ''' compute the EFMs of the fluxcone using cdd, efmtool or the milp approach'''
    
    def get_efms(self, algo = "efmtool"):
        if algo == "cdd":
            efms = get_efms(self.stoich,self.rev,algo = "cdd")
            self.efms = efms
            return efms
    
        if algo == "efmtool":
            efms = get_efms(self.stoich,self.rev,algo = "efmtool")
            self.efms = efms
            return efms
        
        if algo == "milp":
            efms = get_efms(self.stoich, self.rev,algo="milp")
            self.efms = efms
            return efms
    
    def get_rev_efms(self,algo = "efmtool"):
        # reversible EFMs cannot have active reversible reactions, so v_i = 0 is added to stoichiometric matrix for i in Irr
        S = np.r_[self.stoich,np.eye(self.num_reacs)[supp(self.irr)]]
        rev_efms = get_efms(S, self.rev, algo)
        
        self.rev_efms = rev_efms
        return(rev_efms)
    
    ''' fast, unproven method to compute the MMBs of the fluxcone'''
    
    def get_mmbs(self):
        
        # compute v-representation using cdd (no splitting of reversible reactions)
        res = np.array(get_gens(self.stoich,self.rev))
    
        # compute MMBs from the v-representation
        mmbs = []
        for vector in res:
            mmb = []
            for index in supp(self.irr):
                if abs(vector[index]) > 1e-5:
                    mmb.append(index)
                    if mmb != [] and mmb not in mmbs:
                        mmbs.append(mmb)
        
        self.mmbs = mmbs
        return(mmbs)
    
    ''' determine degree of a vector'''
    def degree(self,vector):
        return self.num_reacs - np.linalg.matrix_rank(self.S[zero(np.dot(self.S,vector))])
    
    ''' determine irr.supp of a vector'''
    def irr_supp(self,vector):
        return list(np.intersect1d(supp(vector),supp(self.irr,tol)))
    
    ''' determine irr.zeros of a vector'''
    def irr_zeros(self,vector):
        return list(np.intersect1d(zero(vector), supp(self.irr,tol)))
    
    ''' determine rev.supp of a vector'''
    def rev_supp(self,vector):
        return list(np.intersect1d(supp(vector),supp(self.rev,tol)))
    
    ''' determine rev.zeros of a vector'''
    def rev_zeros(self,vector):
        return list(np.intersect1d(zero(vector), supp(self.rev,tol)))
    
    ''' make a reaction irreversible '''
    def make_irr(self,index):
        self.rev[index] = 0
        self.irr[index] = 1
        
    ''' make a reaction reversible'''
    def make_rev(self,index):
        self.rev[index] = 1
        self.irr[index] = 0
    
    ''' determine irredundant desciption of the flux cone '''
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

    ''' determine indices of blocked irreversible reactions '''
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
    
    ''' determine indices of blocked reversible reactions '''
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
    
   
    ''' determine EFMs with inclusionwise smaller support than vector '''     
    def face_candidates(self,vector):
        return self.efms[np.where(np.all((np.round(self.efms[:,np.setdiff1d(supp(self.irr),self.irr_supp(vector))],10) == 0), axis=1))]
    
    
    ''' find 2 EFMs that can be positively combined to vector '''
    def two_gens(self,vector):
        
        #candidates = self.face_candidates(vector)
        candidates = self.efms
        gen_pairs = []
        for rev_zero_ind in self.rev_zeros(vector):
            pos = candidates[np.where(candidates[:,rev_zero_ind] > tol)]
            neg = candidates[np.where(candidates[:,rev_zero_ind] < -tol)]
            if len(pos) > 0  and len(neg) > 0:
                for pos_efm in pos:
                    for neg_efm in neg:
                        new_vec = -neg_efm[rev_zero_ind]*pos_efm + pos_efm[rev_zero_ind]*neg_efm
                        new_vec = pos_efm - pos_efm[rev_zero_ind]/neg_efm[rev_zero_ind]*neg_efm
                        if abs_max(new_vec - vector) < tol:
                        #if all(np.round(new_vec,5) == np.round(vector,5)):
                                
                            return(pos_efm,( - pos_efm[rev_zero_ind]/neg_efm[rev_zero_ind],neg_efm)) #,-neg_efm[rev_zero_ind],pos_efm[rev_zero_ind])
                            #gen_pairs.append(((pos_efm,self.degree(pos_efm)),(neg_efm,self.degree(neg_efm))))
                            #return gen_pairs
        
        return gen_pairs
    
    
    ''' find all pairs of 2 EFMs that can be positively combined to vector '''

    def all_two_gens(self,vector):
        candidates = self.face_candidates(vector)
        #candidates = self.efms
        gen_pairs = []
        for rev_zero_ind in self.rev_zeros(vector):
            pos = candidates[np.where(candidates[:,rev_zero_ind] > tol)]
            neg = candidates[np.where(candidates[:,rev_zero_ind] < -tol)]
            if len(pos) > 0  and len(neg) > 0:
                for pos_efm in pos:
                    for neg_efm in neg:
                        new_vec = pos_efm + pos_efm[rev_zero_ind]/neg_efm[rev_zero_ind]*neg_efm
                        if supp(new_vec) == supp(vector):
                            gen_pairs.append((pos_efm,neg_efm))
                            
        return gen_pairs
    
    
    ''' determine Face of the flux cone that contains vector '''
    def face_defined_by(self,rep_vector):
        return flux_cone.face(self,rep_vector)
        
    
    class face:
        def __init__(self,flux_cone_instance,rep_vector):
            
            self.rev = flux_cone_instance.rev 
            
            self.irr = flux_cone_instance.irr
            
            # irr_zeros are the indices of the irreversibility constraints 
            # that are fulfilled with equality by rep_vector
            # and these define the facets rep_vector is contained in.
            # numerical inaccuracies are assumed to be removed when the face it is contained in is determined.
            irr_zeros = np.setdiff1d(supp(self.irr), np.nonzero(rep_vector)[0])
            
            self.stoich = np.r_[flux_cone_instance.stoich,np.eye(len(rep_vector))[irr_zeros]]
            
        def get_efms(self, algo = "efmtool"):
            if algo == "cdd":
                efms = get_efms(self.stoich,self.rev,algo = "cdd")
                self.efms = efms
                return efms
        
            if algo == "efmtool":
                efms = get_efms(self.stoich,self.rev,algo = "efmtool")
                self.efms = efms
                return efms
        
        def rev_zeros(self,vector):
            return list(np.intersect1d(zero(vector), supp(self.rev,tol)))
        
        def two_gens(self,vector):
            
            #candidates = self.face_candidates(vector)
            candidates = self.efms
            for rev_zero_ind in self.rev_zeros(vector):
                pos = candidates[np.where(candidates[:,rev_zero_ind] > tol)]
                neg = candidates[np.where(candidates[:,rev_zero_ind] < -tol)]
                if len(pos) > 0  and len(neg) > 0:
                    for pos_efm in pos:
                        for neg_efm in neg:
                            new_vec = -neg_efm[rev_zero_ind]*pos_efm + pos_efm[rev_zero_ind]*neg_efm
                            new_vec = pos_efm - pos_efm[rev_zero_ind]/neg_efm[rev_zero_ind]*neg_efm
                            if abs_max(new_vec - vector) < tol:
                                return(pos_efm,( - pos_efm[rev_zero_ind]/neg_efm[rev_zero_ind],neg_efm)) #,-neg_efm[rev_zero_ind],pos_efm[rev_zero_ind])
                            
        def is_in(self,vec,is_in_tol = tol):
            if len(vec[self.irr_supp(vec,is_in_tol)])>0:
                if min(vec[self.irr_supp(vec,is_in_tol)]) < tol:
                    print("Not in cone, because there is an irreversible reaction with negative flux")
                    return False
                
            if all(supp(np.dot(self.stoich,vec),is_in_tol) == np.array([])):
                return True
            else:
                
                print("S*v not equal to 0")
                return False