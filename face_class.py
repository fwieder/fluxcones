from multiprocessing import Pool
from flux_class import flux_cone,supp,get_efms,get_gens,zero,tol
import numpy as np
import tqdm,sys,time
from collections import Counter
from itertools import combinations
import cdd

class Face:
    def __init__(self, model, rep_efm):
        
        self.stoich = model.stoich
        
        self.rev = model.rev 
        
        self.irr = model.irr
        
        irr_zeros = np.setdiff1d(supp(self.irr), np.nonzero(rep_efm)[0])        
        
        self.S = np.r_[self.stoich,np.eye(len(rep_efm))[irr_zeros]]

    def get_efms(self,algo = "efmtool"):
        self.efms = get_efms(self.S,self.rev,algo)
        return self.efms
    
    def degree(self,vector):
        return len(vector) - np.linalg.matrix_rank(self.S[zero(np.dot(self.S,vector))])
   

    def irr_supp(self,vector,tol = digit_tol):
        return list(np.intersect1d(supp(vector),supp(self.irr,tol)))    
     
    def is_in(self,vec,is_in_tol = digit_tol):

        if len(vec[self.irr_supp(vec,is_in_tol)])>0:
            if min(vec[self.irr_supp(vec,is_in_tol)]) < -is_in_tol:
                return False
                #return("negative irreversible reaction")
                
        if supp(np.dot(self.S,vec),is_in_tol) == []:
            return True
        return False
        #return("S * v not equal 0")
    
    def rev_zeros(self,vector):
        return list(np.intersect1d(zero(vector), supp(self.rev)))
    
    def two_gens(self,vector):
        candidates = self.efms
        gen_pairs = []
        for rev_zero_ind in self.rev_zeros(vector):
            pos = candidates[np.where(candidates[:,rev_zero_ind] > tol)]
            neg = candidates[np.where(candidates[:,rev_zero_ind] < -tol)]
            if len(pos) > 0  and len(neg) > 0:
                for pos_efm in pos:
                    for neg_efm in neg:
                        new_vec = -neg_efm[rev_zero_ind]*pos_efm + pos_efm[rev_zero_ind]*neg_efm
                        if supp(new_vec) == supp(vector):
                            return(pos_efm,neg_efm)