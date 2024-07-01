from multiprocessing import Pool
from flux_class import flux_cone,supp,get_efvs,get_gens,zero,tol
import numpy as np
import tqdm,sys,time
from collections import Counter
from itertools import combinations
import cdd
import cobra
digit_tol = 7
tol = 10**(-digit_tol)

class Face:
    def __init__(self, model, rep_efv):
        
        self.stoich = model.stoich
        
        self.rev = model.rev 
        
        self.irr = model.irr
        
        irr_zeros = np.setdiff1d(supp(self.irr), np.nonzero(rep_efv)[0])        
        
        self.S = np.r_[self.stoich,np.eye(len(rep_efv))[irr_zeros]]

    def get_efvs(self,algo = "efmtool"):
        self.efvs = get_efvs(self.S,self.rev,algo)
        return self.efvs
    
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
        candidates = self.efvs
        gen_pairs = []
        for rev_zero_ind in self.rev_zeros(vector):
            pos = candidates[np.where(candidates[:,rev_zero_ind] > tol)]
            neg = candidates[np.where(candidates[:,rev_zero_ind] < -tol)]
            if len(pos) > 0  and len(neg) > 0:
                for pos_efv in pos:
                    for neg_efv in neg:
                        new_vec = -neg_efv[rev_zero_ind]*pos_efv + pos_efv[rev_zero_ind]*neg_efv
                        if supp(new_vec) == supp(vector):
                            return(pos_efv,neg_efv)