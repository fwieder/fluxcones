import numpy as np
import efmtool,cdd,cobra
from scipy.optimize import linprog

digit_tol = 10
tol = 1e-10

#######################################################################################################################
# "Helper functions" 
#######################################################################################################################


def supp(vector,tol = digit_tol):
    return(list(np.nonzero(np.round(vector,tol))[0]))

def zero(vector,tol = digit_tol):
    return(list(set(np.arange(len(vector)))-set(supp(vector,tol))))

# Return the largest absolute value of a vector
def abs_max(vector):
    if all(vector == np.zeros(len(vector))):
        return 0
    abs_max = np.max(np.absolute(vector[vector!=0]))
    return abs_max

def get_gens(stoich,rev, algo = "cdd"):
    # so far only implemented for algo == cdd
    if algo == "cdd":
        # nonegs is the matrix defining the inequalities for each irreversible reachtion
        irr = (np.ones(len(rev)) - rev).astype(int)
        nonegs = np.eye(len(rev))[np.nonzero(irr)[0]]
        
        
        # initiate Matrix for cdd
        if len(nonegs) > 0:
            mat = cdd.Matrix(nonegs,number_type = 'float')
            mat.extend(stoich,linear = True)
        else:
            mat = cdd.Matrix(stoich,linear = True)
        
        
        # generate polytope and compute generators
        poly = cdd.Polyhedron(mat)
        gens = poly.get_generators()
        
    return(gens)

def get_efvs(stoich,rev, algo = "efmtool"):
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
            if np.count_nonzero(np.round(vector,10)) > 0:
                tokeep.append(index)
        efvs = unsplit[tokeep]
        
        
        return(efvs)
    
    if algo == "efmtool":
        # initiate reaction names and metabolite names from 0 to n resp. m because 
        # efmtool needs these lists of strings as input
        # "normalize options:  [max, min, norm2, squared, none]
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
        efvs = efmtool.calculate_efms(stoich,rev,reaction_names,metabolite_names,opts)
        
        
        return(efvs.T)

#######################################################################################################################
# The actucal flux_cone class
#######################################################################################################################

class flux_cone:
    
    
    ''' initiate class object with a model path, stoichiometric matrix and a {0,1}-vector for reversible reactions '''
    
    def __init__(self, stoichiometry, reversibility,name = None):
        
        self.name = name
        
        self.stoich = stoichiometry
        
        self.rev = reversibility 
        
        self.irr = (np.ones(len(self.rev)) - self.rev).astype(int)
        
        self.nonegs = np.eye(len(self.stoich[0]))[supp(self.irr)]
     
        self.S = np.r_[self.stoich,self.nonegs]
    
    ''' create the fluxcone as flux_cone.from_sbml to use an sbml file as input '''
    
    
    @classmethod
    def from_sbml(cls,path_to_sbml):
        
        sbml_model = cobra.io.read_sbml_model(path_to_sbml)
        
        stoich = cobra.util.array.create_stoichiometric_matrix(sbml_model)
        
        rev = np.array([rea.reversibility for rea in sbml_model.reactions]).astype(int)
        
        return cls(stoich,rev,path_to_sbml)
    
    
    ''' create the fluxcone as flux_cone.from_kegg to use a path to a folder containing the stoichiometrix matrix and the vector of reversible reactions as input'''
    
    
    @classmethod
    def from_kegg(cls,path_to_kegg):
    
        
        stoich = np.genfromtxt(path_to_kegg + "_stoichiometry")
        
        rev = np.genfromtxt(path_to_kegg + "_reversibility").astype(int)
        stoichiometry = stoich
        reversibility = rev
        ext = np.genfromtxt(path_to_kegg + "_externality").astype(int)
        ind = np.nonzero(ext)[0][0]
        
        stoichiometry = np.c_[stoich[:,:ind],stoich[:,np.unique(stoich[:,ind:],axis=1,return_index=True)[1]+ind]]
            
        reversibility = np.append(rev[:ind],rev[np.unique(stoich[:,ind:],axis=1,return_index=True)[1]+ind],axis=0)
        
        return cls(stoichiometry,reversibility,path_to_kegg)

    @classmethod    
    def from_small(cls,path_to_small):
        
        stoich = np.genfromtxt(path_to_small + "_stoichiometry")
        
        rev = np.genfromtxt(path_to_small + "_reversibility").astype(int)
        stoichiometry = stoich
        reversibility = rev
        return cls(stoichiometry,reversibility,path_to_small)
#################################################################################################################################################    
# Callable methods for flux_cone objects:
#################################################################################################################################################    
 
    
    ''' compute the dimension of the lineality space of the cone '''
    
    
    def get_lin_dim(self):    
        lin_dim = len(np.nonzero(self.rev)[0]) - np.linalg.matrix_rank(self.stoich[:,np.nonzero(self.rev)[0]])
        self.lin_dim = lin_dim
        return(lin_dim)
    
    
    ''' Cone dim not working if model contains reduandancies'''
    
    def get_cone_dim(self):
        cone_dim = len(self.stoich[0]) - np.linalg.matrix_rank(self.stoich)
        return(cone_dim)
    
    ''' call is_efm with the support of a flux vector as input. The function returns True, if the support is an elementary flux mode'''
    
    
    def is_efm(self,fluxmode):
        if np.linalg.matrix_rank(self.stoich[:,fluxmode],tol=None) == len(fluxmode) - 1:
            return True
        else:
            return False
    
    
    def is_efv(self,fluxvector):
        fluxvector = np.round(fluxvector,10)
        if len(np.nonzero(fluxvector)[0]) == 0:
            return False
        
        if np.linalg.matrix_rank(self.stoich[:,np.nonzero(fluxvector)[0]],tol=None) == len(np.nonzero(fluxvector)[0]) - 1:
            return True
        
        return False
    
    
    ''' compute the EFVs of the fluxcone '''
    
    
    def get_efvs(self, algo = "efmtool"):
        if algo == "cdd":
            efvs = get_efvs(self.stoich,self.rev,algo = "cdd")
            self.efvs = efvs
            self.efms = set([tuple(supp(efv)) for efv in self.efvs])
            return efvs
    
        if algo == "efmtool":
            efvs = get_efvs(self.stoich,self.rev,algo = "efmtool")
            self.efvs = efvs
            self.efms = set([tuple(supp(efv)) for efv in self.efvs])
            return efvs
    
    def get_rev_efvs(self):
        
        
        irr_zeros = supp(self.irr)
        
        S = np.r_[self.stoich,np.eye(len(self.irr))[irr_zeros]]
        rev_efvs = get_efvs(S, self.rev)
        # transform to original shape to match reaction indices of original model
        
        self.rev_efvs = rev_efvs
        return(rev_efvs)
    
    ''' compute the MMBs of the fluxcone'''
    
    
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
    
 
    def delete_reaction(self, reaction_index):
        self.stoich = np.delete(self.stoich,reaction_index,axis = 1)
        self.rev = np.delete(self.rev, reaction_index)
        self.irr = np.delete(self.irr, reaction_index)
        self.nonegs = np.eye(len(self.stoich[0]))[supp(self.irr)]
        self.S = np.r_[self.stoich,self.nonegs]

    ''' Compute geometric properties of the flux cone'''
    
    def get_geometry(self):
        nonegs = np.eye(len(self.rev))[np.nonzero(self.irr)[0]]
        if len(nonegs) > 0:
            mat = cdd.Matrix(nonegs,number_type = 'float')
            mat.extend(self.stoich,linear = True)
        else:
            mat = cdd.Matrix(self.stoich,linear = True)
                
        poly = cdd.Polyhedron(mat)
        self.generators = poly.get_generators()
        self.adjacency = poly.get_adjacency()
        self.incidence = poly.get_incidence()
        
        return(self.generators,self.adjacency,self.incidence)
    
   
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


