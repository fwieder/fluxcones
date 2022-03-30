import numpy as np
import efmtool,cdd,cobra,tqdm,time
from util import printProgressBar
from scipy.optimize import linprog

digit_tol = 12
tol = 1e-12


#######################################################################################################################
# "Helper functions" 
#######################################################################################################################
def supp(vector):
    return(list(np.nonzero(np.round(vector,digit_tol))[0]))
def zero(vector):
    return(list(set(np.arange(len(vector)))-set(supp(vector))))


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


def get_efvs(stoich,rev, algo = "cdd"):
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
        reaction_names = list(np.arange(len(stoich[0])).astype(str))
        metabolite_names = list(np.arange(len(stoich)).astype(str))
        efvs = efmtool.calculate_efms(stoich,rev,reaction_names,metabolite_names)
        
        
        return(efvs.T)

#######################################################################################################################
# The actucal flux_cone class
#######################################################################################################################

class flux_cone:
    
    
    ''' initiate class object with a model path, stoichiometric matrix and a {0,1}-vector for reversible reactions '''
    
    
    def __init__(self, model_path, stoichiometry, reversibility,name = None):
        
        self.path = model_path
        
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
        
        name = sbml_model.name
        
        return cls(path_to_sbml,stoich,rev,name)
    
    
    ''' create the fluxcone as flux_cone.from_kegg to use a path to a folder containing the stoichiometrix matrix and the vector of reversible reactions as input'''
    
    
    @classmethod
    def from_kegg(cls,path_to_kegg):
        
        stoich = np.genfromtxt(path_to_kegg + "_stoichiometry")
        
        rev = np.genfromtxt(path_to_kegg + "_reversibility").astype(int)
        
        return cls(path_to_kegg,stoich,rev)

#################################################################################################################################################    
# Callable methods for flux_cone objects:
#################################################################################################################################################    
    
    
    ''' compute the dimension of the lineality space of the cone '''
    
    
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
    
    
    def is_efv(self,fluxvector):
        if len(np.nonzero(np.round(fluxvector,5))[0]) == 0:
            return False
        
        if np.linalg.matrix_rank(self.stoich[:,np.nonzero(fluxvector)[0]]) == len(np.nonzero(fluxvector)[0]) - 1:
            return True
        
        return False
    
    
    ''' compute the EFVs of the fluxcone '''
    
    
    def get_efvs(self, algo = "efmtool"):
        if algo == "cdd":
            efvs = get_efvs(self.stoich,self.rev,algo = "cdd")
            self.efvs = efvs
            self.efms = [supp(efv) for efv in self.efvs]
            return efvs
    
        if algo == "efmtool":
            efvs = get_efvs(self.stoich,self.rev,algo = "efmtool")
            self.efvs = efvs
            self.efms = [supp(efv) for efv in self.efvs]
            return efvs
    
    
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
    
    
    ''' compute only the fully reversible EFMs of the fluxcone'''  
    
    
    def get_frev_efvs(self):
        
        stoich = self.stoich[:,np.nonzero(self.rev)[0]]
        rev = np.ones(len(stoich[0]))
        # determine elementary flux vectors in the face cone
        res = get_efvs(stoich,rev,"cdd")
        
        
        # transform to original shape to match reaction indices of original model
        frev_efvs = np.zeros([np.shape(res)[0],np.shape(self.stoich)[1]])
        frev_efvs[:,np.nonzero(self.rev)[0]] = res
        self.frev_efvs = frev_efvs
        return(frev_efvs)
    
    
    ''' compute all EFMs in a given minimal proper face, defined by one MMB '''
    
    
    def efvs_in_mmb(self,mmb):
    
        face_indices = self.rev.copy()
        face_indices[mmb] = 1
    
        res = get_efvs(self.stoich[:,np.nonzero(face_indices)[0]],self.rev[np.nonzero(face_indices)[0]],"cdd")
    
        efvs_in_mmb = np.zeros([np.shape(res)[0],np.shape(self.stoich)[1]])
        efvs_in_mmb[:,np.nonzero(face_indices)[0]] = res
     
        efvs =[]
    
        for efv in efvs_in_mmb:
            if not set(np.nonzero(efv)[0]).issubset(set(np.nonzero(self.rev)[0])):
                efvs.append(efv)
                
        
        return(efvs)
    
    ''' compute all EFMs in all minimal proper faces '''
    
    
    def get_efvs_in_mmbs(self,mmbs = None):
        if  mmbs == None:
            mmbs = self.get_mmbs()
            
        mmb_efvs = list(tqdm.tqdm(map(self.efvs_in_mmb, mmbs),total = len(mmbs)))
        self.mmb_efvs = mmb_efvs
        return mmb_efvs



    ''' compute efvs in the relative interior of the flux cone (usualy None) '''

    def get_int_efvs(self):
        def unit(i):
            unit_i = np.zeros(np.shape(self.stoich)[1])
            unit_i[i] = 1
            return unit_i
    
        irr_inds = np.nonzero(self.irr)[0]
        bound_inds = []
        for i in irr_inds:
            if np.linalg.matrix_rank(np.r_[self.stoich,[unit(i)]]) != np.linalg.matrix_rank(self.stoich):
                bound_inds.append(i)
        bounds = -np.eye(len(self.rev))[bound_inds]
    
        int_efvs =[]
        for efv in self.efvs:
            if max(np.dot(bounds,efv)) < 0:
                int_efvs.append(efv)
        self.int_efvs = int_efvs
        return int_efvs
    
    ''' compute the dimension of the flux cone, requires mmb_efvs '''
    
    def get_cone_dim(self):
        dim = np.linalg.matrix_rank(self.generators)
        self.cone_dim = dim
        return dim
    
    ''' return Counter that counts occuriencies of numbers in the stoichiometric matrix '''
    
    def get_counter(self):
        from collections import Counter
        counter = Counter(self.stoich.reshape(1,np.shape(self.stoich)[0] * np.shape(self.stoich)[1])[0])
        self.counter = counter
        return counter
    
    ''' delete a reaction from the model '''    
    
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
        self.generators = np.round(poly.get_generators(),5)
        self.adjacency = poly.get_adjacency()
        self.incidence = poly.get_incidence()
        
        return(self.generators,self.adjacency,self.incidence)
    
   
    
    ''' split reversible reaction into a forward and a backward irreversible reaction '''
    
    def split_rev(self, index):
        
        if self.rev[index] == 0:
            print("Error: Reaction at index", index, "is not reversible.")
            return 0
        
        else:
            self.stoich = np.insert(self.stoich, index+1, -self.stoich[:,index], axis=1)
            self.rev = np.insert(self.rev,index+1,0)
            self.rev[index] = 0
            self.irr = np.insert(self.irr,index+1,1)
            self.irr[index] = 1
            self.nonegs = np.eye(len(self.stoich[0]))[supp(self.irr)]
            return 0
    
    
    
    def rev_cancellations(self,efvs):
        tol = 12
        
        rev = supp(self.rev)
        new_efvs = efvs
        new_efms = [tuple(supp(efv)) for efv in new_efvs]
        
        
        for rev_ind in rev:
            
            pos = efvs[np.where(efvs[:,rev_ind] > 1e-12)]
            neg = efvs[np.where(efvs[:,rev_ind] < -1e-12)]
            if len(pos) > 0  and len(neg) > 0:
                for pos_efv in pos:
                    for neg_efv in neg:
                        new_efv = pos_efv[rev_ind]*neg_efv - neg_efv[rev_ind]*pos_efv
                        if self.is_efv(new_efv):
                            if tuple(supp(new_efv)) not in new_efms:
                                new_efvs = np.r_[new_efvs,new_efv.reshape(1,len(new_efv))]
                                new_efms.append(tuple(np.nonzero(new_efv)[0]))
                                if set(tuple(np.nonzero(new_efv)[0])) < set(rev):
                                    new_efvs = np.r_[new_efvs,-new_efv.reshape(1,len(new_efv))]
        return(new_efvs)
    
    
    def rev_cancels(self,efvs):
        tol = 5
        rev = np.nonzero(self.rev)[0]
        

        gen_efvs = self.generators
        gen_efms = [tuple(np.nonzero(efv)[0]) for efv in gen_efvs]
        for efv in gen_efvs:
            if set(np.nonzero(efv)[0]) < set(rev):
                gen_efvs = np.r_[gen_efvs,-efv.reshape(1,len(efv))]
        
        new_efvs = efvs
        new_efms = [tuple(np.nonzero(efv)[0]) for efv in new_efvs]
        
        
        for rev_ind in rev:
            gen_pos = gen_efvs[np.where(gen_efvs[:,rev_ind] > 1e-5)]
            gen_neg = gen_efvs[np.where(gen_efvs[:,rev_ind] < -1e-5)]
            
            pos = efvs[np.where(efvs[:,rev_ind] > 1e-5)]
            neg = efvs[np.where(efvs[:,rev_ind] < -1e-5)]
            
            if len(gen_pos) > 0  and len(neg) > 0:
                for pos_efv in gen_pos:
                    for neg_efv in neg:
                        new_vec = pos_efv[rev_ind]*np.round(neg_efv,tol) - neg_efv[rev_ind]*np.round(pos_efv,tol)
                        if self.is_efv(new_vec):
                            if tuple(np.nonzero(new_vec)[0]) not in gen_efms and tuple(np.nonzero(new_vec)[0]) not in new_efms:
                                new_efvs = np.r_[new_efvs,new_vec.reshape(1,len(new_vec))]
                                new_efms.append(tuple(np.nonzero(new_vec)[0]))
                                if set(tuple(np.nonzero(new_vec)[0])) < set(rev):
                                    new_efvs = np.r_[new_efvs,-new_vec.reshape(1,len(new_vec))]
                                    
            if len(pos) > 0  and len(gen_neg) > 0:
                for pos_efv in pos:
                    for neg_efv in gen_neg:
                        new_vec = pos_efv[rev_ind]*np.round(neg_efv,tol) - neg_efv[rev_ind]*np.round(pos_efv,tol)
                        if self.is_efv(new_vec):
                            if tuple(np.nonzero(new_vec)[0]) not in gen_efms and tuple(np.nonzero(new_vec)[0]) not in new_efms:
                                new_efvs = np.r_[new_efvs,new_vec.reshape(1,len(new_vec))]
                                new_efms.append(tuple(np.nonzero(new_vec)[0]))
                                if set(tuple(np.nonzero(new_vec)[0])) < set(rev):
                                    new_efvs = np.r_[new_efvs,-new_vec.reshape(1,len(new_vec))]
        return(new_efvs)    
    
    
    def check_dim(self,vector):
        return len(vector) - np.linalg.matrix_rank(self.S[zero(np.dot(self.S,vector))])
    
    
    def irr_supp(self,vector):
        return list(np.intersect1d(supp(vector),supp(self.irr)))
    
    def rev_zeros(self,vector):
        return list(np.intersect1d(zero(vector), supp(self.rev)))
    
    
        
    def two_gens(self,vector):
        
        efm = supp(vector)
        
        candidates = self.efvs[np.where(np.all((np.round(self.efvs[:,np.setdiff1d(supp(self.irr),self.irr_supp(vector))],5) == 0), axis=1))]
        
        #gen_pairs = []
        for rev_zero_ind in self.rev_zeros(vector):
            pos = candidates[np.where(candidates[:,rev_zero_ind] > tol)]
            neg = candidates[np.where(candidates[:,rev_zero_ind] < -tol)]
            
            if len(pos) > 0  and len(neg) > 0:
                for pos_efv in pos:
                    for neg_efv in neg:
                        new_vec = pos_efv[rev_zero_ind]*neg_efv - neg_efv[rev_zero_ind]*pos_efv
                        if set(supp(new_vec)) == set(efm):
                                
                            return(pos_efv,neg_efv)
                            #gen_pairs.append((pos_efv,neg_efv))
        return([])
        #return gen_pairs
    
    def make_irr(self,index):
        self.irr[index] = 1
        self.rev[index] = 0
        
    def make_rev(self,index):
        self.rev[index] = 1
        self.irr[index] = 0
    
    def check_coupling(self,reac1,reac2):
        for efm in self.efms:
            if reac1 in efm and reac2 not in efm:
                return False
       
        return True
    
    def make_irredundant(self):
        redundants = "a"
        
        while len(redundants) > 0:
            if redundants != "a":
                self.make_rev(redundants[0])
            redundants = []
            for index in supp(self.irr):
                c = -np.eye(len(self.stoich.T))[index]
                A_ub = np.eye(len(self.stoich.T))[np.setdiff1d(supp(self.irr),index)]
                A_eq = self.stoich
                b_ub = np.zeros(len(A_ub))
                b_eq = np.zeros(len(A_eq))
                bounds = (None,None)
                if abs(linprog(c,A_ub,b_ub,A_eq,b_eq,bounds).fun) < .1:
                    redundants.append(index)

        