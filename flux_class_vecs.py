import numpy as np
import efmtool,cdd,cobra,tqdm,time
from util import printProgressBar

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
            if np.count_nonzero(np.round(vector,5)) > 0:
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
            return efvs
    
        if algo == "efmtool":
            efvs = get_efvs(self.stoich,self.rev,algo = "efmtool")
            self.efvs = efvs
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
        dim1 = np.linalg.matrix_rank(self.get_efvs("efmtool"))
        dim2 = np.linalg.matrix_rank(self.get_geometry()[0])
        self.cone_dim = dim1
        return dim1,dim2
    
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
    
    ''' compute efvs in a 2-face defined by two extreme rays '''
    
    def efvs_in_2face(self,efv0,efv1):
        irr = np.nonzero(self.irr)[0]
        efv0_inds = np.intersect1d(np.nonzero(efv0)[0],irr)
        efv1_inds = np.intersect1d(np.nonzero(efv1)[0],irr)
        face2_indices = np.union1d(efv0_inds,efv1_inds)
        
        face_indices = self.rev.copy()
        face_indices[face2_indices] = 1

        res = get_efvs(self.stoich[:,np.nonzero(face_indices)[0]],self.rev[np.nonzero(face_indices)[0]],"cdd")
       
        efvs = np.zeros([np.shape(res)[0],np.shape(self.stoich)[1]])
        efvs[:,np.nonzero(face_indices)[0]] = res
    
        return(efvs)
    
    ''' compute efvs in a simplicial 3-face defined by three extreme rays '''
    
    def efvs_in_3face(self,efv0,efv1,efv2):
        irr = np.nonzero(self.irr)[0]
        efv0_inds = np.intersect1d(np.nonzero(efv0)[0],irr)
        efv1_inds = np.intersect1d(np.nonzero(efv1)[0],irr)
        efv2_inds = np.intersect1d(np.nonzero(efv2)[0],irr)
        face3_indices = np.union1d(np.union1d(efv0_inds,efv1_inds),efv2_inds)
        
        face_indices = self.rev.copy()
        face_indices[face3_indices] = 1
        
        res = get_efvs(self.stoich[:,np.nonzero(face_indices)[0]],self.rev[np.nonzero(face_indices)[0]],"cdd")
       
        efvs = np.zeros([np.shape(res)[0],np.shape(self.stoich)[1]])
        efvs[:,np.nonzero(face_indices)[0]] = res
        return(efvs)
    
    ''' compute efvs in all 2-faces defined by pairs of adjacent extreme rays '''
    
    def get_efms_in_all_2faces(self):
        face_ind_pairs = []
        
        adj = self.adjacency
        gens = self.generators        
        for ind1,adj_list in enumerate(adj):
            for ind2 in adj_list:
                if sorted((ind1,ind2)) not in face_ind_pairs:
                    face_ind_pairs.append(sorted((ind1,ind2)))
                    
        
        print("determining efms in a total of" , len(face_ind_pairs), "2-faces")
        new_efms = []
                
        start = time.perf_counter()
        
        for ind,pair in enumerate(face_ind_pairs):
            
            temp = self.efvs_in_2face(gens[pair[0]], gens[pair[1]])
            for efv in temp:
                new_efms.append(list(np.nonzero(np.round(efv,5))[0]))
            printProgressBar(ind,len(face_ind_pairs),starttime = start)
        new_efms = np.unique(new_efms)
        
        self.face2_efms = new_efms
        return(new_efms)
    
    ''' compute efvs in all simplicial 3-faces defined by triplets of adjacent extreme rays '''
    
    def face3_cancellations(self):
        face_ind_triplets = []
        
        adj = self.adjacency
        gens = self.generators
      
        start = time.perf_counter()
        print("Computing triplets for simplicial 3-faces")
        print("")
        for ind1,adj_list in enumerate(adj):
            printProgressBar(ind1,len(gens),starttime = start)        
            for ind2 in adj_list:
                for ind3 in adj[ind2]:
                    if ind3 != ind1:
                        face_ind_triplets.append(sorted((ind1,ind2,ind3)))
        '''   
            for ind2 in random.sample(adj_list,len(adj_list)):
                for ind3 in random.sample(adj[ind2], len(adj[ind2])):
                    if ind3 != ind1:
                        face_ind_triplets.append(sorted((ind1,ind2,ind3)))
        '''            
        face_ind_triplets = np.unique(face_ind_triplets,axis=0)
        print("")
        face3_cancel_efvs = list(tqdm.tqdm(map(self.rev_cancels,gens[face_ind_triplets]),total = len(face_ind_triplets)))
        
        return face3_cancel_efvs
    
    
    
    
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
            return 0
    
    
    def face2_cancellations(self):
        
        tol = 12
        adj = self.adjacency
        gens = self.generators
        
        rev = supp(self.rev)
        print("Computing pairs for 2-faces...")
        pairs = []
        
        for ind1,adj_list in enumerate(adj):
            for ind2 in adj_list:
                if sorted((ind1,ind2)) not in pairs:
                    pairs.append(sorted((ind1,ind2)))
        pairlen = len(pairs)
        
        
        new_efvs = []
        new_efms = []
        print("determining efvs in a total of" , pairlen, "2-faces by looking for cancellations of reversible reactions")
        start = time.perf_counter()
        
        for index,pair in enumerate(pairs):
            if index % 100 == 0:
                printProgressBar(index,pairlen,starttime = start)
            vec1 = gens[pair[0]][rev]
            vec2 = gens[pair[1]][rev]

            no_dubs = []
            for ind in range(len(vec1)):
                if (vec1[ind] > 1e-12 and vec2[ind] < -1e-12) or (vec1[ind] < -1e-12 and vec2[ind] > 1e-12):
                    if (vec1[ind],vec2[ind]) not in no_dubs:
                        no_dubs.append((vec1[ind],vec2[ind]))
                        new_vec = abs(vec2[ind])*gens[pair[0]] + abs(vec1[ind])*gens[pair[1]]
                        if self.is_efv(new_vec):
                            if supp(new_vec) not in new_efms:
                                new_efms.append(supp(new_vec))
                                new_efvs.append(new_vec)
                        #else:
                         #   print("problem here" , gens[pair[0]],gens[pair[1]])
        self.face2_efvs = new_efvs
        return new_efvs
        '''        
        new_efms = [list(supp(efv)) for efv in new_efvs]
        new_efms = list(np.unique(new_efms))
        
        self.face2_cancel_efms = new_efms
        return new_efms
        '''
    
    def rev_cancellations(self,efvs):
        tol = 6
        
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
    
    def check(self,vector):
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
        