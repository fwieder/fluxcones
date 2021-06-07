import numpy as np
import efmtool,cdd,cobra,tqdm,time
from util import printProgressBar

#######################################################################################################################
# "Helper functions" 
#######################################################################################################################


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


def get_efvs(model, algo = "cdd"):
    if algo == "cdd":
        # Store information about original shape to be able to revert splitting of reversible reactions later
        original_shape = np.shape(model.stoich)
        rev_indices = np.nonzero(model.rev)[0]
        
        
        # split reversible reactions by appending columns
        S_split = np.c_[model.stoich,-model.stoich[:,rev_indices]]
        
        
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
        reaction_names = list(np.arange(len(model.stoich[0])).astype(str))
        metabolite_names = list(np.arange(len(model.stoich)).astype(str))
        efvs = efmtool.calculate_efms(model.stoich,model.rev,reaction_names,metabolite_names)
        
        
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
        if np.linalg.matrix_rank(self.stoich[:,np.nonzero(fluxvector)[0]]) == len(np.nonzero(fluxvector)[0]) - 1:
            return True
        else:
            return False
    
    
    ''' compute the EFVs of the fluxcone '''
    
    
    def get_efvs(self, algo = "efmtool"):
        if algo == "cdd":
            efvs = get_efvs(self,algo = "cdd")
            self.efvs = efvs
            return efvs
    
        if algo == "efmtool":
            efvs = get_efvs(self,algo = "efmtool")
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
        # initiate temporaray model that defines the reversible metabolic space
    
        rms = type('rms', (object,), {})()
        rms.stoich = self.stoich[:,np.nonzero(self.rev)[0]]
        rms.rev = np.ones(len(rms.stoich[0]))
    
    
        # determine elementary flux vectors in the temporary cone
        res = get_efvs(rms,"cdd")
        
        
        # transform to original shape to match reaction indices of original model
        frev_efvs = np.zeros([np.shape(res)[0],np.shape(self.stoich)[1]])
        frev_efvs[:,np.nonzero(self.rev)[0]] = res
        self.frev_efvs = frev_efvs
        return(frev_efvs)
    
    
    ''' compute all EFMs in a given minimal proper face, defined by one MMB '''
    
    
    def efvs_in_mmb(self,mmb):
    
        face_indices = self.rev.copy()
        face_indices[mmb] = 1
    
        face = type('min_face', (object,), {})()
        face.stoich = self.stoich[:,np.nonzero(face_indices)[0]]
        face.rev = self.rev[np.nonzero(face_indices)[0]]
    
        res = get_efvs(face,"cdd")
    
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
        if self.get_lin_dim() == 0:
            dim = np.linalg.matrix_rank(np.array(self.mmb_efvs).reshape(np.shape(self.mmb_efvs)[0],np.shape(self.mmb_efvs)[2]))
            self.cone_dim = dim
            return dim
        else:
            gen_mat = np.array(self.mmb_efvs[0])
            for i in range(1,len(self.mmbs)):
                gen_mat = np.r_[gen_mat,np.array(self.mmb_efvs[i])]
            dim = np.linalg.matrix_rank(gen_mat)
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
        
    ''' compute adjacency of extreme rays of the cone '''
    
    def get_adjacency(self):
        nonegs = np.eye(len(self.rev))[np.nonzero(self.irr)[0]]
        if len(nonegs) > 0:
            mat = cdd.Matrix(nonegs,number_type = 'float')
            mat.extend(self.stoich,linear = True)
        else:
            mat = cdd.Matrix(self.stoich,linear = True)
                
        poly = cdd.Polyhedron(mat)
        self.adjacency = poly.get_adjacency()
        return(self.adjacency)
    
    ''' compute incidence of extreme rays of the cone '''
    
    def get_incidence(self):
        nonegs = np.eye(len(self.rev))[np.nonzero(self.irr)[0]]
        if len(nonegs) > 0:
            mat = cdd.Matrix(nonegs,number_type = 'float')
            mat.extend(self.stoich,linear = True)
        else:
            mat = cdd.Matrix(self.stoich,linear = True)
                
        poly = cdd.Polyhedron(mat)
        self.incidence = poly.get_incidence()
        return(self.incidence)

    ''' compute geerators of the cone '''
    
    def get_generators(self):
        nonegs = np.eye(len(self.rev))[np.nonzero(self.irr)[0]]
        if len(nonegs) > 0:
            mat = cdd.Matrix(nonegs,number_type = 'float')
            mat.extend(self.stoich,linear = True)
        else:
            mat = cdd.Matrix(self.stoich,linear = True)
                
        poly = cdd.Polyhedron(mat)
        self.generators = np.round(poly.get_generators(),5)
        return(self.generators)
    
    ''' compute efvs in a 2-face defined by two extreme rays '''
    
    def efvs_in_2face(self,efv0,efv1):
        irr = np.nonzero(self.irr)[0]
        efv0_inds = np.intersect1d(np.nonzero(efv0)[0],irr)
        efv1_inds = np.intersect1d(np.nonzero(efv1)[0],irr)
        face2_indices = np.union1d(efv0_inds,efv1_inds)
        
        face_indices = self.rev.copy()
        face_indices[face2_indices] = 1
        
        face = type('min_face', (object,), {})()
        face.stoich = self.stoich[:,np.nonzero(face_indices)[0]]
        face.rev = self.rev[np.nonzero(face_indices)[0]]

        res = get_efvs(face,"cdd")
       
        efvs = np.zeros([np.shape(res)[0],np.shape(self.stoich)[1]])
        efvs[:,np.nonzero(face_indices)[0]] = res
    
        return(efvs)
    
    ''' compute efvs in all 2-faces defined by pairs of adjacent extreme rays '''
    
    def get_efms_in_all_2faces(self):
        face_ind_pairs = []
        
        adj = self.get_adjacency()
        gens = self.get_generators()        
        for ind1,adj_list in enumerate(adj):
            for ind2 in adj_list:
                if sorted((ind1,ind2)) not in face_ind_pairs:
                    face_ind_pairs.append(sorted((ind1,ind2)))
                    
        
        print("determining efms in a total of" , len(face_ind_pairs), "2-faces")
        new_efms = []
    
        start = time.perf_counter()
        start_time = time.time()
        for ind,pair in enumerate(face_ind_pairs):
            printProgressBar(ind,len(face_ind_pairs),starttime = start)
            temp = self.efvs_in_2face(gens[pair[0]], gens[pair[1]])
            for efv in temp:
                new_efms.append(list(np.nonzero(np.round(efv,5))[0]))
        new_efms = np.unique(new_efms)
        end_time = time.time()
        print("")
        print(len(new_efms), "efms found in dim t+1 faces in" , end_time - start_time)
        self.face2_efms = new_efms
        return(new_efms)