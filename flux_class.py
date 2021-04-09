import numpy as np
import efmtool,cdd,cobra,tqdm


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
        
        
    ''' compute the EFMs of the fluxcone '''
    
    
    def get_efms(self, algo = "cdd"):
        if algo == "cdd":
            efvs = get_efvs(self,algo = "cdd")
            efms = [np.nonzero(efv)[0] for efv in efvs]
            self.efms = efms
            return efms
    
        if algo == "efmtool":
            efvs = get_efvs(self,algo = "efmtool")
            efms = [np.nonzero(efv)[0] for efv in efvs]
            self.efms = efms
            return efms
    
    
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
    
    
    ''' compute all EFMs in a given minimal proper face, defined by one MMB '''
    
    
    def efms_in_mmb(self,mmb):
    
        face_indices = self.rev.copy()
        face_indices[mmb] = 1
    
        face = type('min_face', (object,), {})()
        face.stoich = self.stoich[:,np.nonzero(face_indices)[0]]
        face.rev = self.rev[np.nonzero(face_indices)[0]]
    
        res = get_efvs(face,"efmtool")
    
        efvs_in_mmb = np.zeros([np.shape(res)[0],np.shape(self.stoich)[1]])
        efvs_in_mmb[:,np.nonzero(face_indices)[0]] = res
    
    
        efms = [list(np.nonzero(np.round(efvs_in_mmb[i],5))[0]) for i in range(len(efvs_in_mmb))]
   
        efms_in_mmb =[]
    
        for efm in efms:
            if not set(efm).issubset(set(np.nonzero(self.rev)[0])):
                efms_in_mmb.append(efm)
                
        efms_in_mmb.sort()
        
        return(efms_in_mmb)
    
    
    ''' compute all EFMs in all minimal proper faces '''
    
    
    def get_efms_in_mmbs(self,mmbs = None):
        if  mmbs == None:
            mmbs = self.get_mmbs()
            
        mmb_efms = list(tqdm.tqdm(map(self.efms_in_mmb, mmbs),total = len(mmbs)))
        self.mmb_efms = mmb_efms
        return mmb_efms
