from multiprocessing import Pool
from flux_class_vecs import flux_cone,supp,get_efvs,get_gens
import numpy as np
import tqdm,sys,time
from collections import Counter
from itertools import combinations

model = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")
model.delete_reaction(12)
model.make_irredundant()

#model.get_geometry()


'''
for i in model.adjacency[1]:
    face_efvs = model.rev_cancellations(np.r_[model.generators[0],model.generators[i]].reshape(2,len(model.generators[0])))
    if len(face_efvs) > 0:
        print(model.irr_supp(face_efvs[0]),model.check_dim(face_efvs[0]),model.irr_supp(face_efvs[1]),model.check_dim(face_efvs[1]))        
'''

def efvs_in_same_face(efv):
    S = np.r_[model.stoich,np.eye(len(efv))[np.setdiff1d(supp(model.irr),model.irr_supp(efv))]]
    return get_efvs(S,model.rev)


class face:
    def __init__(self, model, rep_efv):
        
        self.stoich = model.stoich
        
        self.rev = model.rev 
        
        self.irr = model.irr
        
        self.S = np.r_[self.stoich,np.eye(len(rep_efv))[np.setdiff1d(supp(self.irr),self.irr_supp(rep_efv))]]

    def get_efvs(self):
        self.efvs = get_efvs(self.S,self.rev)
    
    def check_dim(self,vector):
        return len(vector) - np.linalg.matrix_rank(self.S[zero(np.dot(self.S,vector))])
   
if __name__ == "__main__":
    print(model.name)
    model.efvs = np.load("./e_coli_no_bio_efvs.npy")
    model.generators = np.load("./e_coli_no_bio_gens.npy")
    #model.efvs = np.load("./e_coli_efvs.npy")
    #model.generators = np.load("e_coli_gens.npy")
    #model.get_geometry()
    
    face1_efv_inds = []
    face2_efv_inds = []
    face3_efv_inds = []
    face4_efv_inds = []
    face5_efv_inds = []
    face6_efv_inds = []
    
    
    for index,efv in enumerate(model.efvs):
        dim = model.check_dim(efv)
        if dim == 1:
            face1_efv_inds.append(index)
        if dim == 2:
            face2_efv_inds.append(index)
        if dim == 3:
            face3_efv_inds.append(index)
        if dim == 4:
            face4_efv_inds.append(index)
        if dim == 5:
            face5_efv_inds.append(index)
        if dim == 6:
            face6_efv_inds.append(index)
        
    face1_efvs = model.efvs[face1_efv_inds]
    face2_efvs = model.efvs[face2_efv_inds]
    face3_efvs = model.efvs[face3_efv_inds]
    face4_efvs = model.efvs[face4_efv_inds]
    face5_efvs = model.efvs[face5_efv_inds]
    face6_efvs = model.efvs[face6_efv_inds]
    
    
    print(len(face1_efvs),len(face2_efvs),len(face3_efvs),len(face4_efvs),len(face5_efvs),len(face6_efvs))
    
    
    more_than_two = []
    counter = 0
    counter2 = 0
    probs = []
    for candidate_ind,candidate in enumerate(face4_efvs):
        counter2 +=1
        x = efvs_in_same_face(candidate)
        x_dims = [model.check_dim(efv) for efv in x]
        if x_dims.count(4) == 2:
            more_than_two.append(x_dims.count(4))
            counter+=1
            in_a_3face = x[np.where(np.array(x_dims) == 4)[0]]
            if model.irr_supp(in_a_3face[0]) != model.irr_supp(in_a_3face[1]):
                
                print("here")
                print(candidate_ind)
                probs.append(candidate_ind)
        if x_dims.count(4) == 3:
            more_than_two.append(x_dims.count(4))
            counter+=1
            in_a_3face = x[np.where(np.array(x_dims) == 4)[0]]
            if model.irr_supp(in_a_3face[0]) != model.irr_supp(in_a_3face[1]) or model.irr_supp(in_a_3face[0]) != model.irr_supp(in_a_3face[2]):
                print("here")
                print(candidate_ind)
                probs.append(candidate_ind)
        
        if counter2 % 100 == 0:
            print(counter2)
    
    print(" ")
    print(counter)
    
    
    sys.exit()
    print("next one")
    print(model.irr_supp(x[2]))
    print("")
    S = np.r_[model.stoich,np.eye(len(x[2]))[np.setdiff1d(supp(model.irr),model.irr_supp(x[2]))]]
    
    x = get_efvs(S,model.rev)
    y = get_gens(S,model.rev) 
    
    for efv in x:
        print(model.check_dim(efv))
    print("")
    for efv in y:
        print(model.check_dim(efv))
    
    
    
    sys.exit()
    efvs_in_2faces = []
    
    for index,er in enumerate(model.generators):
        if index%25 == 0:
            print(index)
        for neigh_ind in model.adjacency[index]:
            neigh = model.generators[neigh_ind]
            face2_efvs = model.rev_cancellations(np.r_[er,neigh].reshape(2,len(er)))
            if len(face2_efvs) > 2:
                efvs_in_2faces.append(face2_efvs)
    face_lens = [len(face) for face in efvs_in_2faces]
    print(sorted(Counter(face_lens).items()))
    two_in_a_face = []
    three_in_a_face = []
    for efv_set in efvs_in_2faces:
        if len(efv_set) == 4:
            two_in_a_face.append([efv_set[2],efv_set[3]])
        if len(efv_set) == 5:
            three_in_a_face.append([efv_set[2],efv_set[3],efv_set[4]])
    
    for efv_set in two_in_a_face:
        if not model.irr_supp(efv_set[0]) == model.irr_supp(efv_set[1]):
            print("Issue")
            print(efv_set)
            break
    print("All pairs of 2 efvs in the interior of the same 2-face have equal irr_supp")
    
    for efv_set in three_in_a_face:
        if not (model.irr_supp(efv_set[0]) == model.irr_supp(efv_set[1]) and model.irr_supp(efv_set[1]) == model.irr_supp(efv_set[2])):
            print("Issue")
            print(efv_set)
            break
    print("All triplets of 3 efvs in the interior of the same 2-face have equal irr_supp")
    
    
    
    
    