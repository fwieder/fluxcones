import numpy as np
import cobra
TOLERANCE = 1e-10


# Support function returns a np.array containing the indices of all entries of a vector larger than the tol
def supp(vector, tol=TOLERANCE):
    return list(np.where(abs(vector) > tol)[0])


# Zero function returns a np.array containing the indices of all entries of a vector smaller than the tol
def zero(vector, tol=TOLERANCE):
    return list(np.where(abs(vector) < tol)[0])


# Return the maximal absolute value
def abs_max(vector):
    if all(vector == np.zeros(len(vector))):
        return 0
    abs_max = np.max(np.absolute(vector[vector != 0]))
    return abs_max

# Standardize model so that blocked reactions are deleted and irreversible reacctions are positively oriented
def standardize_model(cobra_model):
    
    # this loop appends the indices of blocked reactions to "blocked reactions" and inverts negative irreversible reactions
    
    
    toadd = []
    todel = []
    for rea in cobra_model.reactions:
        
        if rea.bounds == (0,0):
            todel.append(rea)
            #print(rea.id , "deleted")
        if not rea.reversibility and rea.lower_bound < 0:
           
            reaction = cobra.Reaction(rea.id + "_inverted")
            
            reaction.lower_bound = -rea.upper_bound
            reaction.upper_bound = -rea.lower_bound
            metabs = {}
            for key in rea.metabolites.keys():
                metabs[key] = -rea.metabolites[key]
            reaction.add_metabolites(metabs)
            #print(rea.id , "inverted")
            toadd.append(reaction)
            todel.append(rea)
            
    cobra_model.add_reactions(toadd)
    cobra_model.remove_reactions(todel)
           
    return cobra_model