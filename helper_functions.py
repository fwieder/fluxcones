import numpy as np
import efmtool,cdd
from mip import *
#######################################################################################################################
# "Helper functions" 
#######################################################################################################################
tol = 1e-10

# Support function returns a np.array containing the indices of all entries of a vector larger than the tol
def supp(vector,tol = tol):
    return np.where(abs(vector) > tol)[0]

# Zero function returns a np.array containing the indices of all entries of a vector smaller than the tol
def zero(vector,tol = tol):
    return np.where(abs(vector) < tol)[0]

# Return the maximal absolute value
def abs_max(vector):
    if all(vector == np.zeros(len(vector))):
        return 0
    abs_max = np.max(np.absolute(vector[vector!=0]))
    return abs_max


def milp_efms(S, rev):
    # Create the extended stoichiometric matrix for reversible reactions
    for index in np.nonzero(rev)[0]:
        S = np.c_[S, -S[:, index]]
    
    n = np.shape(S)[1]
    # Initialize the MILP model
    m = Model(sense=MINIMIZE)

    m.verbose = False    


    # Add binary variables for each reaction
    a = [m.add_var(var_type=BINARY) for _ in range(n)]

    # Add continuous variables for each reaction rate
    v = [m.add_var() for _ in range(n)]

    # Add stoichiometric constraints
    for row in S:
        m += xsum(row[i] * v[i] for i in range(n)) == 0

    # Define the Big M value for constraints
    M = 1000
    for i in range(n):
        m += a[i] <= v[i]
        m += v[i] <= M * a[i]

    # Exclude the zero vector solution
    m += xsum(a[i] for i in range(n)) >= 1

    # Set the objective to minimize the number of non-zero variables
    m.objective = xsum(a[i] for i in range(n))

    efms = []

    while True:
        # Solve the MILP model
        m.optimize()

        # Get the solution vector
        efm = np.array([v.x for v in m.vars[:n]])

        # Check for optimality
        if efm.any() is None:
            break

        # Add constraint to exclude the current solution in the next iteration
        m += xsum(a[i] for i in supp(efm)) <= len(supp(efm)) - 1

        efms.append(efm)

    efms = np.array(efms)

    # Separate positive and negative parts for reversible reactions
    efms_p = efms[:, :len(rev)]
    efms_m = np.zeros(np.shape(efms_p))
    
    counter = 0
    for r in supp(rev):
        efms_m[:, r] = efms[:, len(rev) + counter]
        counter += 1

    efms = efms_p - efms_m

    # Remove zero rows
    efms = efms[np.any(efms != 0, axis=1)]

    return efms

# get_efms is a wrapper for efmtool and CDD to compute EFMs,
# INPUT np.array stoich: stoichiometric matrix,
# np.array rev: (0,1) vector for reversibility of reactions
# returns np.array that contains EFMs as rows
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
def get_efms(stoich,rev, algo = "efmtool"):
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
            if len(supp(vector)) > 0:
                tokeep.append(index)
        efms = unsplit[tokeep]
        
        
        return(efms)
    
    if algo == "efmtool":
        
        ''' 
        initiate reaction names and metabolite names from 0 to n resp. m because 
        efmtool needs these lists of strings as input
        "normalize options:  [max, min, norm2, squared, none] 
        '''
        
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
        efms = efmtool.calculate_efms(stoich,rev,reaction_names,metabolite_names,opts)
        
        
        return(efms.T)
    
    if algo == "milp":
        efms = milp_efms(stoich, rev)
        return efms
