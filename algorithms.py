import numpy as np
import mip
from helper_functions import *
from algorithms import *
###########################################################
    
def MILP_shortest_decomp(target_vector, candidates,tolerance = 1e-7):
    
    m = mip.Model()    
    
    # numeric tolerances:
    m.infeas_tol = tolerance
    m.integer_tol = tolerance
    
    # suppress console output
    m.verbose = False
        
    
    # Define binary variables
    a = [m.add_var(var_type=mip.BINARY) for i in range(len(candidates))]
    
    # Define real variables
    x = [m.add_var(var_type=mip.CONTINUOUS)for i in range(len(candidates))]
    
    # Define bigM
    M = 100
    
    # Logic constraints a[i] = 0 => x[i]=0
    for i in range(len(candidates)):
        m.add_constr(x[i] <= M*a[i])
        m.add_constr(x[i] >= 0)
    
    # Stoichiometric constraints
    for flux in range(len(target_vector)):
        m += mip.xsum(x[i]*candidates[i][flux] for i in range(len(candidates))) == target_vector[flux]
    
    # Make sure that at least one candidate is used
    m += mip.xsum(a[i]for i in range(len(candidates))) >= 1
    
    # Define objective function
    m.objective = mip.minimize(mip.xsum(a[i]for i in range(len(candidates))))
    
    # Solve the MILP
    m.optimize()
        
    # Determine coefficients in decomposition
    coefficients = np.array([x[i].x for i in range(len(x))])
      
    # Clear model variables to avoid possible issues when doing multiple MILPs consecutively
    m.clear()  
    
    return coefficients

def check_conjecture(model):
    model.get_efms("cdd")
    
    if len(model.efms) <= 3:
        # Conjecture holds trivially
        return True
    
    else:
        for index,efm in enumerate(model.efms):
            if model.degree(efm) <= 2:
                # Conjecture holds for EFMs of degree smaller 3
                continue
        
            if len(model.two_gens(efm))==2:
                # Decomposition of length 2 was found
                continue
            
            coeffs = MILP_shortest_decomp(efm,np.delete(model.efms,index,axis=0))
            
            #check if efm was decomposable
            if any(element is None for element in coeffs.ravel()):
                # efm was not decomposable
                continue
            
            if len(supp(coeffs)) > 2:
                print("A counterexample was found")
                return False
        return True

