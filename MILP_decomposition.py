import numpy as np
import mip

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
      
        
    

if __name__ == "__main__":

    target = np.array([10, -20, 30])
    candidatess = np.array([[1, -2, 0], [0, -20, 0], [0, 0, 30],[10,-20,0]])  # Example candidatesdidate vectors
   
    M = MILP_shortest_decomp(target, candidatess,1e-6)
    print(M)    
    