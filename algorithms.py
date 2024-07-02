import numpy as np
import mip
from flux_cones import *

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
    
    if len(model.efms) < 3:
        return True
    else:
        for index,efm in enumerate(model.efms):
            if model.degree(efm) > 2:
                if len(model.two_gens(efm))==0:
                    coeffs = MILP_shortest_decomp(efm,np.delete(model.efms,index,axis=0))
                    if len(supp(coeffs)) > 2:
                        print("A counterexample was found")
                        return False
        return True


from mip import *

def milp_efms(S, rev):
    # Create the extended stoichiometric matrix for reversible reactions
    for index in np.nonzero(rev)[0]:
        S = np.c_[S, -S[:, index]]
    
    n = np.shape(S)[1]

    # Initialize the MILP model
    m = Model(sense=MINIMIZE)

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
        print(efm)

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
