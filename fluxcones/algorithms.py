import numpy as np


from fluxcones.helpers import abs_max, supp, TOLERANCE
from fluxcones import FluxCone
import itertools
from ortools.linear_solver import pywraplp

def MILP_shortest_decomp(target_vector, candidates, tolerance=1e-7, bigM=1000):
    solver = pywraplp.Solver.CreateSolver('CBC')
    if not solver:
        raise Exception("CBC solver unavailable in OR-Tools.")

    n = len(candidates)
    dim = len(target_vector)

    # Variables
    a = [solver.BoolVar(f'a_{i}') for i in range(n)]
    x = [solver.NumVar(0.0, solver.infinity(), f'x_{i}') for i in range(n)]

    # Constraints: x[i] <= M * a[i]
    for i in range(n):
        solver.Add(x[i] <= bigM * a[i])

    # Stoichiometric constraints
    for flux in range(dim):
        constraint_expr = solver.Sum(x[i] * candidates[i][flux].item() for i in range(n))
        solver.Add(constraint_expr == target_vector[flux].item())

    # At least one candidate is used
    solver.Add(solver.Sum(a[i] for i in range(n)) >= 1)

    # Objective: minimize number of active candidates (sum of a)
    solver.Minimize(solver.Sum(a))

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        coefficients = np.array([x[i].solution_value() for i in range(n)])
        return coefficients
    else:
        print("No optimal solution found.")
        return None





def check_conjecture(model, efms):

    if len(efms) <= 3:
        # Conjecture holds trivially
        return True

    else:
        for index, efm in enumerate(efms):
            if model.degree(efm) <= 2:
                # Conjecture holds for EFMs of degree smaller 3
                continue

            if len(two_gens(efm, efms, model)) == 2:
                # Decomposition of length 2 was found
                continue

            coeffs = MILP_shortest_decomp(efm, np.delete(efms, index, axis=0))

            # check if efm was decomposable
            if any(element is None for element in coeffs.ravel()):
                # efm was not decomposable
                continue

            if len(supp(coeffs)) > 2:
                print("A counterexample was found")
                return False
        return True


""" find all pairs of 2 EFMs that can be positively combined to vector """


def two_gens(
    vector,
    efms: np.array,
    fluxcone: FluxCone,
    face_candidates: bool = False,
    all_pairs: bool = False,
):
    if face_candidates:
        # Return the unique values in non-zero irr of fluxcone that are not in non-zero irr values of vector.
        efms = efms[
            np.where(
                np.all((np.round(efms[:, fluxcone.irr_zeros(vector)], 10) == 0), axis=1)
            )
        ]

    gen_pairs = []
    for rev_zero_ind in fluxcone.rev_zeros(vector):
        pos = efms[np.where(efms[:, rev_zero_ind] > TOLERANCE)]
        neg = efms[np.where(efms[:, rev_zero_ind] < -TOLERANCE)]

        if len(pos) <= 0 or len(neg) <= 0:
            continue

        for pos_efm, neg_efm in itertools.product(pos, neg):
            new_vec = pos_efm - pos_efm[rev_zero_ind] / neg_efm[rev_zero_ind] * neg_efm
            if len(supp(new_vec)) == len(supp(vector)):
                if all(supp(new_vec) == supp(vector)):
                    if all_pairs:
                        gen_pairs.append((pos_efm, neg_efm))
                    else:
                        return (pos_efm, neg_efm)

    return gen_pairs
