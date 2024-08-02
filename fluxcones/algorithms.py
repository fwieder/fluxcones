import numpy as np
import mip
from fluxcones.helpers import abs_max, supp, TOLERANCE
from fluxcones import FluxCone
import itertools


def MILP_shortest_decomp(target_vector, candidates, tolerance=1e-7, bigM = 1000):

    m = mip.Model()

    # numeric tolerances:
    m.infeas_tol = tolerance
    m.integer_tol = tolerance

    # suppress console output
    m.verbose = False

    # Define binary variables
    a = [m.add_var(var_type=mip.BINARY) for i in range(len(candidates))]

    # Define real variables
    x = [m.add_var(var_type=mip.CONTINUOUS) for i in range(len(candidates))]

    # Define bigM
    M = bigM

    # Logic constraints a[i] = 0 => x[i]=0
    for i in range(len(candidates)):
        m.add_constr(x[i] <= M * a[i])
        m.add_constr(x[i] >= 0)

    # Stoichiometric constraints
    for flux in range(len(target_vector)):
        m += (
            mip.xsum(x[i] * candidates[i][flux].item() for i in range(len(candidates)))
            == target_vector[flux].item()
        )

    # Make sure that at least one candidate is used
    m += mip.xsum(a[i] for i in range(len(candidates))) >= 1

    # Define objective function
    m.objective = mip.minimize(mip.xsum(a[i] for i in range(len(candidates))))

    # Solve the MILP
    m.optimize()

    # Determine coefficients in decomposition
    coefficients = np.array([x[i].x for i in range(len(x))])

    # Clear model variables to avoid possible issues when doing multiple MILPs consecutively
    m.clear()

    return coefficients


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
            if abs_max(new_vec - vector) < TOLERANCE:
                if all_pairs:
                    gen_pairs.append((pos_efm, neg_efm))
                else:
                    return (pos_efm, neg_efm)

    return gen_pairs
