import numpy as np


from fluxcones.helpers import abs_max, supp, TOLERANCE
from fluxcones import FluxCone
import itertools
import pulp


def nonneg_lincomb(v: np.ndarray, A: np.ndarray, bigM: float = 1e6) -> np.ndarray:
    """
    Express a vector v as a sparse nonnegative linear combination of rows of A.
    Minimizes L1 reconstruction error, then sparsity (#nonzero lambdas).

    Parameters
    ----------
    v : np.ndarray (shape: (n,))
        Target vector.
    A : np.ndarray (shape: (m, n))
        Candidate row vectors.
    bigM : float
        Upper bound on λ_i values (should be >= any feasible coefficient).

    Returns
    -------
    np.ndarray (shape: (m,))
        Sparse nonnegative coefficients λ.
    """
    m, n = A.shape
    assert v.shape[0] == n, "Dimension mismatch: v must have same length as A's columns."

    # Define problem
    prob = pulp.LpProblem("SparseNonNegLinComb", pulp.LpMinimize)

    # λ_i ≥ 0
    lambdas = [pulp.LpVariable(f"lambda_{i}", lowBound=0) for i in range(m)]

    # Binary usage indicators
    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(m)]

    # Link λ and y: λ_i <= M * y_i
    for i in range(m):
        prob += lambdas[i] <= bigM * y[i]

    # Slack variables for L1 error
    slacks_plus = [pulp.LpVariable(f"splus_{j}", lowBound=0) for j in range(n)]
    slacks_minus = [pulp.LpVariable(f"sminus_{j}", lowBound=0) for j in range(n)]

    # Constraints: A^T λ - v = slack_plus - slack_minus
    for j in range(n):
        prob += (pulp.lpSum(A[i, j] * lambdas[i] for i in range(m)) - v[j] 
                 == slacks_plus[j] - slacks_minus[j])

    # Objective: first minimize reconstruction error, then #nonzeros
    # Use a big weight to prioritize reconstruction
    error_term = pulp.lpSum(slacks_plus) + pulp.lpSum(slacks_minus)
    sparsity_term = pulp.lpSum(y)
    prob += 1e6 * error_term + sparsity_term

    # Solve
    solver = pulp.HiGHS(msg=False)
    prob.solve(solver)

    # Extract solution
    solution = np.array([pulp.value(lmbd) for lmbd in lambdas])
    return solution




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

            coeffs = nonneg_lincomb(efm,efms)

            # check if efm was decomposable
            if coeffs == None:
                return True

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
