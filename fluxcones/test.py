import numpy as np
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
        #print("No optimal solution found.")
        return None


def test_exact_match():
    target_vector = np.array([5, 10])
    candidates = [np.array([5, 10])]
    coeffs = MILP_shortest_decomp(target_vector, candidates)
    assert coeffs is not None, "No solution found"
    assert np.allclose(sum(coeffs[i] * candidates[i] for i in range(len(candidates))), target_vector)
    print("test_exact_match passed.")

def test_multiple_candidates():
    target_vector = np.array([3, 5])
    candidates = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
    coeffs = MILP_shortest_decomp(target_vector, candidates)
    assert coeffs is not None, "No solution found"
    reconstructed = sum(coeffs[i] * candidates[i] for i in range(len(candidates)))
    assert np.allclose(reconstructed, target_vector)
    print("test_multiple_candidates passed.")

def test_no_solution():
    target_vector = np.array([1, 1])
    candidates = [np.array([2, 3]), np.array([3, 5])]
    coeffs = MILP_shortest_decomp(target_vector, candidates)
    assert coeffs is None, "Expected no solution"
    print("test_no_solution passed.")

def test_zero_target():
    target_vector = np.array([0, 0])
    candidates = [np.array([1, 2]), np.array([3, 4])]
    coeffs = MILP_shortest_decomp(target_vector, candidates)
    assert coeffs is not None, "No solution found"
    assert np.allclose(coeffs, 0)
    print("test_zero_target passed.")

def test_tolerance():
    target_vector = np.array([1.000001, 2.000001])
    candidates = [np.array([1, 2])]
    coeffs = MILP_shortest_decomp(target_vector, candidates, tolerance=1e-5)
    assert coeffs is not None, "No solution found"
    reconstructed = sum(coeffs[i] * candidates[i] for i in range(len(candidates)))
    assert np.allclose(reconstructed, target_vector, atol=1e-5)
    print("test_tolerance passed.")

if __name__ == "__main__":
    test_exact_match()
    test_multiple_candidates()
    test_no_solution()
    test_zero_target()
    test_tolerance()
