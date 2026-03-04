import numpy as np

from solver import (

    build_hankel_matrix, solve_recurrence_coefficients, find_roots,

    solve_vandermonde, compute_residual, one_shot_reconstruction,

    classical_sylvester, svd_baseline

)





def create_test_case():

    

    

    

    

    

    

    

    roots = np.array([1])

    weights = np.array([1])

    d = 4

    a = []

    for i in range(d + 1):

        a_i = np.sum(weights * (roots ** i))

        a.append(a_i)

    a = np.array(a)

    print(f"Test case created:")

    print(f"Roots: {roots}")

    print(f"Weights: {weights}")

    print(f"Coefficients: {a}")

    print(f"True rank: {len(roots)}")

    return a, roots, weights



def create_test_case_rank2():

    

    roots = np.array([1, 2])

    weights = np.array([1, 1])

    d = 4

    a = []

    for i in range(d + 1):

        a_i = np.sum(weights * (roots ** i))

        a.append(a_i)

    a = np.array(a)

    print(f"\nRank 2 test case created:")

    print(f"Roots: {roots}")

    print(f"Weights: {weights}")

    print(f"Coefficients: {a}")

    print(f"True rank: {len(roots)}")

    return a, roots, weights



def test_components(a, roots, weights):

    print("\n=== Testing Components ===")

    

    r = len(roots)

    print(f"Testing for rank {r}")

    

    

    print("\n0. Debug Info:")

    H = build_hankel_matrix(a, r + 1)

    print(f"Hankel matrix shape: {H.shape}")

    print(f"Hankel matrix:")

    print(H)

    c = solve_recurrence_coefficients(H)

    print(f"Recurrence coefficients: {c}")

    print(f"Expected length of c: {r}")

    if len(c) == r:

        roots_found = find_roots(c)

        print(f"Found roots: {roots_found}")

        print(f"Number of roots found: {len(roots_found)}")

        

        weights_found = solve_vandermonde(roots_found, a)

        print(f"Found weights: {weights_found}")

        residual = compute_residual(a, roots_found, weights_found)

        print(f"Residual: {residual}")

        print(f"Tau: 1e-6")

        print(f"Residual < Tau: {residual < 1e-6}")

    

    

    print("\n1. Testing One-shot Reconstruction:")

    result = one_shot_reconstruction(a, r, tau=1e-6)

    print(f"Result: {result}")

    if result:

        r_pred, found_roots, found_weights = result

        residual = compute_residual(a, found_roots, found_weights)

        print(f"Predicted rank: {r_pred}")

        print(f"Found roots: {found_roots}")

        print(f"Found weights: {found_weights}")

        print(f"Residual: {residual}")

    

    

    print("\n2. Testing Classical Sylvester:")

    classical_result = classical_sylvester(a, r, tau=1e-6)

    print(f"Result: {classical_result}")

    if classical_result:

        r_pred, found_roots, found_weights = classical_result

        residual = compute_residual(a, found_roots, found_weights)

        print(f"Predicted rank: {r_pred}")

        print(f"Found roots: {found_roots}")

        print(f"Found weights: {found_weights}")

        print(f"Residual: {residual}")

    

    

    print("\n3. Testing SVD Baseline:")

    svd_result = svd_baseline(a, r, tau=1e-6)

    print(f"Result: {svd_result}")

    if svd_result:

        r_pred, found_roots, found_weights = svd_result

        residual = compute_residual(a, found_roots, found_weights)

        print(f"Predicted rank: {r_pred}")

        print(f"Found roots: {found_roots}")

        print(f"Found weights: {found_weights}")

        print(f"Residual: {residual}")

    

    

    print("\n4. Testing with higher R_max:")

    classical_result_high = classical_sylvester(a, r+2, tau=1e-6)

    print(f"Classical Sylvester with R_max={r+2}: {classical_result_high}")

    svd_result_high = svd_baseline(a, r+2, tau=1e-6)

    print(f"SVD baseline with R_max={r+2}: {svd_result_high}")



if __name__ == "__main__":

    

    print("=== Testing Rank 1 Case ===")

    a, roots, weights = create_test_case()

    test_components(a, roots, weights)

    

    

    print("\n=== Testing Rank 2 Case ===")

    a2, roots2, weights2 = create_test_case_rank2()

    test_components(a2, roots2, weights2)

