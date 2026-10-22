import numpy as np
 
sizes = [20, 200, 400, 800, 1200, 1600, 2000]
 
 
def read_matrix(filename):
    with open(filename, "r") as f:
        n = int(f.readline())
        matrix = [list(map(float, f.readline().split())) for _ in range(n)]
    return np.array(matrix)
 
 
for n in sizes:
    A = read_matrix(f"data/matrix_a_{n}.txt")
    B = read_matrix(f"data/matrix_b_{n}.txt")
    C_cpp = read_matrix(f"results/matrix_C_{n}.txt")
 
    C_ref = np.matmul(A, B)
 
    if np.allclose(C_cpp, C_ref, rtol=1e-9, atol=1e-9):
        print(f"{n}: passed")
    else:
        diff = np.max(np.abs(C_cpp - C_ref))
        print(f"{n}: not passed, max difference: {diff}")