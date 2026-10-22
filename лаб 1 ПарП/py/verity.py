import numpy as np

def read_matrix(filename):
    with open(filename,"r") as f:
        n=int(f.readline())

        matrix=[]

        for _ in range (n):
            row=list(map(float,f.readline().split()))
            matrix.append(row)

    return np.array(matrix)

A = read_matrix("data/matrix_A.txt")
B = read_matrix("data/matrix_B.txt")
C_cpp = read_matrix("results/matrix_C.txt")

C_ref=np.matmul(A,B)

if np.allclose(C_cpp,C_ref,rtol=1e-9,atol=1e-9):
    print("passed")
else:
    print("not passed dawg")

    diff=np.max(np.abs(C_cpp-C_ref))
    print("max difference:",diff)