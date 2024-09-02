import numpy as np

# Coefficient matrix A
A = np.array([[3, 1], [1, 2]])

# Constants vector b
b = np.array([9, 8])

# Solve for x
x = np.linalg.solve(A, b)

print("A:\n", A)
print("b:\n", b)
print("Solution:", x)


Ax = np.dot(A, x)
print("Ax:\n", Ax)


# Square matrix A
A = np.array([[1, 2], [3, 4]])

# Compute the inverse of A
A_inv = np.linalg.inv(A)

print("Inverse of A:\n", A_inv)
Axinv = np.dot(A, A_inv)
print("A*inv:\n", np.round(Axinv).astype(int))
