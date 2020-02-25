import random
import time
import numpy as np


def timer(func, *args, **kwargs):
    def _wrapper_function(*args, **kwargs):
        tic = time.perf_counter()
        response = func(*args)
        toc = time.perf_counter()
        print(f"Function has been running for {toc - tic} seconds.")
        return response, toc - tic

    return _wrapper_function

@timer
def multiply_matrix(A, B):
    n = len(A)
    m = len(B)
    B_Transformed = [[B[i][j] for i in range(m)] for j in range(n)]
    M = [[vector_scalar(row, column) for column in B_Transformed] for row in A]
    return M


def vector_scalar(a, b):
    S = 0
    for el1, el2 in zip(a, b):
        S += el1 * el2
    return S


def zeros(n, m):
    return [[None for j in range(m)] for i in range(n)]


@timer
def multiply_matrix_np(A, B):
    A = np.array(A)
    B = np.array(B)
    M = A.dot(B)
    return M


if __name__ == '__main__':
    n = 100
    m = 100
    A = [[random.random() for i in range(m)] for i in range(n)]
    B = [[random.random() for i in range(m)] for i in range(n)]

    print("Numpy matrix multiplication: ")
    M, time_np = multiply_matrix_np(A, B)
    print("My matrix multiplication: ")
    M, time_my = multiply_matrix(A, B)

    print("Difference: {} or {} times.".format(time_my-time_np, time_my/time_np))