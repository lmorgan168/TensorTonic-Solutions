import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    ## set A to be an array of size MxN
    ## for i,j set Aji. This is O(N^2) 

    A = np.array(A)
    transpose = np.zeros(A.shape[::-1], dtype = A.dtype)

    for i in range(A.shape[0]):
        for j in range (A.shape[1]):
            transpose[j][i] = A[i][j]
    
    return transpose
    pass
