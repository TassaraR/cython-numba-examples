#cython: language_level=3
cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double mmult_cy(double[:, :] m1, double[:, :] m2):

    cdef:
        int size = m1.shape[0]
        double result = 0
        int i
        int j
        
    for i in range(size):
        for j in range(size):
            result += m1[i, j] * m2[i, j]
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def conv_cy(double[:, :, :] img, double[:, :] f, int s=1):
    
    cdef: 
        int h = img.shape[0]
        int w = img.shape[1]
        int d = img.shape[2]

        int f_size = f.shape[0]
        int f_sum = np.sum(f)

        int new_size = int(((h - f_size) / s) + 1)

        double[:, :, :] result_view = np.zeros((new_size, new_size, d),
                                               dtype=np.float64)

        double[:, :] img_seg_view = np.zeros((f_size, f_size),
                                             dtype=np.float64)
        double conv
        int k
        int i
        int j
    
    for k in range(d):
        for i in range(new_size):
            for j in range(new_size):
                    img_seg_view[:, :] = img[s*i:s*i+f_size, s*j:s*j+f_size, k]
                    result_view[i, j, k] = mmult_cy(img_seg_view, f) / f_sum

    return np.asarray(result_view)
