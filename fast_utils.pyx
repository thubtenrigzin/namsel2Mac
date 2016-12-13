from libc.math cimport sqrt, floor, fmod, log, abs
import numpy as np
cimport numpy as np
cimport cython
from cv2 import resize, INTER_CUBIC

cdef extern from "math.h":
    bint isnan(double x)

def gausslogprob(double mean, double std, double x):
    cdef double pi = np.pi
    cdef double df = x - mean
    cdef double dn = log(sqrt(2*pi))
    return (-(df*df)/(2*std*std)) - dn - log(std)
#     return (-(x-mean)/(-2*std))-log(sqrt(2*std)) # - log()
#     cdef ninf = -np.inf
#     cdef double p = log(abs(x - mean)) - log(std)
#     if isnan(p):
#         return ninf
#     else:
#         return p 


cpdef scale_transform(double [:] x, double [:] mean, double [:] o_std, int size):
    cdef int i
    for i in range(size):
        x[i] = (x[i] - mean[i])*o_std[i]

@cython.boundscheck(False)
cpdef fadd_padding(np.uint8_t [:,:] arr, int padding):
    cdef int i, j
    cdef int h = arr.shape[0]
    cdef int w = arr.shape[1]
    cdef int nh = h+2*padding
    cdef int nw = w+2*padding
    cdef int ed1 = padding
    cdef int ed2 = nw-padding
    cdef int top1 = padding
    cdef int top2 = nh-padding
    cdef np.ndarray[np.uint8_t, ndim=2] newarr = np.empty((nh, nw), np.uint8)
    
    for i in range(nh):
        for j in range(nw):
            if (i < top1) or (i >= top2):
                newarr[i,j] = 1
            elif (j < ed1) or (j >= ed2):
                newarr[i,j] = 1
                
            else:
                newarr[i,j] = arr[i-padding, j-padding]
    return newarr
    

@cython.boundscheck(False)
cpdef ftrim(np.ndarray[np.uint8_t, ndim=2] arr, sides='trbl', new_offset=False):
    cdef int top = 0
    cdef int left = 0
    cdef int right = arr.shape[1]
    cdef int rows = arr.shape[0]
    cdef int bottom = rows
    cdef int i, j
    cdef int oft = 0
    cdef int ofb = 0
    cdef int ofr = 0
    cdef int ofl = 0
#     cdef np.ndarray[np.uint8_t, ndim=1] row
    cdef np.uint8_t [:,:] arrT = arr.T
#     cdef np.float32_t [:,:] log_transmatT = log_transmat.T
 
#     offset = {'top':0, 'bottom':0, 'right':0, 'left':0}
    if 't' in sides:
        brk = False
        for i in range(rows):
#             row = arr[i]# print dir(scaler)
# print scaler.mean_.shape, scaler.mean_.dtype
# print scaler.std_.shape, scaler.std_.dtype
#             if not row.all():
            for j in range(right):
                if arr[i,j] == 0:
                    top = i
                    oft = i
                    brk = True
                    break
            if brk: break
             
    if 'b' in sides:
        brk = False
        for i in range(bottom-1, 0, -1):
#             row = arr[i]
#             if not row.all():
            for j in range(right):
                if arr[i,j] == 0:
                    ofb = -(bottom-i)
                    bottom = i
                    brk = True
                    break
            if brk: break
     
    if 'l' in sides:
        brk = False
        for i in range(right):
#             row = arrT[i]
#             if not row.all():
            for j in range(rows):
                if arrT[i,j] == 0:
                    left = i
                    ofl = i
                    brk = True
                    break
            if brk: break
     
    if 'r' in sides:
        brk = False
        for i in range(right-1, 0, -1):
#             row = arrT[i]
#             if not row.all():
            for j in range(rows):
                if arrT[i,j] == 0:
                    ofr = -(right-i)
                    right = i
                    brk = True
                    break
            if brk: break
     
#    print bottom, top, left, right
    if not new_offset:
        return arr[top:bottom, left:right]
    else:
        return arr[top:bottom, left:right], {'top':oft, 'bottom':ofb, 'right':ofr, 'left':ofl}



# cpdef to255(np.uint8_t [:,:] a):
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef to255(np.ndarray[np.uint8_t, ndim=2] a):
    cdef int i, j
    cdef int h = a.shape[0]
    cdef int w = a.shape[1]
    for i in range(h):
        for j in range(w):
            a[i,j] = a[i,j]*255
    return a



@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cpdef fnormalize(np.ndarray[np.uint8_t, ndim=2] a, np.ndarray[np.uint8_t, ndim=2] c):
    cdef double h = a.shape[0]
    cdef double w = a.shape[1] 
    cdef double L = 32.0
    cdef int LL = 32
    cdef double o_2 = 1.0/2.0
    cdef double R1, R2, H2, W2, offset, start, end, alpha, beta, sm, bg, smn, df
#     cdef np.ndarray[np.uint8_t, ndim=2] c = np.empty((L,L), dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=2] b
    cdef int smi, bgi, i, j, starti, endi
    if h >= w:
        bg = h
        sm = w
        smi = 1
        bgi = 0
    else:
        bg = w
        sm = h
        smi = 0
        bgi = 1
        
    R1 = sm/bg

    R2 = sqrt(R1)

        
    if sm == h:
        H2 = L*R2
        W2 = L
    else:
        H2 = L
        W2 = L*R2
    
    alpha = W2 / w
    beta = H2 / h
#    print alpha,beta

#     b = resize(a, (0,0), fy=beta, fx=alpha, interpolation=INTER_CUBIC)
    b = resize(a, (0,0), fy=beta, fx=alpha, interpolation=INTER_CUBIC)
    smn = b.shape[smi]
    df = L - smn
#     b = np.array(b)
#        raise
    
    offset = floor(df * o_2)
    
#        print a
#        print a.shape
    if fmod(df, 2) == 1.0:
        start = offset+1.0
        end = offset
    else:
        start = end = offset
    
#     print start, end, L
    
    starti = int(start)
    endi = int(end)
    
    if sm == h:

        for i in range(LL):
            for j in range(LL):
                if i < starti or i >= LL-endi:
                    c[i,j] = 1
                else:
                    c[i,j] = b[i-starti,j]

                     
        
    else:
        
        for i in range(32):
            for j in range(32):
                if j < starti or j >= LL-endi:
                    c[i,j] = 1
                else:
                    c[i,j] = b[i,j-starti]

#     return c