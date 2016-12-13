from __future__ import division
import numpy as np
# import cPickle as pickle
cimport numpy as np
cimport cython
# cimport scipy.linalg.cython_blas as blas
# from scipy.linalg.blas import dgemm
# dgemm = blas.dgemm

# import scipy.linalg.blas
# from cpython cimport (PY_VERSION_HEX, PyCObject_Check,
#     PyCObject_AsVoidPtr, PyCapsule_CheckExact, PyCapsule_GetPointer)
# 
# cdef void* f2py_pointer(obj):
#     if PY_VERSION_HEX < 0x03000000:
#         if (PyCObject_Check(obj)):
#             return PyCObject_AsVoidPtr(obj)
#     elif PY_VERSION_HEX >= 0x02070000:
#         if (PyCapsule_CheckExact(obj)):
#             return PyCapsule_GetPointer(obj, NULL);
#     raise ValueError("Not an object containing a void ptr")
# 
# ctypedef int dgemm_t(
#     char *transa, char *transb,
#     int *mm, int *nn, int *kk,
#     double *alpha,
#     double *aa, int *lda,
#     double *bb, int *ldb,
#     double *beta,
#     double *cc, int *ldc)

# cdef dgemm_t *dgemm = <dgemm_t*>f2py_pointer(scipy.linalg.blas.dgemm._cpointer)

# from cython.parallel import prange
# from datetime import datetime

# now = datetime.now
#cdef np.ndarray[double, ndim=2] D, Bpqk, Ipi

#D = pickle.load(open('D_matrix.pkl','r'))
#Bpqk = pickle.load(open('Bpqk17.pkl', 'rb'))
#Ipi = pickle.load(open('Ipi32.pkl','rb'))

 
# cdef double cabs(np.complex64_t z):
#     return sqrt(z.real*z.real + z.imag*z.imag) 
# cpdef zernike_features(np.uint8_t [:,:] A, 
 
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef zernike_features(np.ndarray[double, ndim=2] A, 
        np.ndarray[double, ndim=2] D,
        np.ndarray[double, ndim=3] Bpqk,
#         np.ndarray[double, ndim=2] Ipi,
        double[::1, :] Ipi,
        double[::1, :] Mpqs,
#         double[:,:] Ipi,
#         double[:,:] Mpqs,
#         np.ndarray[double, ndim=2] Mpqs,
        np.ndarray[complex, ndim=2] Rpq,
#         np.ndarray[double, ndim=2] Yiq,
        double[::1, :] Yiq,
#         double[:,:] Yiq,
        np.ndarray[complex] ws,
        np.ndarray[double] Zpq
                     ):
#    print 'init section'
#    start = now()
    cdef int p, q, k, j, S, diff, m, i, pp1
    cdef int N = 32 # A.shape[0]
    cdef double o_PI = 1.0/np.pi
    cdef double yiq, mpq, val, zrl, zim
    cdef int deg = 17
#    cdef int deg = 16
    cdef double complex w = -1.0j
    cdef double complex zpk, rpq, dw
#     cdef np.ndarray[double, ndim=2] Mpqs = np.empty((deg+1,deg+1))
#     cdef np.ndarray[complex, ndim=2] Rpq = np.empty((deg+1,deg+1), complex)
#    cdef np.ndarray[complex, ndim=2] Zpq = np.zeros((deg+1,deg+1), complex)
    
#    print '\t pickle loading',
#    innerstart = now()# cdef dgemm_t *dgemm = <dgemm_t*>f2py_pointer(scipy.linalg.blas.dgemm._cpointer)
#    cdef np.ndarray[double, ndim=2] D = pickle.load(open('D_matrix.pkl','r'))
#    cdef np.ndarray[double, ndim=3] Bpqk =  pickle.load(open('Bpqk17.pkl', 'rb'))
#    cdef np.ndarray[double, ndim=2] Ipi = pickle.load(open('Ipi32.pkl','rb'))
#    print now() - innerstart
#    cdef np.ndarray[complex] Zpq = np.zeros((90), complex)
#     cdef np.ndarray[double] Zpq = np.empty((90), np.double) # deg =17
#    cdef np.ndarray[double] Zpq = np.zeros((256), np.double) # deg = 30

#     cdef np.ndarray[complex] ws = np.array([1,-1j,-1,1j], complex)
    
#    cdef np.ndarray[long] ix, iy
#    
#    inx = np.where(A==0)
#    ix = inx[0]
#    iy = inx[1]
#    
#    IX = ix.shape[0]

#     cdef np.ndarray[double, ndim=2] Yiq = np.zeros((deg+1,N), np.double)
#    print now() - start
#    for q in range(0, deg+1):
#        for i in range(0,N):
#            yiq = 0.0   
#            for j in range(0,N):
##                print A[i,j]
#                if A[i,j] == 1:
#                    yiq += Ipi[q, j]
#            Yiq[q,i] = yiq
    
    
    # Only iterate through black pixels (this isn't really much faster...)
    # presumably np.where is the overhead of this approach
    
#    print 'Yiq block',
#    start = now()
    for i in range(0,N):
        for j in range(0,N):
            val = A[i,j]
#            if not val:
            if not val:
                for q in range(0, deg+1):
                    Yiq[q,i] += Ipi[q,j]#*val
#                     Yiq[i,q] += Ipi[q,j]#*val
#    print now() - start
            
#     print np.array(Yiq)
#     A = A*-1 + 1
#     print dgemm(1,A, Ipi.T) # <-- if use dgemm, need to invert A colors and transpose Ipi
#     import sys; sys.exit()
#    for k in range(0,IX):
#        i = ix[k]
#        j = iy[k]
#        for q in range(0, deg+1):
#        
#            Yiq[q,i] += Ipi[q,j]

    
#    print 'mpq block',
#    start = now()
#     cdef int mm = 18
#     cdef int nn = 18
#     cdef int kk = 32
#     cdef int lda = 18
#     cdef int ldb = 32
#     cdef int ldc = 18
#     cdef double alpha = 1.0
#     cdef double BETA = 0
#     for row in Mpqs:
#         for c in row:
#             print c
#     print 'what what'

#     print Yiq
#     print Ipi

#     blas.dgemm('N', 'N', &mm, &nn, &kk, &alpha, &Ipi[0,0],&lda, &Yiq[0,0], &ldb, &BETA, &Mpqs[0,0], &ldc)
#     blas.dgemm('N', 'N', &M, &NN, &K, &SC, &Ipi[0,0],&K, &Yiq[0,0], &K, &BETA, &Mpqs[0,0], &NN)
#     print Mpqs
#     print(np.asarray(Mpqs))
#     Mpqs = dgemm(1, Ipi, Yiq)
#     Mpqs = np.dot(Ipi, Yiq)
#     print np.array(Mpqs)
#     import sys; sys.exit()
    
    ####################3
    
    for p in range(0,deg+1): # Could replace this loop with DGEMM
            
        for q in range(0,deg+1):
                
            mpq = 0.0
##            yiq = 0.0
##            cur_row = ix[0]
##            for k in range(0,N):
##                i = ix[k]
##                j = iy[k]
##                Yiq[q,i] = Yiq[q,i] + Ipi[q,j]
###                next_row = ix[k+1]
###                yiq = yiq + Ipi[q,iy[k]]
##                Mpqs[p,q] = Mpqs[p,q] + Yiq[q,i]*Ipi[p,i]
##                if cur_row != next_row:
##                    yiq = 0.0
##                    mpq = 0.0
##                cur_row = next_row
##            yiq = yiq + Ipi[q,iy[k+1]]
##            mpq = mpq + yiq*Ipi[p,cur_row]
##            Mpqs[p,q] = mpq
#            
            for i in range(0,N):
#                yiq = 0.0
#
#                    
#                for j in range(0,N):
#                    yiq += Ipi[q, j]*A[i,j]
                        
#                mpq += yiq*Ipi[p,i]
                mpq += Yiq[q,i]*Ipi[p,i]
#                 Mpqs[p,q] += Yiq[q,i]*Ipi[p,i]
                  
            Mpqs[p,q] = mpq
    #############################33
#     print np.array(Mpqs)
#     import sys; sys.exit()
#            rpq = 0.0
#            diff = p-q
#            if diff%2 == 0:
#                S = diff//2
#            else:
#                continue
#            for j in range(S+1):
#                for m in range(q+1):
#                    rpq += (w**m)*D[S,j]*D[q,m]*Mpqs[p-2*j-m,2*j+m]
#    
#            Rpq[p,q] = rpq
    
#    print now()-start
    
#    print 'rpq block',
#    start = now()
    for p in range(0,deg+1):
        for q in range(0,deg+1):
            
            rpq = 0.0
            diff = p-q
            if diff%2 == 0:
                S = diff//2
            else:
                continue
            for j in range(S+1):
                for m in range(q+1):
#                    rpq = rpq + (w**m)*D[S,j]*D[q,m]*Mpqs[p-2*j-m,2*j+m]
                    rpq = rpq + ws[m%4]*D[S,j]*D[q,m]*Mpqs[p-2*j-m,2*j+m]
    
            Rpq[p,q] = rpq
#    print now()-start
#    
#    print 'zpq block',
#    start = now() 
    i = 0
#    Zpq = []
    for p in range(deg+1):
        pp1 = p + 1
        for q in range(p%2,pp1, 2):
            zpk = 0.0
            
            for k in range(q,pp1,2):
    
                zpk = zpk + Bpqk[p,q,k]*Rpq[k,q]
    
#            Zpq[p,q] = ((p+1)/PI)*zpk
#            Zpq.append(abs(pp1*o_PI*zpk))

            ### The absolute value gives us the magniture of the zernike moment.
            ### The zernike moment itself is pp1*o_PI*zpk and is complex
            Zpq[i] = abs(pp1*o_PI*zpk)
#            zpk = pp1*o_PI*zpk
#            zrl = zpk.real
#            zim = zpk.imag
#            Zpq[i] = pow(zrl*zrl + zim*zim, .5)
            i = i+1
#    print now()-start
#     return Zpq