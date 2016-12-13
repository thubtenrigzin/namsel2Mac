
from __future__ import division
# import numpy as np
from cv2 import Sobel
cimport numpy as np
cimport cython
from libc.math cimport sqrt, atan2, fmod, M_PI
# from cython.parallel import prange, parallel
ctypedef np.int_t DTYPE_t

# PI = np.pi

# @cython.boundscheck(False)
# def invert(np.ndarray[double, ndim=2] a):
#     cdef int M = a.shape[0]
#     cdef int N = a.shape[1]
#     for i in range(M):
#         for j in range(N):
#             a[i,j] = -1.0*a[i,j]+1.0
#     return a


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef sobel_features(np.ndarray[double, ndim=2] a,
                     np.ndarray[double, ndim=2] magnitude,
                     np.ndarray[double, ndim=2] direction,
                     np.ndarray[double, ndim=2] sx,
                     np.ndarray[double, ndim=2] sy,
                     np.ndarray[DTYPE_t] vector
                     ):
#     cdef int imgh = a.shape[0]
#     cdef int imgw = a.shape[1]
    cdef int imgh = 32
    cdef int imgw = 32
    cdef double o_tsize = 1.0/(imgh*imgw)
    cdef int ix0, ix1, i, ftmat_len, j, sec, cls, cls1, cls2, l
    cdef double rbar, msum, curdeg, dx, dy, mg
    cdef double d_30 = 1.0 / 30.0
#     cdef double d_PI = 1.0 / np.pi
    cdef double degc = 180.0 / M_PI
#     cdef np.ndarray[double, ndim=2] sx
#     cdef np.ndarray[double, ndim=2] sy
#     cdef np.ndarray[double, ndim=2] magnitude = np.empty((imgh, imgw), np.double)
#     cdef np.ndarray[double, ndim=2] direction = np.empty((imgh, imgw), np.double)
#     cdef np.ndarray[double, ndim=2] sx = np.empty((imgh, imgw), np.double)
#     cdef np.ndarray[double, ndim=2] sy = np.empty((imgh, imgw), np.double)
    
#     cdef np.ndarray[DTYPE_t] vector = np.zeros((192), np.int)
#    cdef np.ndarray[DTYPE_t] feature_mat, ix, iy
#    cdef np.ndarray[double] degrees

#     a = -1.0*a+1.0 # invert the image
    
#     for i in range(imgh):
#         for j in range(imgw):
#             a[i,j] = -1.0*a[i,j]+1.0
    

    Sobel(a, dst=sx, ddepth=-1, dx=1, dy=0, ksize=3)
    Sobel(a, dst=sy, ddepth=-1, dx=0, dy=1, ksize=3)

    
    for i in range(imgh):
        for j in range(imgw):
            dx = sx[i,j]
            dy = sy[i,j]
            mg = sqrt(dx*dx + dy*dy)
            magnitude[i,j] = mg
            msum += mg
            direction[i,j] = atan2(dy,dx)
            

    
    rbar = msum * o_tsize
    for k in range(imgh):
        for l in range(imgh):
            if magnitude[k,l] >= rbar:
                curdeg =  direction[k,l] * degc
                ix0 = k
                ix1 = l
                if ix1 < 8:
                    j = 0
                elif 8 <= ix1 < 16:
                    j = 1
                elif 16 <= ix1 < 24:
                    j = 2
                else:
                    j = 3
                
                if ix0 < 8:
                    sec = 0 + j
                elif 8 <= ix0 < 16:
                    sec = 4 + j
                elif 16 <= ix0 < 24:
                    sec = 8 + j
                else:
                    sec = 12 +j

#                 if curdeg % 30.0 != 0.0:
                if fmod(curdeg, 30.0) != 0.0:
                    if 0.0 < curdeg < 30.0:
                        cls1 = 0
                        cls2 = 1
                    elif 30.0 < curdeg < 60.0:
                        cls1 = 1
                        cls2 = 2
                    elif 60.0 < curdeg < 90.0:
                        cls1 = 2
                        cls2 = 3
                    elif 90.0 < curdeg < 120.0:
                        cls1 = 3
                        cls2 = 4
                    elif 120.0 < curdeg < 150.0:
                        cls1 = 4
                        cls2 = 5
                    elif 150.0 < curdeg < 180.0:
                        cls1 = 5
                        cls2 = 6
        
                    elif -30.0 < curdeg < 0.0:
                        cls1 = 0
                        cls2 = 11
                    elif -60.0 < curdeg < -30.0:
                        cls1 = 11
                        cls2 = 10
                    elif -90.0 < curdeg < -60.0:
                        cls1 = 10
                        cls2 = 9
                    elif -120.0 < curdeg < -90.0:
                        cls1 = 9
                        cls2 = 8
                    elif -150.0 < curdeg < -120.0:
                        cls1 = 8
                        cls2 = 7
                    elif -180.0 < curdeg < -150.0:
                        cls1 = 7
                        cls2 = 6
                    
                    sec = sec * 12
#                     vector[sec+cls1] = vector[sec+cls1] + 1
#                     vector[sec+cls2] = vector[sec+cls2] + 1
                    vector[sec+cls1] += 1
                    vector[sec+cls2] += 1
                else:
                    cls = int(curdeg * d_30)
                    sec = sec*12+cls
                    vector[sec] += 1 

###
#        cls = feature_mat[i] - 1
#        
#        if cls % 2 == 1:
#            if cls == 7:
#                cls2 = 0
#            else:
#                cls2 = (cls + 1) / 2
#            
#            cls1 = (cls - 1) / 2
#            vector[sec*4+cls1] = vector[sec*4+cls1] + 1
#            vector[sec*4+cls2] = vector[sec*4+cls2] + 1
#        else:
#        
##        sec = sec*(bin_num-1) + cls - 2
#            sec = sec*4 + cls / 2
            

    #        print cls
    #        print sec
#                    vector[sec] = vector[sec] + 1
    

#     return vector
    
    
    
    
    
    
    
    
    ##################################333
#    inx = np.where(magnitude >= rbar)
#    ix = inx[0]
#    iy = inx[1]
#    degrees = np.degrees(direction[inx])
#    feature_mat = np.digitize(degrees, bins)
#   
##    print len(feature_mat), feature_mat.shape[0]
#    ftmat_len = feature_mat.shape[0]
##    with nogil, parallel():
#    for i in range(ftmat_len):
#        ix0 = ix[i]
#        ix1 = iy[i]
##        j = ix1 >= 8
##        j = ix1 >= 16
#        
#        #split into sections horiz
##        if ix1 < 5:
##            j = 0
##        elif 5 <= ix1 < 11:
##            j = 1
##        else:
##            j = 2
#        if ix1 < 8:
#            j = 0
#        elif 8 <= ix1 < 16:
#            j = 1
#        elif 16 <= ix1 < 24:
#            j = 2
#        else:
#            j = 3
#        
#        if ix0 < 8:
#            sec = 0 + j
#        elif 8 <= ix0 < 16:
#            sec = 2 + j
#        elif 16 <= ix0 < 24:
#            sec = 4 + j
#        else:
#            sec = 6 +j
##        cls = feature_mat[i]
##        sec = sec*8 + cls - 2
##        if ix0 < 10:
##            sec = 0 + j
##        elif 10 <= ix0 < 22:
##            sec = 2 + j
##        else:
##            sec = 4 +j
#
#        curdeg = degrees[i]
#        if curdeg % 30.0 != 0.0:
#            if 0.0 < curdeg < 30.0:
#                cls1 = 0
#                cls2 = 1
#            elif 30.0 < curdeg < 60.0:
#                cls1 = 1
#                cls2 = 2
#            elif 60.0 < curdeg < 90.0:
#                cls1 = 2
#                cls2 = 3
#            elif 90.0 < curdeg < 120.0:
#                cls1 = 3
#                cls2 = 4
#            elif 120.0 < curdeg < 150.0:
#                cls1 = 4
#                cls2 = 5
#            elif 150.0 < curdeg < 180.0:
#                cls1 = 5
#                cls2 = 6
#
#            elif -30.0 < curdeg < 0.0:
#                cls1 = 0
#                cls2 = 11
#            elif -60.0 < curdeg < -30.0:
#                cls1 = 11
#                cls2 = 10
#            elif -90.0 < curdeg < -60.0:
#                cls1 = 10
#                cls2 = 9
#            elif -120.0 < curdeg < -90.0:
#                cls1 = 9
#                cls2 = 8
#            elif -150.0 < curdeg < -120.0:
#                cls1 = 8
#                cls2 = 7
#            elif -180.0 < curdeg < -150.0:
#                cls1 = 7
#                cls2 = 6
#            
#            sec = sec * 12
#            vector[sec+cls1] = vector[sec+cls1] + 1
#            vector[sec+cls2] = vector[sec+cls2] + 1
#        else:
#            cls = int(curdeg * d_30)
#            sec = sec*12+cls
#            vector[sec] = vector[sec] + 1 
#
####
##        cls = feature_mat[i] - 1
##        
##        if cls % 2 == 1:
##            if cls == 7:
##                cls2 = 0
##            else:
##                cls2 = (cls + 1) / 2
##            
##            cls1 = (cls - 1) / 2
##            vector[sec*4+cls1] = vector[sec*4+cls1] + 1
##            vector[sec*4+cls2] = vector[sec*4+cls2] + 1
##        else:
##        
###        sec = sec*(bin_num-1) + cls - 2
##            sec = sec*4 + cls / 2
#            
#
#    #        print cls
#    #        print sec
#            vector[sec] = vector[sec] + 1
#    
#
#    return vector