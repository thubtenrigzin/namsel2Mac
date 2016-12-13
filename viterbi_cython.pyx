from libc.math cimport exp, log
import numpy as np
# from cython.parallel import prange, parallel
cimport numpy as np
cimport cython
np.import_array()

ctypedef np.float32_t dtype_t
# 
# cdef dtype_t _NINF = -np.inf
cdef np.float32_t _NINF = np.float32(-np.inf)


# @cython.boundscheck(False)
# cdef dtype_t _max(dtype_t[:] values):
#     # find maximum value (builtin 'max' is unrolled for speed)
#     cdef dtype_t value
#     cdef dtype_t vmax = _NINF
#     for i in range(values.shape[0]):
#         value = values[i]
#         if value > vmax:
#             vmax = value
#     return vmax



# @cython.boundscheck(False)
# cpdef dtype_t _logsum(dtype_t[:] X):
#     cdef dtype_t vmax = _max(X)
#     cdef dtype_t power_sum = 0
# 
#     for i in range(X.shape[0]):
#         power_sum += exp(X[i]-vmax)
# 
#     return log(power_sum) + vmax

@cython.boundscheck(False)
cdef int argmax(np.ndarray[float] a):
    cdef int i, maxi
    cdef int h = a.shape[0]
    cdef float mx = -np.inf
     
    for i in range(h):
        if a[i] > mx:
            mx = a[i]
            maxi = i
     
    return maxi

@cython.boundscheck(False)
cpdef viterbi_cython(int n_observations, int n_components,
        np.ndarray[np.float32_t, ndim=1] log_startprob,
        np.ndarray[np.float32_t, ndim=2] log_transmatT,
        np.ndarray[np.float32_t, ndim=2] framelogprob
#         dtype_t [:] log_startprob,
#         dtype_t [:, :] log_transmatT,
#         dtype_t [:, :] framelogprob
        ):
 
    cdef int t, max_pos
    cdef np.float32_t mx = _NINF
    cdef int i, j
    cdef np.ndarray[double, ndim = 2] viterbi_lattice = _NINF*np.ones((n_observations, n_components))
#     cdef np.ndarray[np.float_t, ndim = 1] row
    cdef np.ndarray[np.int_t, ndim = 1] state_sequence = np.empty(n_observations, dtype=np.int)
    cdef dtype_t logprob
#     cdef np.ndarray[dtype_t, ndim = 2] work_buffer
#     cdef np.ndarray[np.float32_t, ndim = 2] log_transmatT = np.transpose(log_transmat).copy()
#     cdef np.float32_t [:,:] log_transmatT = log_transmat.T
#     cdef np.ndarray[np.float_t, ndim = 1] row
    
    cdef float NINF = -np.inf
    cdef float max_row, val, lt, lprb
    cdef float thresh = log(.0000001)
#     cdef float thresh = np.log(.001)
    # Initialization
#     state_sequence = np.empty(n_observations, dtype=np.int)
#     viterbi_lattice = np.zeros((n_observations, n_components))
#     viterbi_lattice[0] = log_startprob + framelogprob[0]
    
#     with nogil, parallel(): 
    for i in range(n_components):
        viterbi_lattice[0,i] = log_startprob[i] + framelogprob[0,i]
     
        # Induction
    #     for t in range(1, n_observations):
    #         work_buffer = viterbi_lattice[t-1] + log_transmat.T
    #         viterbi_lattice[t] = np.max(work_buffer, axis=1) + framelogprob[t]
     
        #Induction with explicit loops
    for t in range(1, n_observations):
        for i in range(n_components):
            max_row = NINF
            lprb = framelogprob[t, i]
            if lprb < thresh:
                continue
            for j in range(n_components):
                val = viterbi_lattice[t-1, j] + log_transmatT[i, j]
                if val > max_row:
                    max_row = val
            viterbi_lattice[t,i] = max_row + lprb
     
    # Observation traceback
#     max_pos = argmax(viterbi_lattice[n_observations - 1, :])
#     max_pos = argmax(viterbi_lattice[n_observations - 1])
    

    for i in range(n_components):
        lt = viterbi_lattice[n_observations - 1, i]
        if lt > mx:
            mx = lt
            max_pos = i

    state_sequence[n_observations - 1] = max_pos
    logprob = viterbi_lattice[n_observations - 1, max_pos]
 
    for t in range(n_observations - 2, -1, -1):
        max_row = NINF
        for j in range(n_components):
            val = viterbi_lattice[t, j] + log_transmatT[state_sequence[t + 1], j]
            if val > max_row:
                max_row = val
                max_pos = j
#         max_pos = argmax(viterbi_lattice[t] + log_transmat[:, state_sequence[t + 1]])
        state_sequence[t] = max_pos
 
    return logprob, state_sequence

# @cython.boundscheck(False)
# def viterbi_no_startprob(int n_observations, int n_components,
#         np.ndarray[dtype_t, ndim=2] log_transmat,
#         np.ndarray[dtype_t, ndim=2] framelogprob):
# 
#     cdef int t, max_pos
#     cdef np.ndarray[dtype_t, ndim = 2] viterbi_lattice
#     cdef np.ndarray[np.int_t, ndim = 1] state_sequence
#     cdef dtype_t logprob
#     cdef np.ndarray[dtype_t, ndim = 2] work_buffer
# 
#     # Initialization
#     state_sequence = np.empty(n_observations, dtype=np.int)
#     viterbi_lattice = np.zeros((n_observations, n_components))
#     viterbi_lattice[0] = framelogprob[0]
# 
#     # Induction
#     for t in range(1, n_observations):
#         work_buffer = viterbi_lattice[t-1] + log_transmat.T
#         viterbi_lattice[t] = np.max(work_buffer, axis=1) + framelogprob[t]
# 
#     # Observation traceback
#     max_pos = np.argmax(viterbi_lattice[n_observations - 1, :])
#     state_sequence[n_observations - 1] = max_pos
#     logprob = viterbi_lattice[n_observations - 1, max_pos]
# 
#     for t in range(n_observations - 2, -1, -1):
#         max_pos = np.argmax(viterbi_lattice[t, :] \
#                 + log_transmat[:, state_sequence[t + 1]])
#         state_sequence[t] = max_pos
# 
#     return logprob, state_sequence