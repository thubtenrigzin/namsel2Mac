#from mahotas.features import zernike_moments
import cPickle as pickle
from cv2 import GaussianBlur
from cv2 import HuMoments, moments, GaussianBlur
from fast_utils import fnormalize, scale_transform
import numpy as np
from sklearn.externals import joblib
from sklearn.mixture import GMM
from sobel_features import sobel_features
from transitions import transition_features
# from utils_extra.utils_extra import local_file
from zernike_moments import zernike_features
import os
from utils import local_file

SCALER_PATH = 'zernike_scaler-latest'
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(local_file(SCALER_PATH))
    transform = scaler.transform
    try:
        sc_o_std = 1.0/scaler.scale_
    except AttributeError:
        sc_o_std = 1.0/scaler.std_
    sc_mean = scaler.mean_
    SCALER_DEFINED = True
else:
    SCALER_DEFINED = False
    
FEAT_SIZE = 346

hstack = np.hstack

NORM_SIZE = 32
ARR_SHAPE = (NORM_SIZE, NORM_SIZE)
x3 = np.empty(NORM_SIZE*2, dtype=np.uint8)
newarr = np.empty(ARR_SHAPE, dtype=np.uint8)

magnitude = np.empty(ARR_SHAPE, np.double)
direction = np.empty(ARR_SHAPE, np.double)
sx = np.empty(ARR_SHAPE, np.double)
sy = np.empty(ARR_SHAPE, np.double)
x2 = np.zeros((192), np.int)

D = pickle.load(open(local_file('features/D_matrix.pkl'),'r'))
Bpqk = pickle.load(open(local_file('features/Bpqk17.pkl'), 'rb'))
Ipi = pickle.load(open(local_file('features/Ipi32.pkl'),'rb'))
Ipi = np.array(Ipi, Ipi.dtype, order='F')
deg = 17
# Mpqs = np.empty((deg+1,deg+1), np.double)
Mpqs = np.zeros((deg+1,deg+1), np.double, order = 'F')
Rpq = np.empty((deg+1,deg+1), complex)
ws = np.array([1,-1j,-1,1j], complex)
Zpq = np.empty((90), np.double)
Yiq = np.zeros((deg+1,NORM_SIZE), np.double, order='F')


def normalize_and_extract_features(arr):
    global newarr, x3, Zpq
    newarr = newarr.astype(np.uint8)
    fnormalize(arr, newarr)
    return extract_features(newarr)


def extract_features(arr, scale=True):
    global x3, Zpq
    transition_features(arr, x3)
    arr = arr.astype(np.double)
    Yiq.fill(0.0)
    zernike_features(arr,D,Bpqk,Ipi, Mpqs, Rpq, Yiq, ws, Zpq)
    GaussianBlur(arr, ksize=(5,5), sigmaX=1, dst=newarr)
    x2.fill(0)
    sobel_features(arr, magnitude, direction, sx, sy, x2)
    x1 = hstack((Zpq,x2, x3))
    if scale:
        if not SCALER_DEFINED:
            raise ValueError, 'Scaler not defined'
        scale_transform(x1, sc_mean, sc_o_std, FEAT_SIZE)
    return x1


def invert_binary_image(arr):
    '''
    Invert a binary image so that zero-pixels are considered as background.
    This is assumed by various functions in OpenCV and other libraries.
    
    Parameters:
    -----------
    arr: 2D numpy array containing only 1s and 0s
    
    Returns:
    --------
    2d inverted array
    '''
    if not hasattr(arr, 'dtype'):
        arr = np.array(arr, np.uint8)
    
    if np.max(arr) == 255:
        return (arr / -255) + 1
    else:
        return (arr * -1) + 1
   

def get_zernike_moments(arr):

    if arr.shape != (32, 32):
        arr.shape = (32, 32)
    
    zernike_features(arr,D,Bpqk,Ipi, Mpqs, Rpq, Yiq, ws, Zpq)
    return Zpq

def get_hu_moments(arr):
    arr = invert_binary_image(arr)
    if arr.shape != (32, 32):
        arr.shape = (32, 32)
    m = moments(arr.astype(np.float64), binaryImage=True)
    hu = HuMoments(m)
    return hu.flatten()
