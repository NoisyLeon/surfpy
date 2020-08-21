import numpy as np
from numba import jit, float32, int32, boolean, float64, int64
from numba import njit, prange
import numba 

def _bad_station_detector(inarr, thresh = 250.):
    latlst1 = inarr[:, 1]
    lonlst1 = inarr[:, 2]
    latlst2 = inarr[:, 3]
    lonlst2 = inarr[:, 4]
    res     = inarr[:, 7]
    Ndata   = inarr.shape[0]
    # get station lst
    stlas   = np.array([])
    stlos   = np.array([])
    # for i in range(Ndata):
    #     if i == 0:
    #         stlas       = np.append(stlas, latlst1[i])
    #         stlos       = np.append(stlos, lonlst1[i])
    #         continue
    #     if latlst1[i] != latlst1[i-1] or lonlst1[i] != lonlst1[i-1]:
    #         if np.any((stlas == latlst1[i])*(stlos == lonlst1[i])):
    #             continue
    #         stlas       = np.append(stlas, latlst1[i])
    #         stlos       = np.append(stlos, lonlst1[i])
    # Nsta                = stlas.size
    # ressum              = np.zeros(Nsta, dtype=np.float64)
    # Nsum                = np.zeros(Nsta, dtype=np.float64)
    # for i in range(Ndata):
    #     ind1            = (latlst1[i] == stlas)*(lonlst1[i] == stlos)
    #     ind2            = (latlst2[i] == stlas)*(lonlst2[i] == stlos)
    #     ressum[ind1]    += abs(res[i])
    #     ressum[ind2]    += abs(res[i])
    #     # ressum[ind1]    += res[i]
    #     # ressum[ind2]    += res[i]
    #     Nsum[ind1]      += 1
    #     Nsum[ind2]      += 1
    # avgres              = abs(ressum/Nsum)
    # absressum           = abs(ressum)
    validarr            = np.ones(Ndata, dtype=np.bool)
    # for i in range(Ndata):
    #     ind1            = (latlst1[i] == stlas)*(lonlst1[i] == stlos)
    #     ind2            = (latlst2[i] == stlas)*(lonlst2[i] == stlos)
    #     # if avgres[ind1] > thresh or avgres[ind2] > thresh:
    #     #     validarr[i] = False
    #     if absressum[ind1] > thresh or absressum[ind2] > thresh:
    #         validarr[i] = False
    
    for i in range(Ndata):
        # if (lonlst1[i] < 200. and (latlst1[i] > 62. and latlst1[i] < 68.)):
        #     validarr[i] = False
        # if (lonlst2[i] < 200. and (latlst2[i] > 62. and latlst2[i] < 68.)):
        #     validarr[i] = False
        # if (lonlst1[i] > 220. and (latlst1[i] < 65.)):
        #     validarr[i] = False
        # if (lonlst2[i] > 225. and (latlst2[i] < 65.)):
        #     validarr[i] = False
        if (lonlst1[i] < 205. and (latlst1[i] < 60.)):
            validarr[i] = False
        if (lonlst2[i] < 205. and (latlst2[i] < 60.)):
            validarr[i] = False
        # ind1            = (latlst1[i] == stlas)*(lonlst1[i] == stlos)
        # ind2            = (latlst2[i] == stlas)*(lonlst2[i] == stlos)
        
    
    return validarr

@njit(numba.types.Tuple((int64[:], float64[:], float64[:]))(float64[:], float64[:], float64[:, :]))
def _station_residual(stlas, stlos, residual):
    Ndat, Ncol  = residual.shape
    Nsta        = stlas.size
    Ncounts     = np.zeros(Nsta, dtype = np.int64)
    absres      = np.zeros(Nsta, dtype = np.float64)
    res         = np.zeros(Nsta, dtype = np.float64)
    for ista in range(Nsta):
        stlo    = stlos[ista]
        stla    = stlas[ista]
        if stlo < 0.:
            stlo    += 360.
        for i in range(Ndat):
            if (abs(residual[i, 1] - stla) < 0.01 and abs(residual[i, 2] - stlo)<0.01) \
                or (abs(residual[i, 3] - stla) < 0.01 and abs(residual[i, 4] - stlo)<0.01):
                Ncounts[ista]   += 1
                absres[ista]    += abs(residual[i, 7])
                res[ista]       += residual[i, 7] 
    return Ncounts, absres, res