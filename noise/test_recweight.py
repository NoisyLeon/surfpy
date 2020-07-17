import _xcorr_funcs
import numpy as np
import matplotlib.pyplot as plt
arr1 = np.zeros((1, 2), dtype=np.float32)
arr2 = np.zeros((1, 2), dtype=np.float32)
# 
# arr1[0, 1] = 86398
# arr2[0, 1] = 86398

# arr1[0, 1] = 0
# arr2[0, 1] = 0
# arr1[0, 1] = 84000
# arr2[0, 1] = 84000
# 
# weight = _xcorr_funcs._CalcRecCor(arr1, arr2, np.int32(3000))
# 
Narr = np.random.randint(low = 83000, high = 86400, size = 1)
N = 84001
for N in Narr:
    N = 84001
    tmp1 = np.random.uniform(0, 1, N)
    tmp2 = np.random.uniform(0, 1, N)
    mask1= tmp1< 0.01
    mask2= tmp2< 0.01
    mask1= np.zeros(N, dtype=bool)
    mask2= np.zeros(N, dtype=bool)
    # mask1[int(N/2):] = True
    # mask2[:int(N/2)] = True
    print (N)
    Nreclst1, Nrec1 = _xcorr_funcs._rec_lst(mask1)
    Nreclst2, Nrec2 = _xcorr_funcs._rec_lst(mask2)
    
    # # Nreclst1[:, 1] +=1
    # # Nreclst2[:, 1] +=1
    
    weight = _xcorr_funcs._CalcRecCor(np.int32(Nreclst1[:Nrec1]), np.int32(Nreclst2[:Nrec2]), np.int32(3000))
    
    weight2= np.correlate(np.logical_not(mask2).astype(np.int32), np.logical_not(mask1).astype(np.int32), mode='full')
    weight2= weight2[N-1-3000:N-1+3000+1]
    dw = weight - weight2
    print (np.sum(np.abs(dw)))
    # weight2
    
    
