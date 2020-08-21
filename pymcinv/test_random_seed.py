#from numpy import random
import random
import numba
import numpy as np
from multiprocessing import Process

# @numba.njit()
# def printrand(number):
#     # rndn = np.random.random()
#     rndn = random.random()
#     print ("n:",number,"r:",rndn)
#     return 
#     # print ("n:",number,"r:",rndn)
#  
# if __name__=='__main__':
#     num = 1
#     p_list = []
#     for proc in range(10):
#         p_list.append(Process(target=printrand, args=(proc,)))
#         p_list[-1].start()
        
# @numba.njit()
def test():
    j   = 0
    while (j < 1000000):
        newval      = random.gauss(0., 2.)
        j           += 1
        # print (newval)