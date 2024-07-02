# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 22:20:56 2024

@author: fred
"""

import numpy as np
from flux_cones import *
from algorithms import *

S = np.array([[ 1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
              [ 0.,  1.,  1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
              [ 0.,  1.,  0., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
              [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  1., -1.,  0., -1.,  0.],
              [ 0.,  0.,  0.,  0.,  0.,  1., -1., -1.,  0.,  0.,  0.,  0.],
              [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.],
              [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.]])
rev = np.array([1,0,1,1,1,0,0,0,1,1,1,1])

model = flux_cone(S,rev)





import time
if __name__ == "__main__":
    start_time = time.perf_counter()
    model.get_rev_efms("efmtool")
    end_time = time.perf_counter()
    
    print("efmtool", end_time-start_time)
    
    
    start_time = time.perf_counter()
    model.get_rev_efms("cdd")
    end_time = time.perf_counter()
    
    print("cdd", end_time-start_time)
    
    
    start_time = time.perf_counter()
    model.get_rev_efms("milp")
    end_time = time.perf_counter()
    
    print("milp", end_time-start_time)
