# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 22:20:56 2024

@author: fred
"""

import numpy as np
from fluxcones import FluxCone
from fluxcones.algorithms import *

S = np.array([[ 1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
              [ 0.,  1.,  1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
              [ 0.,  1.,  0., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
              [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  1., -1.,  0., -1.,  0.],
              [ 0.,  0.,  0.,  0.,  0.,  1., -1., -1.,  0.,  0.,  0.,  0.],
              [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.],
              [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1., -1.]])
rev = np.array([1,0,1,1,1,0,0,0,1,1,1,1])

model = FluxCone(S,rev)
# model.get_efms("cdd")

# F = model.face_defined_by(model.efms[20])

# F.get_efms("cdd")

# print(F.efms)

# import sys

# sys.exit(0)

import time
if __name__ == "__main__":
    efms = model.get_efms_cdd()
    print(len(efms))
    for efm in efms:
        a = two_gens(efm,efms,model,face_candidates=True, all_pairs=True)
        b = two_gens(efm,efms,model,face_candidates=False, all_pairs=True)
        print(len(a),len(b))
        assert np.array_equal(a,b)
    import sys
    sys.exit()
    start_time = time.perf_counter()
    efms = model.get_efms_efmtool()
    end_time = time.perf_counter()
    
    print("efmtool", end_time-start_time)
    print(len(efms))
    
    print(two_gens(efms[0]))
    
    start_time = time.perf_counter()
    model.get_efms_cdd(True)
    end_time = time.perf_counter()
    
    print("cdd", end_time-start_time)
    print(len(model.efms))
    
    start_time = time.perf_counter()
    model.get_efms_milp(True)
    end_time = time.perf_counter()
    
    print("milp", end_time-start_time)
    print(len(model.efms))