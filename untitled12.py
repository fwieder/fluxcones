# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 16:12:59 2022

@author: Frederik Wieder
"""

from flux_class_vecs import flux_cone,supp
    
import numpy as np

S = np.array(((1,-1,0,0),(0,1,1,-1)))

rev = np.array([1,1,0,0])

model = flux_cone("Example", S, rev)

efvs = model.get_efvs("cdd")

efv_degs = [model.check_dim(efv) for efv in model.efvs]
S_irr = model.stoich[:,supp(model.irr)]
print("efv_degs:")
print(efv_degs)
bound = model.get_lin_dim() + np.linalg.matrix_rank(S_irr)+1
print("bound=", bound)
