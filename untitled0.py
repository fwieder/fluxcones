# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 15:19:11 2022

@author: Frederik Wieder
"""

from flux_class_vecs import supp, flux_cone
import numpy as np

S = np.array([[1,-1,0,0,0,0,0,0,0,0,0,0],
             [0,-1,0,0,1,1,-1,0,0,0,0,0],
             [0,0,0,0,0,-1,0,0,0,0,1,-1],
             [0,1,-1,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,1,-1,0,0,0,0],
             [0,0,0,0,0,0,0,-1,0,0,0,1],
             [0,0,1,1,0,0,0,0,0,1,0,0],
             [0,0,0,0,0,0,0,1,1,-1,0,0]]
             )

rev = np.array([0,0,0,1,0,0,0,0,1,1,0,0])

model = flux_cone("Exercise", S, rev)
