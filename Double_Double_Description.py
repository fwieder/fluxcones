# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 08:49:49 2021

@author: Frederik Wieder
"""

from flux_class_vecs import *

import cdd
import numpy as np

model = flux_cone.from_kegg("./Biomodels/small_examples/illusnet/illusnet")

mat = cdd.Matrix(model.nonegs, number_type='fraction')
mat.rep_type = cdd.RepType.INEQUALITY
mat.extend(model.stoich, linear=True)
poly = cdd.Polyhedron(mat)
gen = poly.get_generators()
print(gen)

gens = np.array(gen)
print(gens)

mat2 = cdd.Matrix(gens[[0,1,2]], number_type = 'fraction')
mat2.rep_type = cdd.RepType.GENERATOR
mat2.extend(gens[[3,4]], linear = True)
print(mat2)
poly2 = cdd.Polyhedron(mat2)
ineqs2 = poly2.get_inequalities()
print(ineqs2)