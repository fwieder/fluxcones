# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:44:07 2022

@author: Frederik Wieder
"""

from flux_class_vecs import *

model = flux_cone("test",np.array([1,-1,-1]).reshape(1,3),np.array([0,1,0]))





model.get_efvs("cdd")
model.get_geometry()
print("S: \n", model.stoich)
print("I_irr: \n", model.nonegs)
print("")
print("gens: \n", model.generators)

print("")
print("")
print("")
model.split_rev(1)
model.get_efvs("cdd")
model.get_geometry()
print("S: \n", model.stoich)
print("I_irr: \n", model.nonegs)
print("")
print("gens: \n", model.generators)