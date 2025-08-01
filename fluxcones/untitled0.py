#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 10:25:17 2025

@author: frederik
"""

from fluxcones import FluxCone
from fluxcones.helpers import supp
from fluxcones.algorithms import two_gens
import json
import numpy as np


model = FluxCone.from_sbml("/Users/frederik/Downloads/iAF1260_m9.xml")

with open("/Users/frederik/Downloads/RAC_1.json","r") as f:
    rac = json.load(f)
    
rac_keys = list(rac.keys())

adjusted_rac = rac.copy()

for item in rac:
    if item[:4] == "REV_":
        del adjusted_rac[item]
        try:
            adjusted_rac[item[4:]] += rac[item]
        except:
            adjusted_rac[item[4:]] = rac[item]


rac_vec = np.zeros(len(model.rev))
for index,rea in enumerate(model.cobra.reactions):
    if rea.id in adjusted_rac:
        rac_vec[index] = adjusted_rac[rea.id]

threshold = .9

v = rac_vec.copy()
v[v< threshold*max(rac_vec)] = 0

print("v is steady state", model.is_in(v))
"""
face = model.face_defined_by(v)
if __name__ == "__main__":
    face_efms = face.get_efms_efmtool()

    print(len(face_efms))
"""