# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 13:53:28 2021

@author: Frederik Wieder
"""

from flux_class_vecs import *
import numpy as np
from collections import Counter
import tqdm

model = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")

model.efvs = np.load("./e_coli_efvs.npy")
model.generators = np.load("e_coli_gens.npy")

efv_lens = [len(supp(efv)) for efv in model.efvs]
irr_lens = [len(model.irr_supp(efv)) for efv in model.efvs]
print(sorted(Counter(efv_lens).items()))
print(sorted(Counter(irr_lens).items()))
