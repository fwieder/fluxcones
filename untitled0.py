# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:56:49 2021

@author: Frederik Wieder
"""

from flux_class_vecs import flux_cone
import numpy as np
import cdd , sys

model1 = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")
model1.stoich = np.delete(model1.stoich,12,axis = 1)
model1.rev = np.delete(model1.rev, 12)
model1.irr = np.delete(model1.irr, 12)


model2 = flux_cone.from_kegg("./Biomodels/kegg/Butanoate/kegg65")


'''
model.stoich = np.array([[-1,1,1]])
model.rev = np.array([0,0,1])
model.irr = np.array([1,1,0])
'''
if __name__ == "__main__":
    print(np.shape(model1.get_efvs()))
    print(np.shape(model1.get_efvs_in_mmbs()))
    print(np.shape(model1.get_mmbs()))
    print(model1.get_cone_dim())
    
    print(np.shape(model2.get_efvs()))
    print(np.shape(model2.get_efvs_in_mmbs()))
    print(np.shape(model2.get_mmbs()))
    print(model2.get_cone_dim())
    
    sys.exit()
    nonegs = np.eye(len(model.rev))[np.nonzero(model.irr)[0]]

    mat = cdd.Matrix(nonegs)
    mat.extend(model.stoich, linear = True)
    
    poly = cdd.Polyhedron(mat)
    
    gens = poly.get_generators()
    print(gens)
    mat2 = cdd.Matrix(gens)
    mat2.rep_type = cdd.RepType.GENERATOR
    
    poly2 = cdd.Polyhedron(mat2)
    print(np.shape(gens))
    ineqs = np.round(poly2.get_inequalities(),5)
    
    print(np.shape(ineqs))
    print(ineqs)