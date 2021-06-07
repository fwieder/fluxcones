# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:56:49 2021

@author: Frederik Wieder
"""

from flux_class_vecs import flux_cone,get_efvs
import numpy as np
import cdd , sys
import time
import random
from collections import Counter
import matplotlib.pyplot as plt

#model = flux_cone.from_sbml("./Biomodels/bigg/iAB_RBC_283.xml")



model = flux_cone.from_sbml("./Biomodels/bigg/e_coli_core.xml")
#model.delete_reaction(12)


#model = flux_cone.from_kegg("./Biomodels/kegg/Butanoate/kegg65")


if __name__ == "__main__":
    model.get_efms_in_all_2faces()
    model.get_efvs_in_mmbs()
    irr = np.nonzero(model.irr)[0]
    mmb_efms = []
    for mmb in model.mmb_efvs:
        mmb_efms.append(np.nonzero(np.round(mmb[0],5)))
    
    mmb_efms = [tuple(l[0].tolist()) for l in mmb_efms]
    
    face2_efms = model.face2_efms.tolist()
    face2_efms = list(set(tuple(i) for i in face2_efms) - set(tuple(i) for i in mmb_efms))
    
    print(len(face2_efms), "efms in 2 faces, excluding 1-faces")
    print(len(mmb_efms), "efms in 1-faces")
    
    
    
    mmb_efm_lens = [len(efm) for efm in mmb_efms]
    c1 = Counter(mmb_efm_lens)
    print("cards of supps of efms in 1-faces", dict(sorted(c1.items(),key=lambda i:i[0])))
    
    mmb_efm_irr_lens = [len(np.intersect1d(efm,irr)) for efm in mmb_efms]
    i1 = Counter(mmb_efm_irr_lens)
    print("cards of irr_reacs in efms in 1-faces", dict(sorted(i1.items(),key=lambda i:i[0])))
    
    face2_efm_lens = [len(efm) for efm in face2_efms]
    c2 = Counter(face2_efm_lens)
    print("cards of supps of efms in 2-faces", dict(sorted(c2.items(),key=lambda i:i[0])))
    
    
    face2_efm_irr_lens = [len(np.intersect1d(efm,irr)) for efm in face2_efms]
    i2 = Counter(face2_efm_irr_lens)
    print("cards of irr-reacs of efms in 2-faces", dict(sorted(i2.items(),key=lambda i:i[0])))
    

    model.get_efvs("efmtool")
    efms = [np.nonzero(np.round(efv,5))[0] for efv in model.efvs]
    efm_lens = [len(efm) for efm in efms]
    c3 = Counter(efm_lens)
    print("cards of supps of all efms", dict(sorted(c3.items(),key=lambda i: i[0])))
    
    efm_irr_lens = [len(np.intersect1d(efm,irr)) for efm in efms]
    i3 = Counter(efm_irr_lens)
    print("card of irr_reacs in all efms", dict(sorted(i3.items(),key=lambda i:i[0])))
    
    
    
    
    fig1,axs1 = plt.subplots(3,1,constrained_layout = True)
    fig1.suptitle("Occuriencies of cardinalities of supports")
    axs1[0].bar(c1.keys(),c1.values())
    axs1[1].bar(c2.keys(),c2.values())
    axs1[2].bar(c3.keys(),c3.values())
    axs1[0].set_title("1-faces");
    axs1[1].set_title("2-faces");
    axs1[2].set_title("all efms")
    
    plt.show()
    
    fig2,axs2 = plt.subplots(3,1,constrained_layout = True)
    fig2.suptitle("Occuriencies of cardinalities of irreversible reactions")
    axs2[0].bar(i1.keys(),i1.values())
    axs2[1].bar(i2.keys(),i2.values())
    axs2[2].bar(i3.keys(),i3.values())
    axs2[0].set_title("1-faces");
    axs2[1].set_title("2-faces");
    axs2[2].set_title("all efms")
    
    plt.show()