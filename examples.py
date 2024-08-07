import numpy as np
import tqdm
from itertools import product, combinations
from collections import Counter

from fluxcones import FluxCone
from fluxcones.helpers import supp
from fluxcones.algorithms import check_conjecture


def all_networks(num_metabs, num_reacs, value_list):
    cols = [np.array(col) for col in product(value_list, repeat=num_metabs)]
    stoichs = [
        np.array(i).reshape(num_metabs, num_reacs)
        for i in combinations(cols, num_reacs)
    ]
    revs = [
        np.array(i)
        for i in product([0, 1], repeat=num_reacs)
        if len(supp(np.array(i))) > 1
    ]
    model_ids = [(i, j) for i, j in product(range(len(stoichs)), range(len(revs)))]

    print(len(model_ids), "models")

    data = []
    degrees = []
    for model_id in tqdm.tqdm(model_ids):
        model = FluxCone(stoichs[model_id[0]], revs[model_id[1]])
        efms = model.get_efms_cdd()
        res = check_conjecture(model, efms)
        if res == False:
            print("Conjecture disproven!")
            break
        data.append(len(efms))
        for efm in efms:
            degrees.append(model.degree(efm))
    print(" ")
    print(np.count_nonzero(data), "models with EFMs")
    print("largest number of EFMs = ", max(data))
    print(sum(data), "EFMs in total")
    print(Counter(data))
    print(Counter(degrees))
    return data

if __name__ == "__main__":
    data = all_networks(num_metabs=2, num_reacs=9, value_list=[-1, 0, 1])
    
