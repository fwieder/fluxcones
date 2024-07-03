# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:58:32 2024

@author: fred
"""
# import needed packages

import numpy as np
import efmtool
import cdd
import cobra
from scipy.optimize import linprog
from helper_functions import *
import copy
import mip


class flux_cone:

    def __init__(self, stoichiometry, reversibility):
        """
        Parameters
        ----------
        stoichiometry : np.array
            Stoichcometry matrix
        reversibility : np.array
            Reversibility {0,1}-vector

        """

        # stote size of stoichiometric matrix
        self.num_metabs, self.num_reacs = np.shape(stoichiometry)

        self.stoich = stoichiometry

        self.rev = reversibility

        # self.irr only depends on self.rev
        self.irr = (np.ones(self.num_reacs) - self.rev).astype(int)

        # non-negativity constraints defined by v_irr >= 0
        nonegs = np.eye(self.num_reacs)[supp(self.irr)]

        # outer description of the flux cone by C = { x | Sx >= 0}
        self.S = np.r_[self.stoich, nonegs]

    ''' create the fluxcone as flux_cone.from_sbml to use an sbml file as input '''

    @classmethod
    def from_sbml(cls, path_to_sbml):

        # read sbml-file
        sbml_model = cobra.io.read_sbml_model(path_to_sbml)

        # extract stoichiometric matrix
        stoich = cobra.util.array.create_stoichiometric_matrix(sbml_model)

        # extract reversibility vector
        rev = np.array(
            [rea.reversibility for rea in sbml_model.reactions]).astype(int)

        # initialize class object from extracted parameters
        return cls(stoich, rev)


#################################################################################################################################################
# Callable methods for flux_cone objects:
#################################################################################################################################################

    ''' compute the dimension of the lineality space of the cone '''

    def get_lin_dim(self):
        lin_dim = len(supp(self.rev)) - \
            np.linalg.matrix_rank(self.stoich[:, supp(self.rev)])
        self.lin_dim = lin_dim
        return(lin_dim)

    ''' get_cone_dim might not work if description of model contains reduandancies'''

    def get_cone_dim(self):
        cone_dim = self.num_reacs - np.linalg.matrix_rank(self.stoich)
        return(cone_dim)

    ''' test whether a given np.array is a steady-state fluxvector of the flux_cone instance'''

    def is_in(self, vec):
        # test whether v_irr >= 0
        if len(vec[self.irr_supp(vec, tol)]) > 0:
            if min(vec[self.irr_supp(vec, tol)]) < 0:
                print(
                    "Not in cone, because there is an irreversible reaction with negative flux")
                return False
        # test whether S*v = 0
        if all(supp(np.dot(self.stoich, vec), tol) == np.array([])):
            return True

        else:
            print("S*v not equal to 0")
            return False

    ''' test whether a given np.array is an EFM of the flux_cone instance by applying the rank test'''

    def is_efm(self, vector):
        # 0 is not an EFM by defintion
        if len(supp(vector)) == 0:
            return False

        # rank test
        if np.linalg.matrix_rank(self.stoich[:, supp(vector)]) == len(supp(vector)) - 1:
            return True

        return False

    def get_efms_efmtool(self):
        """
        initiate reaction names and metabolite names from 0 to n resp. m because
        efmtool needs these lists of strings as input
        "normalize options:  [max, min, norm2, squared, none]
        """

        opts = dict({
            "kind": "stoichiometry",
            "arithmetic": "double",
            "zero": "1e-10",
            "compression": "default",
            "log": "console",
            "level": "OFF",
            "maxthreads": "-1",
            "normalize": "max",
            "adjacency-method": "pattern-tree-minzero",
            "rowordering": "MostZerosOrAbsLexMin"
        })

        if only_reversible:
            S = np.r_[self.stoich, np.eye(self.num_reacs)[supp(self.irr)]]
        else:
            S = self.stoich

        reaction_names = list(np.arange(len(S[0])).astype(str))
        metabolite_names = list(np.arange(len(S)).astype(str))

        efms_cols = efmtool.calculate_efms(
            S, self.rev, reaction_names, metabolite_names, opts)

        if only_reversible:
            self.rev_efms = efms_cols.T
        else:
            self.efms = efms_cols.T

    def get_efms_cdd(self, only_reversible=False):
        if only_reversible:
            S = np.r_[self.stoich, np.eye(self.num_reacs)[supp(self.irr)]]
        else:
            S = self.stoich

        # Store information about original shape to be able to revert splitting
        # of reversible reactions later
        original_shape = np.shape(S)
        rev_indices = np.nonzero(self.rev)[0]

        # split reversible reactions by appending columns
        S_split = np.c_[S, -S[:, rev_indices]]

        # compute generators of pointed cone by splitting (all reactions irreversible)
        res = np.array(get_gens(S_split, np.zeros(len(S_split[0]))))

        # reverse splitting by combining both directions that resulted from splitting
        orig = res[:, :original_shape[1]]
        torem = np.zeros(np.shape(orig))
        splits = res[:, original_shape[1]:]

        for i, j in enumerate(rev_indices):
            torem[:, j] = splits[:, i]
        unsplit = orig - torem
        tokeep = []

        # remove spurious cycles
        for index, vector in enumerate(unsplit):
            if len(supp(vector)) > 0:
                tokeep.append(index)

        efms = unsplit[tokeep]

        if only_reversible:
            self.rev_efms = efms
        else:
            self.efms = efms

    def get_efms_milp(self, only_reversible=False):
        """
        compute the EFMs of the fluxcone using the milp approach

        """

        if only_reversible:
            S = np.r_[self.stoich, np.eye(self.num_reacs)[supp(self.irr)]]
        else:
            S = self.stoich

        # Create the extended stoichiometric matrix for reversible reactions
        for index in np.nonzero(self.rev)[0]:
            S = np.c_[S, -S[:, index]]

        n = np.shape(S)[1]
        # Initialize the MILP model
        m = mip.Model(sense=mip.MINIMIZE)

        m.verbose = False

        # Add binary variables for each reaction
        a = [m.add_var(var_type=mip.BINARY) for _ in range(n)]

        # Add continuous variables for each reaction rate
        v = [m.add_var() for _ in range(n)]

        # Add stoichiometric constraints
        for row in S:
            m += mip.xsum(row[i] * v[i] for i in range(n)) == 0

        # Define the Big M value for constraints
        M = 1000
        for i in range(n):
            m += a[i] <= v[i]
            m += v[i] <= M * a[i]

        # Exclude the zero vector solution
        m += mip.xsum(a[i] for i in range(n)) >= 1

        # Set the objective to minimize the number of non-zero variables
        m.objective = mip.xsum(a[i] for i in range(n))

        efms = []

        while True:
            # Solve the MILP model
            m.optimize()

            # Get the solution vector
            efm = np.array([v.x for v in m.vars[:n]])

            # Check for optimality
            if efm.any() is None:
                break

            # Add constraint to exclude the current solution in the next iteration
            m += xsum(a[i] for i in supp(efm)) <= len(supp(efm)) - 1

            efms.append(efm)

        efms = np.array(efms)

        # Separate positive and negative parts for reversible reactions
        efms_p = efms[:, :len(self.rev)]
        efms_m = np.zeros(np.shape(efms_p))

        counter = 0
        for r in supp(self.rev):
            efms_m[:, r] = efms[:, len(self.rev) + counter]
            counter += 1

        efms = efms_p - efms_m

        # Remove zero rows
        self.efms = efms[np.any(efms != 0, axis=1)]



    def get_mmbs(self):

        # compute v-representation using cdd (no splitting of reversible reactions)
        res = np.array(get_gens(self.stoich, self.rev))

        # compute MMBs from the v-representation
        mmbs = []
        for vector in res:
            mmb = []
            for index in supp(self.irr):
                if abs(vector[index]) > 1e-5:
                    mmb.append(index)
                    if mmb != [] and mmb not in mmbs:
                        mmbs.append(mmb)

        self.mmbs = mmbs
        return(mmbs)

    ''' determine degree of a vector'''

    def degree(self, vector):
        return self.num_reacs - np.linalg.matrix_rank(self.S[zero(np.dot(self.S, vector))])

    ''' determine irr.supp of a vector'''

    def irr_supp(self, vector):
        return list(np.intersect1d(supp(vector), supp(self.irr, tol)))

    ''' determine irr.zeros of a vector'''

    def irr_zeros(self, vector):
        return list(np.intersect1d(zero(vector), supp(self.irr, tol)))

    ''' determine rev.supp of a vector'''

    def rev_supp(self, vector):
        return list(np.intersect1d(supp(vector), supp(self.rev, tol)))

    ''' determine rev.zeros of a vector'''

    def rev_zeros(self, vector):
        return list(np.intersect1d(zero(vector), supp(self.rev, tol)))

    ''' make a reaction irreversible '''

    def make_irr(self, index):
        self.rev[index] = 0
        self.irr[index] = 1

    ''' make a reaction reversible'''

    def make_rev(self, index):
        self.rev[index] = 1
        self.irr[index] = 0

    ''' determine irredundant desciption of the flux cone '''

    def make_irredundant(self):
        redundants = "a"

        while len(redundants) > 0:
            irr = supp(self.irr)
            import random
            random.shuffle(irr)
            if redundants != "a":
                self.make_rev(redundants[0])
            redundants = []
            for index in irr:
                c = -np.eye(len(self.stoich.T))[index]
                A_ub = np.eye(len(self.stoich.T))[
                    np.setdiff1d(supp(self.irr), index)]
                A_eq = self.stoich
                b_ub = np.zeros(len(A_ub))
                b_eq = np.zeros(len(A_eq))
                bounds = (None, None)
                if abs(linprog(c, A_ub, b_ub, A_eq, b_eq, bounds).fun) < .1:
                    redundants.append(index)

    ''' determine indices of blocked irreversible reactions '''

    def blocked_irr_reactions(self):
        blocked = []
        for index in supp(self.irr):
            c = -np.eye(len(self.stoich.T))[index]
            A_ub = np.eye(len(self.stoich.T))[supp(self.irr)]
            A_eq = self.stoich
            b_ub = np.zeros(len(A_ub))
            b_eq = np.zeros(len(A_eq))
            bounds = (None, None)
            if abs(linprog(c, A_ub, b_ub, A_eq, b_eq, bounds).fun) < .001 and abs(linprog(-c, A_ub, b_ub, A_eq, b_eq, bounds).fun) < .001:
                blocked.append(index)
        blocked.reverse()
        return(blocked)

    ''' determine indices of blocked reversible reactions '''

    def blocked_rev_reactions(self):
        blocked = []
        for index in supp(self.rev):
            c = -np.eye(len(self.stoich.T))[index]
            A_ub = np.eye(len(self.stoich.T))[supp(self.irr)]
            A_eq = self.stoich
            b_ub = np.zeros(len(A_ub))
            b_eq = np.zeros(len(A_eq))
            bounds = (None, None)
            if abs(linprog(c, A_ub, b_ub, A_eq, b_eq, bounds).fun) < .001 and abs(linprog(-c, A_ub, b_ub, A_eq, b_eq, bounds).fun) < .001:
                blocked.append(index)
        blocked.reverse()
        return(blocked)

    ''' determine EFMs with inclusionwise smaller support than vector '''

    def face_candidates(self, vector):
        return self.efms[np.where(np.all((np.round(self.efms[:, np.setdiff1d(supp(self.irr), self.irr_supp(vector))], 10) == 0), axis=1))]

    ''' find 2 EFMs that can be positively combined to vector '''

    def two_gens(self, vector):

        # candidates = self.face_candidates(vector)
        candidates = self.efms
        gen_pairs = []
        for rev_zero_ind in self.rev_zeros(vector):
            pos = candidates[np.where(candidates[:, rev_zero_ind] > tol)]
            neg = candidates[np.where(candidates[:, rev_zero_ind] < -tol)]
            if len(pos) > 0 and len(neg) > 0:
                for pos_efm in pos:
                    for neg_efm in neg:
                        new_vec = -neg_efm[rev_zero_ind] * \
                            pos_efm + pos_efm[rev_zero_ind]*neg_efm
                        new_vec = pos_efm - \
                            pos_efm[rev_zero_ind]/neg_efm[rev_zero_ind]*neg_efm
                        if abs_max(new_vec - vector) < tol:
                            # if all(np.round(new_vec,5) == np.round(vector,5)):

                            # ,-neg_efm[rev_zero_ind],pos_efm[rev_zero_ind])
                            return(pos_efm, (- pos_efm[rev_zero_ind]/neg_efm[rev_zero_ind], neg_efm))
                            # gen_pairs.append(((pos_efm,self.degree(pos_efm)),(neg_efm,self.degree(neg_efm))))
                            # return gen_pairs

        return gen_pairs

    ''' find all pairs of 2 EFMs that can be positively combined to vector '''

    def all_two_gens(self, vector):
        candidates = self.face_candidates(vector)
        # candidates = self.efms
        gen_pairs = []
        for rev_zero_ind in self.rev_zeros(vector):
            pos = candidates[np.where(candidates[:, rev_zero_ind] > tol)]
            neg = candidates[np.where(candidates[:, rev_zero_ind] < -tol)]
            if len(pos) > 0 and len(neg) > 0:
                for pos_efm in pos:
                    for neg_efm in neg:
                        new_vec = pos_efm + \
                            pos_efm[rev_zero_ind]/neg_efm[rev_zero_ind]*neg_efm
                        if supp(new_vec) == supp(vector):
                            gen_pairs.append((pos_efm, neg_efm))

        return gen_pairs

    ''' determine Face of the flux cone that contains vector '''

    def face_defined_by(self, rep_vector):
        face = copy.deepcopy(self)
        # irr_zeros are the indices of the irreversibility constraints
        # that are fulfilled with equality by rep_vector
        # and these define the facets rep_vector is contained in.
        # numerical inaccuracies are assumed to be removed when the face it is contained in is determined.
        irr_zeros = np.setdiff1d(supp(self.irr), np.nonzero(rep_vector)[0])

        face.stoich = np.r_[self.stoich, np.eye(len(rep_vector))[irr_zeros]]

        return face
