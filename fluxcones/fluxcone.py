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
import copy
import mip

from fluxcones.helpers import supp, zero, TOLERANCE


class FluxCone:

    def __init__(self, stoichiometry: np.array, reversibility: np.array):
        """
        This Python function initializes a class instance with stoichiometry and reversibility arrays to
        represent a chemical reaction system.
        """

        # Stoichiometric matrix
        self.stoich = stoichiometry  # np.array

        # Number of metabolites and reactions
        self.num_metabs, self.num_reacs = np.shape(stoichiometry)  # int

        # {0,1} vector for reversible reactions
        self.rev = reversibility  # np.array

        # {0,1} vector for irreversible reactions
        self.irr = (np.ones(self.num_reacs) - self.rev).astype(int)  # np.array

    @classmethod
    def from_sbml(cls, path_to_sbml: str):
        """
        The `from_sbml` function reads an SBML file, extracts the stoichiometric matrix and
        reversibility vector, and initializes a FluxCone object with the extracted parameters.
        """

        # read sbml-file
        sbml_model = cobra.io.read_sbml_model(path_to_sbml)

        # extract stoichiometric matrix
        stoich = cobra.util.array.create_stoichiometric_matrix(sbml_model)

        # extract reversibility vector
        rev = np.array([rea.reversibility for rea in sbml_model.reactions]).astype(int)

        # initialize class object from extracted parameters
        return cls(stoich, rev)

    def get_lin_dim(self):
        """
        Calculate and returns the linear dimension of the flux cone based on its row and column spaces.
        """
        return len(supp(self.rev)) - np.linalg.matrix_rank(
            self.stoich[:, supp(self.rev)]
        )

    def get_cone_dim(self):
        """
        Returns the flux cone dimension based on the number of reactions and the rank
        of a stoichiometric matrix.

        Note: This function might not work if description of model contains reduandancies
        """
        return self.num_reacs - np.linalg.matrix_rank(self.stoich)

    """ test whether a given np.array is a steady-state fluxvector of the flux_cone instance"""

    def is_in(self, vec):
        """
        Returns True if a given vector is within a specified flux cone
        """
        # test whether v_irr >= 0
        if (
            len(vec[self.irr_supp(vec)]) > 0
            and min(vec[self.irr_supp(vec)]) < 0
        ):
            # Not in cone, because there is an irreversible reaction with negative flux
            return False

        # test whether S*v = 0
        if supp(np.dot(self.stoich, vec), TOLERANCE) == []:
            return True

        # S*v not equal to 0
        return False

    def is_efm(self, vector):
        """
        Checks if a given vector is an Elementary Flux Mode (EFM) based on rank
        tests and the support of the vector.
        """
        # 0 is not an EFM by defintion
        if len(supp(vector)) == 0:
            return False

        # rank test
        if np.linalg.matrix_rank(self.stoich[:, supp(vector)]) == len(supp(vector)) - 1:
            return True

        return False

    def get_efms_efmtool(self, only_reversible=False):
        """
        The function `get_efms_efmtool` calculates elementary flux modes using the efmtool library
        
        if only_reversible is set to true, only reversible efms are calculated
        """

        # Initiate reaction names and metabolite names from 0 to n resp. m because
        # efmtool needs these lists of strings as input
        # "normalize options:  [max, min, norm2, squared, none]
        opts = dict(
            {
                "kind": "stoichiometry",
                "arithmetic": "double",
                "zero": "1e-10",
                "compression": "default",
                "log": "console",
                "level": "OFF",
                "maxthreads": "-1",
                "normalize": "max",
                "adjacency-method": "pattern-tree-minzero",
                "rowordering": "MostZerosOrAbsLexMin",
            }
        )

        if only_reversible:
            S = np.r_[self.stoich, np.eye(self.num_reacs)[supp(self.irr)]]
        else:
            S = self.stoich

        reaction_names = list(np.arange(len(S[0])).astype(str))
        metabolite_names = list(np.arange(len(S)).astype(str))

        efms_cols = efmtool.calculate_efms(
            S, self.rev, reaction_names, metabolite_names, opts
        )

        return efms_cols.T

    def get_efms_cdd(self, only_reversible=False):
        """
        Calculates EFMs of the flux cone using the double description method
        
        if only_reversible is set to true, only reversible efms are calculated
        """
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
        S_split_rev = np.zeros(len(S_split[0]))

        # compute generators of pointed cone by splitting (all reactions irreversible)
        # nonegs is the matrix defining the inequalities for each irreversible reachtion
        irr = (np.ones(len(S_split_rev)) - S_split_rev).astype(int)
        nonegs = np.eye(len(S_split_rev))[np.nonzero(irr)[0]]

        # initiate Matrix for cdd
        if len(nonegs) > 0:
            mat = cdd.Matrix(nonegs, number_type="float")
            mat.extend(S_split, linear=True)
        else:
            mat = cdd.Matrix(S_split, linear=True)

        # generate polytope and compute generators
        poly = cdd.Polyhedron(mat)
        res = np.array(poly.get_generators())

        # reverse splitting by combining both directions that resulted from splitting
        orig = res[:, : original_shape[1]]
        torem = np.zeros(np.shape(orig))
        splits = res[:, original_shape[1] :]

        for i, j in enumerate(rev_indices):
            torem[:, j] = splits[:, i]
        unsplit = orig - torem
        tokeep = []

        # remove spurious cycles
        for index, vector in enumerate(unsplit):
            if len(supp(vector)) > 0:
                tokeep.append(index)

        return unsplit[tokeep]

    def get_efms_milp(self, only_reversible=False):
        """
        Computes the EFMs of the flux cone using the milp approach
        
        if only_reversible is set to true, only reversible efms are calculated
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
            m += mip.xsum(a[i] for i in supp(efm)) <= len(supp(efm)) - 1

            efms.append(efm)

        efms = np.array(efms)

        # Separate positive and negative parts for reversible reactions
        efms_p = efms[:, : len(self.rev)]
        efms_m = np.zeros(np.shape(efms_p))

        counter = 0
        for r in supp(self.rev):
            efms_m[:, r] = efms[:, len(self.rev) + counter]
            counter += 1

        efms = efms_p - efms_m

        # Remove zero rows
        return efms[np.any(efms != 0, axis=1)]

    def degree(self, vector):
        """
        The function calculates the degree of a vector within the flux cone.
        """
        # non-negativity constraints defined by v_irr >= 0
        nonegs = np.eye(self.num_reacs)[supp(self.irr)]

        # outer description of the flux cone by C = { x | Sx >= 0}
        S = np.r_[self.stoich, nonegs]

        return self.num_reacs - np.linalg.matrix_rank(S[zero(np.dot(S, vector))])

    def irr_supp(self, vector):
        """
        Returns a list of elements that are common between the support of the
        input vector and the support of the irreversible reactions of the flux cone, within a specified tolerance.
        """
        return list(np.intersect1d(supp(vector), supp(self.irr, TOLERANCE)))

    """ determine irr.zeros of a vector"""

    def irr_zeros(self, vector):
        """
        Returns a list of zero element indices that are common between the input vector and the irreversible reactions of 
        the flux cone, within a specified tolerance.
        """
        return list(np.intersect1d(zero(vector), supp(self.irr, TOLERANCE)))

    """ determine rev.supp of a vector"""

    def rev_supp(self, vector):
        return list(np.intersect1d(supp(vector), supp(self.rev, TOLERANCE)))

    """ determine rev.zeros of a vector"""

    def rev_zeros(self, vector):
        return list(np.intersect1d(zero(vector), supp(self.rev, TOLERANCE)))

    """ make a reaction irreversible """

    def make_irr(self, index):
        self.rev[index] = 0
        self.irr[index] = 1

    """ make a reaction reversible"""

    def make_rev(self, index):
        self.rev[index] = 1
        self.irr[index] = 0

    """ determine irredundant desciption of the flux cone """

    def get_redudants(self, irr):
        redundants = []
        for index in irr:
            c = -np.eye(len(self.stoich.T))[index]
            A_ub = np.eye(len(self.stoich.T))[np.setdiff1d(supp(self.irr), index)]
            A_eq = self.stoich
            b_ub = np.zeros(len(A_ub))
            b_eq = np.zeros(len(A_eq))
            bounds = (None, None)
            if abs(linprog(c, A_ub, b_ub, A_eq, b_eq, bounds).fun) < 0.1:
                redundants.append(index)

    def make_irredundant(self):
        redundants = self.get_redudants(supp(self.irr))

        while len(redundants) > 0:
            self.make_rev(redundants[0])
            redundants = self.get_redudants(supp(self.irr))

    """ determine indices of blocked irreversible reactions """

    def blocked_irr_reactions(self):
        blocked = []
        for index in supp(self.irr):
            c = -np.eye(len(self.stoich.T))[index]
            A_ub = np.eye(len(self.stoich.T))[supp(self.irr)]
            A_eq = self.stoich
            b_ub = np.zeros(len(A_ub))
            b_eq = np.zeros(len(A_eq))
            bounds = (None, None)
            if (
                abs(linprog(c, A_ub, b_ub, A_eq, b_eq, bounds).fun) < 0.001
                and abs(linprog(-c, A_ub, b_ub, A_eq, b_eq, bounds).fun) < 0.001
            ):
                blocked.append(index)
        blocked.reverse()
        return blocked

    """ determine indices of blocked reversible reactions """

    def blocked_rev_reactions(self):
        blocked = []
        for index in supp(self.rev):
            c = -np.eye(len(self.stoich.T))[index]
            A_ub = np.eye(len(self.stoich.T))[supp(self.irr)]
            A_eq = self.stoich
            b_ub = np.zeros(len(A_ub))
            b_eq = np.zeros(len(A_eq))
            bounds = (None, None)
            if (
                abs(linprog(c, A_ub, b_ub, A_eq, b_eq, bounds).fun) < 0.001
                and abs(linprog(-c, A_ub, b_ub, A_eq, b_eq, bounds).fun) < 0.001
            ):
                blocked.append(index)
        blocked.reverse()
        return blocked

    """ determine Face of the flux cone that contains vector """

    def face_defined_by(self, rep_vector):
        face = copy.deepcopy(self)
        # irr_zeros are the indices of the irreversibility constraints
        # that are fulfilled with equality by rep_vector
        # and these define the facets rep_vector is contained in.
        # numerical inaccuracies are assumed to be removed when the face it is contained in is determined.
        irr_zeros = np.setdiff1d(supp(self.irr), np.nonzero(rep_vector)[0])

        face.stoich = np.r_[self.stoich, np.eye(len(rep_vector))[irr_zeros]]

        return face
