# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:58:32 2024

@author: fred
"""
# import needed packages

import numpy as np
import efmtool
#import cdd
import cobra
from scipy.optimize import linprog
import copy
from fluxcones.helpers import supp, zero, TOLERANCE
import pulp


class FluxCone:

    def __init__(self, stoichiometry: np.array, reversibility: np.array, cobra_model: cobra.core.model.Model = None):
        """
        This Python function initializes a class instance with stoichiometry and reversibility arrays to
        represent a chemical reaction system.
        """
        
        # Attach cobra model
        
        if cobra_model != None:
            self.cobra = cobra_model
            
        
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
        cobra_model = cobra.io.read_sbml_model(path_to_sbml)

        # extract stoichiometric matrix
        stoich = cobra.util.array.create_stoichiometric_matrix(cobra_model)

        # extract reversibility vector
        rev = np.array([rea.reversibility for rea in cobra_model.reactions]).astype(int)

        # initialize class object from extracted parameters
        return cls(stoich, rev,cobra_model)
    
    @classmethod
    def from_bigg_id(cls,bigg_id: str):
        """
        The `from_bigg_id` function loads a model from the bigg batabase, extracts the stoichiometric matrix and
        reversibility vector, and initializes a FluxCone object with the extracted parameters.
        """
        
        # load model from bigg database
        bigg_model = cobra.io.load_model(bigg_id)
        
        # extract stoichiometric matrix
        stoich = cobra.util.array.create_stoichiometric_matrix(bigg_model)

        # extract reversibility vector
        rev = np.array([rea.reversibility for rea in bigg_model.reactions]).astype(int)

        # initialize class object from extracted parameters
        return cls(stoich, rev,bigg_model)
    
    @classmethod
    def from_cobra_model(cls,cobra_model:cobra.core.model.Model):
        
        # extract stoichiometric matrix
        stoich = cobra.util.array.create_stoichiometric_matrix(cobra_model)

        # extract reversibility vector
        rev = np.array([rea.reversibility for rea in cobra_model.reactions]).astype(int)

        # initialize class object from extracted parameters
        return cls(stoich, rev,cobra_model)
    
    
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
        Returns True if a given vector is within a specified flux cone.
        """
    
        # Check non-negativity on irreversible reactions
        irr_indices = self.irr_supp(vec)
        if len(irr_indices) > 0 and np.min(vec[irr_indices]) < 0:
            # Not in cone: irreversible reaction with negative flux
            return False
    
        # Check S * v == 0 within tolerance
        residual = np.dot(self.stoich, vec)
        # supp() presumably returns indices where abs(residual) > TOLERANCE
        if len(supp(residual, TOLERANCE)) == 0:
            return True
    
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

    def get_efms_efmtool(self, only_reversible=False,opts = dict(
        {
            "kind": "stoichiometry",
            "arithmetic": "double",
            "zero": "1e-10",
            "compression": "default",
            "log": "console",
                "level": "ON",
            "maxthreads": "-1",
            "normalize": "max",
            "adjacency-method": "pattern-tree-minzero",
            "rowordering": "MostZerosOrAbsLexMin",
        })
            ):
        """
        The function `get_efms_efmtool` calculates elementary flux modes using the efmtool library
        
        if only_reversible is set to true, only reversible efms are calculated
        """

        # Initiate reaction names and metabolite names from 0 to n resp. m because
        # efmtool needs these lists of strings as input
        # "normalize options:  [max, min, norm2, squared, none]
        

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



    def get_efms_milp(self, only_reversible=False):
        """
        Computes EFMs of the flux cone using MILP approach with OR-Tools.
        """
        from ortools.linear_solver import pywraplp

        # Build S matrix
        if only_reversible:
            S = np.r_[self.stoich, np.eye(self.num_reacs)[supp(self.irr)]]
        else:
            S = self.stoich
    
        # Duplicate columns for reversible reactions
        for index in np.nonzero(self.rev)[0]:
            S = np.c_[S, -S[:, index]]
    
        n = S.shape[1]
        M = 1000
        efms = []
        exclusion_sets = []
    
        while True:
            # IMPORTANT: create a fresh solver each iteration
            solver = pywraplp.Solver.CreateSolver("SCIP")
            if not solver:
                raise Exception("OR-Tools SCIP solver not available.")
    
            # Decision variables
            a = [solver.BoolVar(f'a_{i}') for i in range(n)]
            v = [solver.NumVar(0.0, solver.infinity(), f'v_{i}') for i in range(n)]
    
            # Constraints: S * v = 0
            for row_idx in range(S.shape[0]):
                solver.Add(
                    sum(S[row_idx, i] * v[i] for i in range(n)) == 0
                )
    
            # Linking constraints
            for i in range(n):
                solver.Add(v[i] <= M * a[i])
    
            # Non-trivial solution
            solver.Add(sum(a) >= 1)
    
            # Exclusion constraints
            for active_set in exclusion_sets:
                solver.Add(sum(a[i] for i in active_set) <= len(active_set) - 1)
    
            # Objective: minimize number of active reactions
            solver.Minimize(sum(a))
    
            status = solver.Solve()
    
            if status != pywraplp.Solver.OPTIMAL:
                break
    
            efm = np.array([v[i].solution_value() for i in range(n)])
            if efm is None or np.allclose(efm, 0, atol=1e-9):
                break
    
            # Store solution
            efms.append(efm)
            active_set = [i for i, val in enumerate(efm) if abs(val) > 1e-9]
            exclusion_sets.append(active_set)
    
        efms = np.array(efms)
        if efms.size == 0:
            return efms  # return empty array directly

        # Handle reversible reactions
        efms_p = efms[:, : len(self.rev)]
        efms_m = np.zeros_like(efms_p)
        counter = 0
        for r in supp(self.rev):
            efms_m[:, r] = efms[:, len(self.rev) + counter]
            counter += 1
    
        efms = efms_p - efms_m
        return efms[np.any(efms != 0, axis=1)]


    def degree(self, vector):
        """
        The function calculates the degree of a vector within the flux cone.
        """
        # non-negativity constraints defined by v_irr >= 0
        nonegs = np.eye(self.num_reacs)[supp(self.irr)]

        # outer description of the flux cone by C = { x | Sx >= 0}
        S = np.r_[self.stoich, nonegs]

        return int(self.num_reacs - np.linalg.matrix_rank(S[zero(np.dot(S, vector))]))

    def irr_supp(self, vector):
        """
        Returns a list of elements that are common between the support of the
        input vector and the support of the irreversible reactions of the flux cone, within a specified tolerance.
        """
        return np.array(np.intersect1d(supp(vector), supp(self.irr, TOLERANCE)))

    """ determine irr.zeros of a vector"""

    def irr_zeros(self, vector):
        """
        Returns a list of zero element indices that are common between the input vector and the irreversible reactions of 
        the flux cone, within a specified tolerance.
        """
        return np.array(np.intersect1d(zero(vector), supp(self.irr, TOLERANCE)))

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

    def get_redundant(self, irr):
        redundants = []
        for index in supp(irr):
            c = -np.eye(len(self.stoich.T))[index]
            A_ub = np.eye(len(self.stoich.T))[np.setdiff1d(supp(self.irr), index)]
            A_eq = self.stoich
            b_ub = np.zeros(len(A_ub))
            b_eq = np.zeros(len(A_eq))
            bounds = (-1000, 1000)
            if abs(linprog(c, A_ub, b_ub, A_eq, b_eq, bounds).fun) < 0.001:
                redundants.append(index)
                return index
            
    # Needs to be fixed!!!!
######################################################################################################################################################    
    def make_irredundant(self):
        redundant = self.get_redundant(supp(self.irr))
        redundants = []
        while redundants != None:
            self.make_rev(redundants[0])
            redundant = self.get_redundant(supp(self.irr))
######################################################################################################################################################

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

    """ determine Face of the flux cone that contains rep_vector """

    def face_defined_by(self, rep_vector):
        face = copy.deepcopy(self)
        # irr_zeros are the indices of the irreversibility constraints
        # that are fulfilled with equality by rep_vector
        # and these define the facets rep_vector is contained in.
        # numerical inaccuracies are assumed to be removed when the face it is contained in is determined.
        irr_zeros = np.setdiff1d(supp(self.irr), np.nonzero(rep_vector)[0])

        face.stoich = np.r_[self.stoich, np.eye(len(rep_vector))[irr_zeros]]

        return face
