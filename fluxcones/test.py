#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 15:55:40 2025

@author: frederik
"""

from fluxcones import FluxCone

model = FluxCone.from_bigg_id("e_coli_core")

if __name__ == "__main__":
    efms = model.get_efms_milp()