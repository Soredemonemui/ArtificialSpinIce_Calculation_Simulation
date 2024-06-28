# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:16:36 2024

@author: Collo
"""

from math import sin, cos, sqrt
import numpy as np

#
def sp_loc(theta):
    return sin(theta) + cos(theta)

#calculate the distance between two spins
def r(loc1, loc2):
    a = loc2-loc1
    return sqrt(a[0]**2 + a[1]**2 + a[2]**2)

def r_(loc1, loc2):
    a = loc2 - loc1
    return sqrt(a[0]**2 + a[1]**2)

#calculate the energy of a pair of spins
def E(loc1, loc2, m1, m2):
    a = loc2 - loc1
    b = 1/r(loc1, loc2)**3
    c = sum(m1*m2)
    d = 3/(r(loc1, loc2)**2)
    f = sum(m1*a)*sum(m2*a)
    return b*(c - d*f)

def mag(theta):
    return np.array([cos(theta), sin(theta), 0])

def find_key_by_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return None