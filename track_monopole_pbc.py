# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:51:52 2024

@author: Collo
"""

from math import *
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from tqdm import trange
from tqdm import tqdm


#unit_length
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

class square_kiri(object):

    def __init__(self, a, h, r, T):
        self.a = a #表示自旋的数量N = 4a^2
        self.h = h #lifting_height h
        self.r = r #角度rotation angle
        self.T = T #温度
    
    def tem_changer(self, tem):
        self.T = tem
    
    def get_r(self):
        return np.degrees(self.r)
    
    def rotate(self, spins, angle):
        unit = sp_loc(self.r)
        new_unit = sp_loc(self.r + angle)
        new_spins = {}
        
        for i in range(2*self.a):
            for j in range(2*self.a):
                if (i+j)%2 == 0:
                    new_spins[i*new_unit, j*new_unit, self.h] = spins[i*unit, j*unit, self.h] - angle
                else:
                    new_spins[i*new_unit, j*new_unit, 0] = spins[i*unit, j*unit, 0] + angle
        self.r += angle
        return new_spins

    def ver_unit(self, k):
        #初始化生成一个2×2 vertex 自旋晶格（所有自旋随机取向)
        #k表示该ver_unit的位置矢量
        #spins的key表示位置，value表示方位角度
        spins = {}
        for i in range(2):
            for j in range(2):
                if (i + j) % 2 == 0:
                    loc_s = ((i + 2*k[0])*sp_loc(self.r),
                             (j + 2*k[1])*sp_loc(self.r),
                             self.h)
                    if i == 0:
                        spins[loc_s] = 1*pi/4 - self.r #+ pi*np.random.choice([0, 1])
                    else:
                        spins[loc_s] = 5*pi/4 - self.r #+ pi*np.random.choice([0, 1])
                else:
                    loc_s = ((i + 2*k[0])*sp_loc(self.r),
                             (j + 2*k[1])*sp_loc(self.r),
                             0)
                    if i == 0:
                        spins[loc_s] = 3*pi/4 + self.r #+ pi*np.random.choice([0, 1])
                    else:
                        spins[loc_s] = -1*pi/4 + self.r #+ pi*np.random.choice([0, 1])
        return spins

    def spin_lattice(self):
        #X形状square ice
        all_spins = {}
        all_key, all_val = [], []
        for k in range(self.a):
            for m in range(self.a):
                loc_uv = (k, m)

                all_key += self.ver_unit(loc_uv).keys()
                all_val += self.ver_unit(loc_uv).values()
        for i in range(len(all_key)):
            all_spins[all_key[i]] = all_val[i]
        return all_spins      
  
    def intro_mono(self, spins, loc, num):
        #loc = location of vertex (i, j) = 0 ~ self.a
        #num = determining the spin in the vertex spin cluster = (00, 01, 10, 11)
        if sum(num) % 2 == 0:
            key = ((num[0] + 2*loc[0])*sp_loc(self.r),(num[1] + 2*loc[1])*sp_loc(self.r),self.h)
        else:
            key = ((num[0] + 2*loc[0])*sp_loc(self.r),(num[1] + 2*loc[1])*sp_loc(self.r),0)
        spins[key] += pi
        return spins
    
    def surr_1(self, spins, key):
        surr = {}
        unit = sp_loc(self.r)
        x,y,z1 = round(key[0]/unit, 0), round(key[1]/unit, 0), key[2]
        
        if z1 == self.h:
            z2 = 0
        else:
            z2 = self.h
        
        for i in range(2):
            loc11 = (x*unit, (y+(-1)**i)%(2*self.a)*unit, z2) #TRUE
            loc21 = ((x+(-1)**i)%(2*self.a)*unit, y*unit, z2) #TRUE
            loc12 = (x*unit, (y+(-1)**i)*unit, z2) #CAL
            loc22 = ((x+(-1)**i)*unit, y*unit, z2) #CAL
            
            surr[loc12] = spins[loc11]
            surr[loc22] = spins[loc21]
            for j in range(2):
                loc31 = ((x+(-1)**i)%(2*self.a)*unit, (y+(-1)**j)%(2*self.a)*unit, z1)
                loc32 = ((x+(-1)**i)*unit, (y+(-1)**j)*unit, z1)
                surr[loc32] = spins[loc31]
        return surr
    
    def surr_2(self, spins, key):
        unit = sp_loc(self.r)
        x,y,z1 = round(key[0]/unit, 0), round(key[1]/unit, 0), key[2]
        if z1 == self.h:
            z2 = 0
        else:
            z2 = self.h
        #NN+NNN eight  
        surr = self.surr_1(spins, key)
        #N4
        for i in range(2):
            loc11 = (x*unit, (y+2*(-1)**i)%(2*self.a)*unit, z1)
            loc21 = ((x+2*(-1)**i)%(2*self.a)*unit, y*unit, z1)
            loc12 = (x*unit, (y+2*(-1)**i)*unit, z1)
            loc22 = ((x+2*(-1)**i)*unit, y*unit, z1) 
            
            surr[loc12] = spins[loc11]
            surr[loc22] = spins[loc21]
            #N6
            for j in range(2):
                loc31 = ((x+2*(-1)**i)%(2*self.a)*unit, (y+2*(-1)**j)%(2*self.a)*unit, z1)
                loc32 = ((x+2*(-1)**i)*unit, (y+2*(-1)**j)*unit, z1)
                
                surr[loc32] = spins[loc31]
        #N5
        for i in range(2):
            for j in range(2):
                loc41 = ((x+1*(-1)**i)%(2*self.a)*unit, (y+2*(-1)**j)%(2*self.a)*unit, z2)
                loc51 = ((x+2*(-1)**i)%(2*self.a)*unit, (y+1*(-1)**j)%(2*self.a)*unit, z2)
                loc42 = ((x+1*(-1)**i)*unit, (y+2*(-1)**j)*unit, z2)
                loc52 = ((x+2*(-1)**i)*unit, (y+1*(-1)**j)*unit, z2)
                
                surr[loc42] = spins[loc41]
                surr[loc52] = spins[loc51]
        return surr
    
    def surr_k(self, spins, key, k):
        #surroundings_k
        surr = {}
        unit = sp_loc(self.r)
        x,y,z1 = round(key[0]/unit, 0), round(key[1]/unit, 0), key[2]    
        if z1 == self.h:
            z2 = 0
        else:
            z2 = self.h
        
        for i in range(-k, k+1):
            for j in range(-k, k+1):
                if (i, j) != (0, 0):
                    if (i+j)%2 == 0:
                        loc1 = ((x+i)%(2*self.a)*unit, (y+j)%(2*self.a)*unit, z1)
                        loc2 = ((x+i)*unit, (y+j)*unit, z1)
                    else:
                        loc1 = ((x+i)%(2*self.a)*unit, (y+j)%(2*self.a)*unit, z2)
                        loc2 = ((x+i)*unit, (y+j)*unit, z2)
                    surr[loc2] = spins[loc1]
                else:
                    continue
        return surr   
        
    def delta_E(self, key_spin, spins):
        #non_pbc
        energy = 0        
        locs = list(spins.keys())        
        for loc in locs:
            if loc != key_spin:
                energy += E(np.array(loc), np.array(key_spin),
                            mag(spins[loc]), mag(spins[key_spin]))
        return energy
    
    def delta_E_local(self, key_spin, spins):
        #with_pbc
        energy = 0
        k = 3
        surr = self.surr_k(spins, key_spin, k)
        locs = list(surr)        
        for loc in locs:
            energy += E(np.array(loc), np.array(key_spin),
                        mag(surr[loc]), mag(spins[key_spin]))
        return energy
    
    def p_acc(self, energy):
        #计算接受翻转的概率
        if energy > 0:
            return 1
        else:
            return np.exp(2*energy/self.T)

    def flip(self, spins):
        #随机选一个自旋，判定其是否反转
        key = random.choice(list(spins.keys()))
        
        energy = self.delta_E(key, spins) #all interaction considered, no pbc
        p = random.random()
        if p < self.p_acc(energy):
            spins[key] = (spins[key] + pi) % (2*pi)
            return (spins,1)
        else:
            return (spins,0)

    def flip_local(self, spins):
        #随机选一个自旋，判定其是否反转
        key = random.choice(list(spins.keys()))
        
        energy = self.delta_E_local(key, spins) #local interactions only, with pbc
        p = random.random()
        if p < self.p_acc(energy):
            spins[key] = (spins[key] + pi) % (2*pi)
            return (spins,1)
        else:
            return (spins,0)
   
    def get_spin_cluster_new(self, spins, vertex):
        #获取一个vertex/loop周围的四个spin,pbc
        cluster = {}
        vx, vy = vertex[0], vertex[1]
        unit = 0.5*sp_loc(self.r) 
        keys = list(spins.keys())
        if (round((vx+unit)/(2*unit), 0)%(2*self.a)*2*unit, \
            round((vy+unit)/(2*unit), 0)%(2*self.a)*2*unit, 0) in keys:
            for i in range(2):
                for j in range(2):
                    if (i+j)%2 == 0:
                        loc1 = (vx+unit*(-1)**i, vy+unit*(-1)**j, 0)
                        loc2 = (round((vx+unit*(-1)**i)/(2*unit), 0)%(2*self.a)*2*unit, \
                                round((vy+unit*(-1)**j)/(2*unit), 0)%(2*self.a)*2*unit, 0)
                        cluster[loc1] = spins[loc2]
                    else:
                        loc1 = (vx+unit*(-1)**i, vy+unit*(-1)**j, self.h)
                        loc2 = (round((vx+unit*(-1)**i)/(2*unit), 0)%(2*self.a)*2*unit, \
                                round((vy+unit*(-1)**j)/(2*unit), 0)%(2*self.a)*2*unit, self.h)
                        cluster[loc1] = spins[loc2]
        else:
            for i in range(2):
                for j in range(2):
                    if (i+j)%2 == 0:
                        loc1 = (vx+unit*(-1)**i, vy+unit*(-1)**j, self.h)
                        loc2 = (round((vx+unit*(-1)**i)/(2*unit), 0)%(2*self.a)*2*unit, \
                                round((vy+unit*(-1)**j)/(2*unit), 0)%(2*self.a)*2*unit, self.h)
                        cluster[loc1] = spins[loc2]
                    else:
                        loc1 = (vx+unit*(-1)**i, vy+unit*(-1)**j, 0)     
                        loc2 = (round((vx+unit*(-1)**i)/(2*unit), 0)%(2*self.a)*2*unit, \
                                round((vy+unit*(-1)**j)/(2*unit), 0)%(2*self.a)*2*unit, 0)
                        cluster[loc1] = spins[loc2]
        return cluster

    def vertex_type(self, cluster, vertex):
        #获取一个vertex的Type
        #self.r = np.radians(90)时不适用
        
        in_or_out = {} #in gives 1, out gives -1
           
        for key in cluster.keys():
            s = [key[0], key[1]]
            direct = np.array(s) - np.array(vertex)
            orient = np.array([cos(cluster[key]),
                               sin(cluster[key])])
            if direct.dot(orient) > 0:
                in_or_out[key] = -1
            else:
                in_or_out[key] = 1

        charge = sum(list(in_or_out.values()))    
        if abs(charge) == 4:
            return 4
        else:
            if charge == 0:
                for key1 in cluster.keys():
                    for key2 in cluster.keys():
                        s1 = [key1[0], key1[1]]
                        s2 = [key2[0], key2[1]]
                        direct1 = np.array(s1) - np.array(vertex)
                        direct2 = np.array(s2) - np.array(vertex)
                        if direct1.dot(direct2) < -0.3: #spins parallel to each other
                            if in_or_out[key1] == in_or_out[key2]:
                                return 1
                            else:
                                return 2
            else:
                if abs(charge) == 2:
                    return 3
                else:
                    raise ValueError(f"Vertex {vertex} is illegal! \n Cluster = {cluster}, charge = {charge}")

    def vertex_type_fig(self, cluster, vertex):
        #获取一个vertex的Type
        #self.r = np.radians(90)时不适用
        
        in_or_out = {} #in gives 1, out gives -1
           
        for key in cluster.keys():
            s = [key[0], key[1]]
            direct = np.array(s) - np.array(vertex)
            orient = np.array([cos(cluster[key]),
                               sin(cluster[key])])
            if direct.dot(orient) > 0:
                in_or_out[key] = -1
            else:
                in_or_out[key] = 1

        charge = sum(list(in_or_out.values()))    
        if abs(charge) == 4:
            if charge == 4:
                return 41
            else:
                return 42 #type IV
        else:
            if charge == 0:
                for key1 in cluster.keys():
                    for key2 in cluster.keys():
                        s1 = [key1[0], key1[1]]
                        s2 = [key2[0], key2[1]]
                        direct1 = np.array(s1) - np.array(vertex)
                        direct2 = np.array(s2) - np.array(vertex)
                        if direct1.dot(direct2) < -0.3: #spins parallel to each other
                            if in_or_out[key1] == in_or_out[key2]:
                                return 1
                            else:
                                return 2
            else:
                if abs(charge) == 2:
                    if charge == 2:
                        return 31
                    else:
                        return 32 #type III
                else:
                    raise ValueError(f"Vertex {vertex} is illegal! \n Cluster = {cluster}, charge = {charge}")

    def vertex_type_all(self, cluster, vertex):
        in_or_out = {} #in gives 1, out gives -1
        keys = cluster.keys()
           
        for key in keys:
            s = [key[0], key[1]]
            direct = np.array(s) - np.array(vertex)
            orient = np.array([cos(cluster[key]),
                               sin(cluster[key])])
            if direct.dot(orient) > 0:
                in_or_out[key] = -1
            else:
                in_or_out[key] = 1

        charge = sum(list(in_or_out.values()))    
        if abs(charge) == 4:
            if charge == 4:
                return 41
            else:
                return 42 #type IV
        else:
            if charge == 0:
                for key1 in keys:
                    for key2 in keys:
                        s1 = [key1[0], key1[1]]
                        s2 = [key2[0], key2[1]]
                        direct1 = np.array(s1) - np.array(vertex)
                        direct2 = np.array(s2) - np.array(vertex)
                        if direct1.dot(direct2) < -0.3: #spins parallel to each other
                            if in_or_out[key1] == in_or_out[key2]:
                                return 1
                            else:
                                magnetization = np.array([0,0])
                                for key in keys:
                                    magnetization = magnetization + np.array([cos(cluster[key]), sin(cluster[key])])
                                if magnetization[0] > 0.1:
                                    return 21
                                elif magnetization[0] < -0.1:
                                    return 22
                                elif magnetization[1] > 0.1:
                                    return 23
                                elif magnetization[1] < -0.1:
                                    return 24
            else:
                if abs(charge) == 2:
                    if charge == 2:
                        return 31
                    else:
                        return 32 #type III
                else:
                    raise ValueError(f"Vertex {vertex} is illegal! \n Cluster = {cluster}, charge = {charge}")        

    def boundary_ver(self, type_lat):
        unit = sp_loc(self.r)
        boudary = [(-1/2)*unit, (-1/2 + 2*self.a)*unit]
        b_v = {}
        for key in type_lat.keys():
            if key[0] in boudary or key[1] in boudary:
                b_v[key] = type_lat[key]
        return b_v
    
    def all_ty_for_fig(self, spins):
        ty = {}
        if self.r <= np.radians(45):
            #vertices
            for i in range(self.a):
                for j in range(self.a):
                    vertex_1 = ((1/2 + 2*i)*sp_loc(self.r),
                                (1/2 + 2*j)*sp_loc(self.r))
                    cluster_1 = self.get_spin_cluster(spins, vertex_1)
                    ty[vertex_1] = self.vertex_type_all(cluster_1, vertex_1)
                    
                    if i < self.a - 1 and j < self.a - 1:
                        vertex_2 = ((3/2 + 2*i)*sp_loc(self.r),
                                    (3/2 + 2*j)*sp_loc(self.r))
                        cluster_2 = self.get_spin_cluster(spins, vertex_2)
                        ty[vertex_2] = self.vertex_type_all(cluster_2, vertex_2)
                    else:
                        continue
        else:
            for i in range(self.a):
                for j in range(self.a - 1):
                    vertices_3 = ((1/2 + 2*i)*sp_loc(self.r),
                                  (3/2 + 2*j)*sp_loc(self.r))
                    vertices_4 = ((3/2 + 2*j)*sp_loc(self.r),
                                  (1/2 + 2*i)*sp_loc(self.r))
                    cluster_3 = self.get_spin_cluster(spins, vertices_3)
                    cluster_4 = self.get_spin_cluster(spins, vertices_4)
                    ty[vertices_3] = self.vertex_type_all(cluster_3, vertices_3)
                    ty[vertices_4] = self.vertex_type_all(cluster_4, vertices_4)
        return ty

    def all_ty_for_fig_pbc(self, spins):
        ty = {}
        unit = sp_loc(self.r)
        if self.r <= np.radians(45):
            #vertices
            #boundary
            for i in range(self.a):
                v1 = ((-1/2)*unit, (-1/2 + 2*i)*unit)
                v2 = ((-1/2 + 2*self.a)*unit, (-1/2 + 2*self.a - 2*i)*unit)
                v3 = ((3/2 + 2*i)*unit, (-1/2)*unit)
                v4 = ((-5/2 + 2*self.a - 2*i)*unit, (-1/2 + 2*self.a)*unit)
                cl_1 = self.get_spin_cluster_new(spins, v1)
                cl_2 = self.get_spin_cluster_new(spins, v2)
                cl_3 = self.get_spin_cluster_new(spins, v3)
                cl_4 = self.get_spin_cluster_new(spins, v4)
                ty[v1] = self.vertex_type_fig(cl_1, v1)
                ty[v2] = self.vertex_type_fig(cl_2, v2)
                ty[v3] = self.vertex_type_fig(cl_3, v3)
                ty[v4] = self.vertex_type_fig(cl_4, v4)
            
            
            #in
            for i in range(self.a):
                for j in range(self.a):
                    vertex_1 = ((1/2 + 2*i)*unit,
                                (1/2 + 2*j)*unit)
                    cluster_1 = self.get_spin_cluster_new(spins, vertex_1)
                    ty[vertex_1] = self.vertex_type_fig(cluster_1, vertex_1)
                    
                    if i < self.a - 1 and j < self.a - 1:
                        vertex_2 = ((3/2 + 2*i)*unit,
                                    (3/2 + 2*j)*unit)
                        cluster_2 = self.get_spin_cluster_new(spins, vertex_2)
                        ty[vertex_2] = self.vertex_type_fig(cluster_2, vertex_2)
                    else:
                        continue
        else:
            for i in range(self.a):
                v1 = ((-1/2)*unit, (1/2 + 2*i)*unit)
                v2 = ((-1/2 + 2*self.a)*unit, (1/2 + 2*i)*unit)
                v3 = ((1/2 + 2*i)*unit, (-1/2)*unit)
                v4 = ((1/2 + 2*i)*unit, (-1/2 + 2*self.a)*unit)
                cl_1 = self.get_spin_cluster_new(spins, v1)
                cl_2 = self.get_spin_cluster_new(spins, v2)
                cl_3 = self.get_spin_cluster_new(spins, v3)
                cl_4 = self.get_spin_cluster_new(spins, v4)
                ty[v1] = self.vertex_type_fig(cl_1, v1)
                ty[v2] = self.vertex_type_fig(cl_2, v2)
                ty[v3] = self.vertex_type_fig(cl_3, v3)
                ty[v4] = self.vertex_type_fig(cl_4, v4)
            for i in range(self.a):
                for j in range(self.a - 1):
                    vertices_3 = ((1/2 + 2*i)*unit,
                                  (3/2 + 2*j)*unit)
                    vertices_4 = ((3/2 + 2*j)*unit,
                                  (1/2 + 2*i)*unit)
                    cluster_3 = self.get_spin_cluster_new(spins, vertices_3)
                    cluster_4 = self.get_spin_cluster_new(spins, vertices_4)
                    ty[vertices_3] = self.vertex_type_fig(cluster_3, vertices_3)
                    ty[vertices_4] = self.vertex_type_fig(cluster_4, vertices_4)
        return ty

    def true_all_ty_for_fig_pbc(self, spins):
        ty = {}
        unit = sp_loc(self.r)
        if self.r <= np.radians(45):
            #vertices
            #boundary
            for i in range(self.a):
                v1 = ((-1/2)*unit, (-1/2 + 2*i)*unit)
                v2 = ((-1/2 + 2*self.a)*unit, (-1/2 + 2*self.a - 2*i)*unit)
                v3 = ((3/2 + 2*i)*unit, (-1/2)*unit)
                v4 = ((-5/2 + 2*self.a - 2*i)*unit, (-1/2 + 2*self.a)*unit)
                cl_1 = self.get_spin_cluster_new(spins, v1)
                cl_2 = self.get_spin_cluster_new(spins, v2)
                cl_3 = self.get_spin_cluster_new(spins, v3)
                cl_4 = self.get_spin_cluster_new(spins, v4)
                ty[v1] = self.vertex_type_all(cl_1, v1)
                ty[v2] = self.vertex_type_all(cl_2, v2)
                ty[v3] = self.vertex_type_all(cl_3, v3)
                ty[v4] = self.vertex_type_all(cl_4, v4)
            
            
            #in
            for i in range(self.a):
                for j in range(self.a):
                    vertex_1 = ((1/2 + 2*i)*unit,
                                (1/2 + 2*j)*unit)
                    cluster_1 = self.get_spin_cluster_new(spins, vertex_1)
                    ty[vertex_1] = self.vertex_type_all(cluster_1, vertex_1)
                    
                    if i < self.a - 1 and j < self.a - 1:
                        vertex_2 = ((3/2 + 2*i)*unit,
                                    (3/2 + 2*j)*unit)
                        cluster_2 = self.get_spin_cluster_new(spins, vertex_2)
                        ty[vertex_2] = self.vertex_type_all(cluster_2, vertex_2)
                    else:
                        continue
        else:
            for i in range(self.a):
                v1 = ((-1/2)*unit, (1/2 + 2*i)*unit)
                v2 = ((-1/2 + 2*self.a)*unit, (1/2 + 2*i)*unit)
                v3 = ((1/2 + 2*i)*unit, (-1/2)*unit)
                v4 = ((1/2 + 2*i)*unit, (-1/2 + 2*self.a)*unit)
                cl_1 = self.get_spin_cluster_new(spins, v1)
                cl_2 = self.get_spin_cluster_new(spins, v2)
                cl_3 = self.get_spin_cluster_new(spins, v3)
                cl_4 = self.get_spin_cluster_new(spins, v4)
                ty[v1] = self.vertex_type_all(cl_1, v1)
                ty[v2] = self.vertex_type_all(cl_2, v2)
                ty[v3] = self.vertex_type_all(cl_3, v3)
                ty[v4] = self.vertex_type_all(cl_4, v4)
            for i in range(self.a):
                for j in range(self.a - 1):
                    vertices_3 = ((1/2 + 2*i)*unit,
                                  (3/2 + 2*j)*unit)
                    vertices_4 = ((3/2 + 2*j)*unit,
                                  (1/2 + 2*i)*unit)
                    cluster_3 = self.get_spin_cluster_new(spins, vertices_3)
                    cluster_4 = self.get_spin_cluster_new(spins, vertices_4)
                    ty[vertices_3] = self.vertex_type_all(cluster_3, vertices_3)
                    ty[vertices_4] = self.vertex_type_all(cluster_4, vertices_4)
        return ty
    
    def all_ty_num(self, type_lat):
        ty_count_all = {}
        ty_count_bou = {}
        ty_count = {}
        
        unit = sp_loc(self.r)
        keys_all = list(type_lat.keys())
        spe_key = ((-1/2)*unit, (-1/2)*unit)
        
        values_all = list(type_lat.values())
        values_bou = list(self.boundary_ver(type_lat).values())
        
        for value in values_all:
            ty_count_all[value] = values_all.count(value)
        for value in values_bou:
            ty_count_bou[value] = values_bou.count(value)
        for key in ty_count_all.keys():
            if key in ty_count_bou.keys():
                ty_count[key] = int(ty_count_all[key] - 1/2*(ty_count_bou[key]))
            else:
                ty_count[key] = int(ty_count_all[key])
        if spe_key in keys_all:
            ty_count[type_lat[spe_key]] = ty_count[type_lat[spe_key]] - 1
        return ty_count     

    def plot_lat_ty_mono(self, lattice, type_lat, num):
        #mag_H = round(np.linalg.norm(self.h), 1)
        #theta = round(np.degrees(self.r),1)
        
        unit = sp_loc(self.r)
        scale = 1.2
        length = 0.4*scale
        head = 0.35*scale
        width = 0.13*scale
        
        fig = plt.figure(figsize = [10, 10], dpi = 100)
        ax = fig.add_subplot(111)

        ty_keys = type_lat.keys()
        
        # draw type 
        for key in ty_keys:
            if type_lat[key] == 1:
                ty_c = 'slateblue'
            elif str(type_lat[key])[0] == '2':
                ty_c = 'lightcoral'
            elif str(type_lat[key])[0] == '3' or str(type_lat[key])[0] == '4':
                ty_c = 'limegreen'
            square = plt.Rectangle(xy = (key[0], key[1]-unit), width = sqrt(2)*unit, \
                                   height = sqrt(2)*unit, angle = 45, ec = 'white', fc = ty_c)
            ax.add_patch(square)
        
        #draw spins       
        for key in lattice.keys():
            # lifted or not
            if key[2] != 0:
                ec = 'gray'
            else:
                ec = 'k'

            # magnetification visualization
            color = 'w'
            ax.arrow(key[0] - length*cos(lattice[key]), key[1] - length*sin(lattice[key]), \
                     2*length*cos(lattice[key]), 2*length*sin(lattice[key]), head_length = head, \
                     head_width = head, fc = color, ec = ec, linestyle = '-', width = width, \
                     length_includes_head = True)

        #draw monopole
        for key in ty_keys:
            if type_lat[key] == 31:
                mono_c = 'red'
                mono_r = 0.3
                circle = plt.Circle(xy = (key[0], key[1]), radius = mono_r, fc = mono_c, ec = 'white')
                ax.add_patch(circle)
            elif type_lat[key] == 32:
                mono_c = 'blue'
                mono_r = 0.3
                circle = plt.Circle(xy = (key[0], key[1]), radius = mono_r, fc = mono_c, ec = 'white')
                ax.add_patch(circle)
            elif type_lat[key] == 41:
                mono_c = 'red'
                mono_r = 0.5
                circle = plt.Circle(xy = (key[0], key[1]), radius = mono_r, fc = mono_c, ec = 'white')
                ax.add_patch(circle)
            elif type_lat[key] == 42:
                mono_c = 'blue'
                mono_r = 0.5
                circle = plt.Circle(xy = (key[0], key[1]), radius = mono_r, fc = mono_c, ec = 'white')
                ax.add_patch(circle)
            else:
                continue
            
        if num == None:
            plt.axis('equal')
            plt.show()
        else:
            #plt.xlim([-1, int(sqrt(2)*self.a) + 2])
            #plt.ylim([-1, int(sqrt(2)*self.a) + 2])
            
            plt.savefig(f'spin_ty_lat_[{self.a, self.h, self.T, num}].jpg')
            plt.close()
            
    def plot_lat(self, lattice, num):
        mag_H = round(np.linalg.norm(self.h), 1)
        #theta = round(np.degrees(self.r),1)

        scale = 1.3
        length = 0.4*scale
        head = 0.3*scale
        width = 0.1*scale
        
        plt.figure(figsize = [10, 10], dpi = 100)
        
        for key in lattice.keys():
            # lifted or not
            if key[2] != 0:
                ls = '--'
            else:
                ls = '-'

            # magnetification visualization
            color = 'w'
            plt.arrow(key[0] - length*cos(lattice[key]), key[1] - length*sin(lattice[key]), \
                      2*length*cos(lattice[key]), 2*length*sin(lattice[key]), head_length = head, \
                      head_width = head, fc = color, ec = 'k', linestyle = ls, width = width, \
                      length_includes_head = True)
        if num == None:
            plt.axis('equal')
            plt.show()
        else:
            #plt.xlim([-1, int(sqrt(2)*self.a) + 2])
            #plt.ylim([-1, int(sqrt(2)*self.a) + 2])
            
            plt.savefig(f'{self.a, self.T, mag_H, num}.jpg')
            plt.close()
    
    def plot_all_ty_lat(self, type_lat, num):
        #mag_H = round(np.linalg.norm(self.h), 1)
        #theta = round(np.degrees(self.r),1)
        
        unit = sp_loc(self.r)
        scale = 1.2
        length = 0.5*scale
        head = 0.45*scale
        width = 0.15*scale
        
        fig = plt.figure(figsize = [10, 10], dpi = 100)
        ax = fig.add_subplot(111)

        ty_keys = type_lat.keys()
        
        # draw type 
        for key in ty_keys:
            if type_lat[key] == 1:
                ty_c = 'slateblue'
            elif str(type_lat[key])[0] == '2':
                ty_c = 'lightcoral'
            elif str(type_lat[key])[0] == '3' or str(type_lat[key])[0] == '4':
                ty_c = 'limegreen'                
            square = plt.Rectangle(xy = (key[0], key[1]-unit), width = sqrt(2)*unit, \
                                   height = sqrt(2)*unit, angle = 45, ec = 'white', fc = ty_c)
            ax.add_patch(square)
        
        #draw type II
        for key in ty_keys:
            color = 'white'
            ec = 'k'
            if str(type_lat[key])[0] == '2':
                if type_lat[key] == 21:
                    angle = 0

                elif type_lat[key] == 22:
                    angle = np.pi

                elif type_lat[key] == 23:
                    angle = np.pi/2

                elif type_lat[key] == 24:
                    angle = 3*np.pi/2

                ax.arrow(key[0] - length*cos(angle), key[1] - length*sin(angle), \
                         2*length*cos(angle), 2*length*sin(angle), \
                         head_length = head, head_width = head, fc = color, ec = ec, linestyle = '-', \
                         width = width, length_includes_head = True)

        #draw monopole
        for key in ty_keys:
            if type_lat[key] == 31:
                mono_c = 'red'
                mono_r = 0.3
                circle = plt.Circle(xy = (key[0], key[1]), radius = mono_r, fc = mono_c, ec = 'white')
                ax.add_patch(circle)
            elif type_lat[key] == 32:
                mono_c = 'blue'
                mono_r = 0.3
                circle = plt.Circle(xy = (key[0], key[1]), radius = mono_r, fc = mono_c, ec = 'white')
                ax.add_patch(circle)
            elif type_lat[key] == 41:
                mono_c = 'red'
                mono_r = 0.5
                circle = plt.Circle(xy = (key[0], key[1]), radius = mono_r, fc = mono_c, ec = 'white')
                ax.add_patch(circle)
            elif type_lat[key] == 42:
                mono_c = 'blue'
                mono_r = 0.5
                circle = plt.Circle(xy = (key[0], key[1]), radius = mono_r, fc = mono_c, ec = 'white')
                ax.add_patch(circle)
            else:
                continue
            
        if num == None:
            plt.axis('equal')
            plt.show()
        else:
            #plt.xlim([-1, int(sqrt(2)*self.a) + 2])
            #plt.ylim([-1, int(sqrt(2)*self.a) + 2])
            plt.title(f'{round(np.degrees(self.r),3)}°')
            plt.savefig(f'spin_ty_lat_[{self.a, self.h, self.T, num}].jpg')
            plt.close()        


# lattice_X
h = 0.0
n = 15
t = 0.1
deg_th = 0

theta = np.radians(deg_th)

test = square_kiri(n, h, theta, t) #class
spins = test.spin_lattice() #initial spins
#spins = test.intro_mono(spins, (0,0), [0,0])
#spins = test.intro_mono(spins, (0,9), [0,1])


loc = (8,9)
length_ds = 8


length_ds = 5
loc = (int(n/2)-int(length_ds/2)-1,int(n/2)-1)
for i in range(length_ds):
    spins = test.intro_mono(spins, (loc[0]+i,loc[1]), [1,1])
    spins = test.intro_mono(spins, (loc[0]+1+i,loc[1]), [0,1])
spins = test.intro_mono(spins, (loc[0],loc[1]), [0,1])


# moving monopole to build DS
'''
#ds=ty2 moving left
for i in range(length_ds):
    spins = test.intro_mono(spins, (loc[0]-i,loc[1]), [1,1])
    spins = test.intro_mono(spins, (loc[0]-i,loc[1]), [0,1])
#and then downward
for i in range(length_ds):
    spins = test.intro_mono(spins, (loc[0]+1,loc[1]-i), [0,1])
    spins = test.intro_mono(spins, (loc[0]+1,loc[1]-i), [0,0])

# bck=ty2, ty2ds
for i in range(length_ds):
    spins = test.intro_mono(spins, (loc[0]-1-i,loc[1]-1-i), [1,1])
    spins = test.intro_mono(spins, (loc[0]-1-i,loc[1]-1-i), [0,0])

#for i in range(length_ds):
    #spins = test.intro_mono(spins, (loc[0]-4-i,loc[1]-3+i), [1,0])
    #spins = test.intro_mono(spins, (loc[0]-4-i,loc[1]-3+i), [0,1])

#for i in range(7):
    #spins = test.intro_mono(spins, (loc[0]+i,loc[1]-length_ds), [1,0])
    #spins = test.intro_mono(spins, (loc[0]+1+i,loc[1]-length_ds), [0,0])

#ty2 ds
ini_y = 2
ds_l = 7
for i in range(2):
    spins = test.intro_mono(spins, (1-i,ini_y+i), [0,1])
    spins = test.intro_mono(spins, (1-i,ini_y+i), [1,0])
for i in range(0, ds_l-2):
    spins = test.intro_mono(spins, (n-1-i,ini_y+2+i), [0,1])
    spins = test.intro_mono(spins, (n-1-i,ini_y+2+i), [1,0])

#ty2 ds
for i in range(5):
    spins = test.intro_mono(spins, (i,i), [0,0])
    spins = test.intro_mono(spins, (i,i), [1,1])
for i in range(5):
    spins = test.intro_mono(spins, (n-1-i,n-1-i), [0,0])
    spins = test.intro_mono(spins, (n-1-i,n-1-i), [1,1])
'''


#produce gif
#unit = sp_loc(theta)
#type_lat = test.all_ty_for_fig_pbc(spins) #pbc considered
#test.plot_lat_ty_mono(spins, type_lat, 0) #draw initial state fig
#type_lat = test.true_all_ty_for_fig_pbc(spins)
type_lat = test.true_all_ty_for_fig_pbc(spins)
test.plot_all_ty_lat(type_lat, 0)
#ty_num = test.all_ty_num(type_lat)

#print(test.all_ty_num(type_lat))
#print(type_lat[(-1/2*unit, -1/2*unit)])
#vertex = (7/2*unit, 3/2*unit)

#cluster = test.surr_k(spins, (2*unit, 2*unit, h), 3)
#test.plot_lat(cluster, None)

#ty = test.vertex_type_fig(cluster, vertex)
#print(cluster, ty)
#test.plot_lat(cluster, None)


#tracking the movement of monopoles
MC_steps = 20
flips = MC_steps*4*n**2
fig_nums = [0]
for i in trange(flips):
    if test.get_r() <= 12:
        spins = test.rotate(spins, np.radians(450/flips)) 
    flip = test.flip_local(spins) #only local interactions considered
    #flip = test.flip(spins) #all interactions considered
    spins = flip[0]
    f_or_n = flip[1]

    if f_or_n == 1:
        type_lat = test.true_all_ty_for_fig_pbc(spins) #pbc considered
        test.plot_all_ty_lat(type_lat, i+1)
        fig_nums.append(i+1)
        
        type_num = list(type_lat.values())
        if type_num.count(31) == 0 and type_num.count(32) == 0: #if there is no monopole, stop
            break
        
#test.plot_all_ty_lat(type_lat, 1) #the last frame

'''
with imageio.get_writer(uri=f'pbc_n={n}, t={t}, theta={deg_th}, h={h}.gif', mode='I', fps=5, loop = 1) as writer:
    for i in fig_nums:
        writer.append_data(imageio.imread(f'spin_ty_lat_[{n, h, t, i}].jpg'))
        #os.remove(f'spin_ty_lat_[{n, h, t, i}].jpg')
'''

'''
def time_ave_ty_num_(n, h, theta, t, times):
    
    v = 2*n**2
    
    all_ty = [1,2,31,32,41,42]
    t1, t2, t3, t4 = [],[],[],[]
    flips = 100*n**2
    for i in range(times):
        test = square_kiri(n, h, theta, t) #class
        spins = test.spin_lattice() #random spins
        
        for j in tqdm(range(flips), leave = False, desc = f'Flips_{round(h,3), i}'): #flips loading bar
            spins = test.flip_local(spins)[0]
        type_lat = test.all_ty_for_fig_pbc(spins)
        all_ty_num = test.all_ty_num(type_lat)
        
        for ty in all_ty:
            if ty not in all_ty_num.keys():
               all_ty_num[ty] = 0 

        t1.append(100*all_ty_num[1]/v) 
        t2.append(100*all_ty_num[2]/v)
        t3.append(100*(all_ty_num[31] + all_ty_num[32])/v)
        t4.append(100*(all_ty_num[41] + all_ty_num[42])/v)
        
    #test.plot_lat(spins, None)
    return t1, t2, t3, t4

#ratio-height
ty1_h, ty2_h, ty3_h, ty4_h = [], [], [], []
n, theta, t = 10, 0, .1
times = 4

dh = 41
hs = np.linspace(0.0, 1.0, dh)
for i in range(dh): #hs loading bar
    all_t = time_ave_ty_num_(n, hs[i], theta, t, times)

    ty1_h.append(all_t[0])
    ty2_h.append(all_t[1])
    ty3_h.append(all_t[2])
    ty4_h.append(all_t[3])


msize = 16
for i in range(dh):
    x = [hs[i] for j in range(times)]
    sizes = [msize for j in range(times)]
    plt.scatter(x, ty1_h[i], sizes, color = 'blue', marker = 'o')
    plt.scatter(x, ty2_h[i], sizes, color = 'red' , marker = '*')
    plt.scatter(x, ty3_h[i], sizes, color = 'green', marker = 's')
    plt.scatter(x, ty4_h[i], sizes, color = 'black', marker = 'x')

plt.xlim(0.0, 1.0)
plt.ylim(0, 100)
plt.show() 
'''