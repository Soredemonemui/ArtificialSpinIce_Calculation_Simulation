from math import pi, sin, cos, sqrt
from funda_func import sp_loc, E, mag, find_key_by_value
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from tqdm import trange
from PIL import Image
import matplotlib.colors as mcolors
import time
import pickle


#绘图的初始化设置
#磁单极子
mono_properties = {
    31: ('red', 0.3),
    32: ('blue', 0.3),
    41: ('red', 0.5),
    42: ('blue', 0.5)
    }
#type_color
ty_colors = {
    1: 'slateblue', 21: 'lightcoral', 22: 'lightcoral', 
    23: 'lightcoral', 24: 'lightcoral', 31: 'limegreen', 
    32: 'limegreen', 41: 'limegreen', 42: 'limegreen'
    }
colors = [(0, 'royalblue'), (0.25, 'deepskyblue'), (0.5, 'limegreen'), (0.75, 'yellow'), (1, 'crimson')]
cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)
cmap = mcolors.ListedColormap(cmap(np.linspace(0, 1, 256)) * 0.95)

#初始自旋构造
angle_map = {
    (0, 0): lambda r: 1*pi/4 - r,
    (1, 1): lambda r: 1*pi/4 - r,
    (0, 1): lambda r: -1*pi/4 + r,
    (1, 0): lambda r: -1*pi/4 + r
    }

class square_kiri(object):

    def __init__(self, a, h, r, T, H):
        self.a = a #表示自旋的数量N = 4a^2
        self.h = h #lifting_height h
        self.r = r #角度rotation angle
        self.T = T #温度
        self.H = H #external magnetic field
        
    def ver_unit(self, k):
        # 初始化生成一个2×2 vertex 自旋晶格（所有自旋随机取向)
        # k 表示该 ver_unit 的位置矢量
        # spins 的 key 表示位置，value 表示方位角度
        spins = {}    
        for i in range(2):
            for j in range(2):
                if (i + j) % 2 == 0:
                    loc_s = ((i + 2*k[0])*sp_loc(self.r),
                             (j + 2*k[1])*sp_loc(self.r),
                             self.h)
                else:
                    loc_s = ((i + 2*k[0])*sp_loc(self.r),
                             (j + 2*k[1])*sp_loc(self.r),
                             0)
            
                # 使用字典映射计算角度
                spins[loc_s] = angle_map[(i, j)](self.r)
    
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

    def boundary(self, spins):
        b_spins = {}
        key_x, key_y = [], []
        for key in spins.keys():
            key_x.append(key[0])
            key_y.append(key[1])
        for key in spins.keys():
            if key[0] == min(key_x) or key[1] == min(key_y):
                b_spins[key] = spins[key]
            elif key[0] == max(key_x) or key[1] == max(key_y):
                b_spins[key] = spins[key]
        return b_spins
    
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
                        loc1 = ((x+i)*unit, (y+j)*unit, z1)
                        loc2 = loc1
                    else:
                        loc1 = ((x+i)*unit, (y+j)*unit, z2)
                        loc2 = loc1
                    if loc1 in spins.keys():
                        surr[loc2] = spins[loc1]
                else:
                    continue
        return surr

    def intro_mono(self, spins, loc, num):
        #loc = location of vertex (i, j) = 0 ~ self.a
        #num = determining the spin in the vertex spin cluster = (00, 01, 10, 11)
        if sum(num) % 2 == 0:
            key = ((num[0] + 2*loc[0])*sp_loc(self.r),(num[1] + 2*loc[1])*sp_loc(self.r),self.h)
        else:
            key = ((num[0] + 2*loc[0])*sp_loc(self.r),(num[1] + 2*loc[1])*sp_loc(self.r),0)
        spins[key] += pi
        return spins
    
    def DS_1(self, spins, length_ds):
        # moving monopole to build DS
        #1-2A & 2A-1 ds
        loc = (int(self.a/2)-int(length_ds/2)-1,int(self.a/2)-1)
        for i in range(length_ds):
            spins = self.intro_mono(spins, (loc[0]+i,loc[1]), [1,1])
            spins = self.intro_mono(spins, (loc[0]+1+i,loc[1]), [0,1])
        spins = self.intro_mono(spins, (loc[0],loc[1]), [0,1])
        #spins = test.intro_mono(spins, (loc[0]+length_ds,loc[1]), [1,1])
        #moving downwards
        #for i in range(length_ds):
            #spins = test.intro_mono(spins, (loc[0]+length_ds,loc[1]-i), [0,0])
            #spins = test.intro_mono(spins, (loc[0]+length_ds,loc[1]-1-i), [0,1])
        return spins
    
    def DS_2(self, spins, length_ds):
        #2-2 ds
        loc = (int(self.a/2)-int(length_ds/2),int(self.a/2)-int(length_ds/2))
        for i in range(length_ds):
            spins = self.intro_mono(spins, (loc[0]+i,loc[1]+i), [1,1])
            spins = self.intro_mono(spins, (loc[0]+i,loc[1]+i), [0,0])
            #spins = test.intro_mono(spins, (loc[0]-1-i,loc[1]-1-i), [1,1])
            #spins = test.intro_mono(spins, (loc[0]-1-i,loc[1]-1-i), [0,0])
            #spins = test.intro_mono(spins, (loc[0]+length_ds,loc[1]+length_ds), [1,1])
        spins = self.intro_mono(spins, (loc[0]+length_ds,loc[1]+length_ds), [0,0])
        #for i in range(length_ds):
            #spins = test.intro_mono(spins, (loc[0]-4-i,loc[1]-3+i), [1,0])
            #spins = test.intro_mono(spins, (loc[0]-4-i,loc[1]-3+i), [0,1])
        return spins
    
    def DS_3(self, spins, length_ds):
        #2B-1 DS
        loc = (int(self.a/2)-1,int(self.a/2)-int(length_ds/2)-1)
        for i in range(length_ds+1):
            spins = self.intro_mono(spins, (loc[0],loc[1]+i), [1,1])
            spins = self.intro_mono(spins, (loc[0],loc[1]+1+i), [1,0])
        spins = self.intro_mono(spins, (loc[0],loc[1]), [1,1])
        return spins

    def delta_E(self, key_spin, spins):
        energy = -np.dot(self.H, mag(spins[key_spin]))
        locs = list(spins.keys())        
        for loc in locs:
            if loc != key_spin:
                energy += E(np.array(loc), np.array(key_spin),
                            mag(spins[loc]), mag(spins[key_spin]))
        return energy
    
    def delta_E_local(self, key_spin, spins):
        energy = -np.dot(self.H, mag(spins[key_spin]))
        k = 5
        surr = self.surr_k(spins, key_spin, k)
        locs = list(surr)        
        for loc in locs:
            energy += E(np.array(loc), np.array(key_spin),
                        mag(surr[loc]), mag(spins[key_spin]))
        return energy
    
    def spin_energies(self, spins):
        energies = {}
        for key in spins.keys():
            #energies[key] = self.delta_E_local(key, spins)
            energies[key] = self.delta_E(key, spins)
        return energies
    
    def p_acc(self, energy):
        #计算接受翻转的概率
        if energy > 0:
            return 1
        return np.exp(2*energy/self.T)

    def flip(self, spins, energies):
        #随机选一个自旋，判定其是否反转
        #计算所有自旋的能量，每次翻转后更新
        key = random.choice(list(spins.keys()))
        #energy = self.delta_E(key, spins)
        energy = energies[key]
        p = random.random()
        if p < self.p_acc(energy):
            spins[key] = (spins[key] + pi) % (2*pi)
            return (spins,1)
        return (spins,0)
        
    def flip_in(self, spins, energies, boundary):
        #随机选取一个非边界的自旋判定翻转
        #计算所有自旋的能量，每次翻转后更新
        while True:
            key = random.choice(list(spins.keys()))
            if key not in boundary:
                break
        energy = energies[key]
        
        #energy = self.delta_E(key, spins)
        p = random.random()
        if p < self.p_acc(energy):
            spins[key] = (spins[key] + pi) % (2*pi)
            return (spins, 1)
        else:
            return (spins, 0)


    def flip_local(self, spins):
        #随机选一个自旋，判定其是否反转
        #每次选中一个自旋后计算其能量，判定是否反转
        key = random.choice(list(spins.keys()))
        
        energy = self.delta_E_local(key, spins)
        p = random.random()
        if p < self.p_acc(energy):
            spins[key] = (spins[key] + pi) % (2*pi)
            return (spins,1)
        else:
            return (spins,0)      
    
    def flip_in_local(self, spins, boundary):
        #随机选取一个非边界的自旋判定翻转
        #每次选中一个自旋后计算其能量，判定是否反转
        while True:
            key = random.choice(list(spins.keys()))
            if key not in boundary:
                break
        
        energy = self.delta_E_local(key, spins)
        p = random.random()
        if p < self.p_acc(energy):
            spins[key] = (spins[key] + pi) % (2*pi)
            return (spins, 1)
        else:
            return (spins, 0)

    def get_spin_cluster(self, spins, vertex):
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

    def vertex_type_(self, cluster, vertex):
        #获取一个vertex的Type
        #self.r = np.radians(90)时不适用
        #type 2 split-> type2 + type 5
        keys = list(cluster.keys())
        keys = sorted(keys, key = lambda x: (x[0], x[1]))        
        in_or_out = {} #in gives 1, out gives -1
           
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

        loc1 = np.array([keys[0][0], keys[0][1], 0]) #left lower
        #loc2 = np.array([keys[1][0], keys[1][1], 0]) #left upper
        loc3 = np.array([keys[2][0], keys[2][1], 0]) #right lower
        loc4 = np.array([keys[3][0], keys[3][1], 0]) #right upper

        m1 = mag(cluster[keys[0]])
        #m2 = mag(cluster[keys[1]])
        m3 = mag(cluster[keys[2]])
        m4 = mag(cluster[keys[3]])
        
        if abs(charge) == 4:
            return 4
        else:
            if charge == 0:
                if E(loc1, loc4, m1, m4) > 0:
                    return 1
                else:
                    if E(loc1, loc3, m1, m3) > 0:
                        return 5
                    else:
                        return 2
            else:
                if abs(charge) == 2:
                    return 3
                else:
                    raise ValueError(f"Vertex {vertex} is illegal! \n Cluster = {cluster}, charge = {charge}")
    
    def all_type_num(self, spins):
        ty = {}
        for i in range(self.a):
            for j in range(self.a):
                vertex_1 = ((1/2 + 2*i)*sp_loc(self.r),
                            (1/2 + 2*j)*sp_loc(self.r))
                cluster_1 = self.get_spin_cluster(spins, vertex_1)
                ty[vertex_1] = self.vertex_type(cluster_1, vertex_1)
                
                if i < self.a - 1 and j < self.a - 1:
                    vertex_2 = ((3/2 + 2*i)*sp_loc(self.r),
                                (3/2 + 2*j)*sp_loc(self.r))
                    cluster_2 = self.get_spin_cluster(spins, vertex_2)
                    ty[vertex_2] = self.vertex_type(cluster_2, vertex_2)

        ty_count = {}
        
        #initializing the number of four types
        for i in range(1, 5):
            ty_count[i] = 0

        for value in list(ty.values()):
            ty_count[value] = list(ty.values()).count(value)
            
        return ty_count

    def new_all_vty(self, spins):
        ty = {}
        if self.r <= np.radians(45):
            #vertices
            for i in range(self.a):
                for j in range(self.a):
                    vertex_1 = ((1/2 + 2*i)*sp_loc(self.r),
                                (1/2 + 2*j)*sp_loc(self.r))
                    cluster_1 = self.get_spin_cluster(spins, vertex_1)
                    ty[vertex_1] = self.vertex_type(cluster_1, vertex_1)
                    
                    if i < self.a - 1 and j < self.a - 1:
                        vertex_2 = ((3/2 + 2*i)*sp_loc(self.r),
                                    (3/2 + 2*j)*sp_loc(self.r))
                        cluster_2 = self.get_spin_cluster(spins, vertex_2)
                        ty[vertex_2] = self.vertex_type(cluster_2, vertex_2)
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
                    ty[vertices_3] = self.vertex_type(cluster_3, vertices_3)
                    ty[vertices_4] = self.vertex_type(cluster_4, vertices_4)
                    
        #initializing the number of four types
        ty_count = {}
        for i in range(1, 5):
            ty_count[i] = 0

        for value in list(ty.values()):
            ty_count[value] = list(ty.values()).count(value)

        #print(ty)
        return ty_count

    def all_ty_for_fig(self, spins):
        ty = {}
        if self.r <= np.radians(45):
            #vertices
            for i in range(self.a):
                for j in range(self.a):
                    vertex_1 = ((1/2 + 2*i)*sp_loc(self.r),
                                (1/2 + 2*j)*sp_loc(self.r))
                    cluster_1 = self.get_spin_cluster(spins, vertex_1)
                    ty[vertex_1] = self.vertex_type_fig(cluster_1, vertex_1)
                    
                    if i < self.a - 1 and j < self.a - 1:
                        vertex_2 = ((3/2 + 2*i)*sp_loc(self.r),
                                    (3/2 + 2*j)*sp_loc(self.r))
                        cluster_2 = self.get_spin_cluster(spins, vertex_2)
                        ty[vertex_2] = self.vertex_type_fig(cluster_2, vertex_2)
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
                    ty[vertices_3] = self.vertex_type_fig(cluster_3, vertices_3)
                    ty[vertices_4] = self.vertex_type_fig(cluster_4, vertices_4)
        return ty

    def all_ty_for_fig_all(self, spins):
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
         
    def plot_lat(self, lattice, num):
        mag_H = round(np.linalg.norm(self.h), 1)
        #theta = round(np.degrees(self.r),1)

        scale = 1.3
        length = 0.4*scale
        head = 0.3*scale
        width = 0.1*scale
        
        plt.figure(figsize = [15, 15], dpi = 72)
        
        for key in lattice.keys():
            # lifted or not
            if key[2] != 0:
                ls = '--'
            else:
                ls = '-'

            # magnetification visualization
            if cos(lattice[key]) > 1e-3:
                color = 'w'
            elif cos(lattice[key]) < -1e-3:
                color = 'w'
            else:
                if sin(lattice[key]) > 1e-3:
                    color = 'w'
                else:
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
            
    def plot_lat_color(self, lattice, energies, type_lat, num):
        mag_H = round(np.linalg.norm(self.h), 1)
        #theta = round(np.degrees(self.r),1)
        e_max, e_min = 2.2, -5.5

        scale = 1.3
        length = 0.4*scale
        head = 0.35*scale
        width = 0.07*scale
        
        ty_keys = type_lat.keys()
        
        fig = plt.figure(figsize = [18, 15], dpi = 72)
        ax = fig.add_subplot(111)
        #cmap = plt.cm.turbo      

        norm = mcolors.Normalize(vmin=e_min, vmax=e_max, clip=True)
        #spin_energies = energies.values()
        #e_max, e_min = max(spin_energies), min(spin_energies)
        #print(e_max, e_min)
        
        for key in lattice.keys():
            # lifted or not
            if key[2] != 0:
                ls = '--'
            else:
                ls = '-'
            
            #fc = plt.cm.turbo((energies[key]-e_min)/(e_max-e_min))
            fc = cmap(norm(energies[key]))
            ax.arrow(key[0] - length*cos(lattice[key]), key[1] - length*sin(lattice[key]), \
                     2*length*cos(lattice[key]), 2*length*sin(lattice[key]), head_length = head, \
                     head_width = head, fc = fc, ec = fc, linestyle = ls, width = width, \
                     length_includes_head = True)

        for key in ty_keys:
            type_value = type_lat.get(key)
            if type_value in mono_properties:
                mono_c, mono_r = mono_properties[type_value]
                circle = plt.Circle(xy = (key[0], key[1]), radius = mono_r, fc = mono_c, ec = 'white')
                ax.add_patch(circle)

        if num == None:
            #plt.colorbar(plt.cm.ScalarMappable(cmap='turbo'), label='Energy')
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # 更新颜色条的数据
            plt.colorbar(sm, ax=ax, label='Energy')
            plt.axis('equal')
            plt.show()
        else:
            #plt.colorbar(plt.cm.ScalarMappable(cmap='turbo'), label='Energy')
            plt.autoscale(enable = 'True', axis = 'both')
            plt.savefig(f'obc,colorful_spin_lat_[{self.a, self.h, self.T, mag_H, num}].jpg')
            plt.close()

    def plot_lat_ty_mono(self, lattice, type_lat, num):
        #mag_H = round(np.linalg.norm(self.h), 1)
        #theta = round(np.degrees(self.r),1)

        scale = 1.2
        length = 0.4*scale
        head = 0.35*scale
        width = 0.13*scale
        unit = sp_loc(self.r)
        
        fig = plt.figure(figsize = [15, 15], dpi = 72)
        ax = fig.add_subplot(111)

        ty_keys = type_lat.keys()
        
        # draw type 
        for key in ty_keys:
            if type_lat[key] == 1:
                ty_c = 'slateblue'
            elif str(type_lat[key])[0] == '2':
                ty_c = 'lightcoral'
            elif type_lat[key] == 31:
                ty_c = 'limegreen'                
            elif type_lat[key] == 32:
                ty_c = 'limegreen'               
            elif type_lat[key] == 41:
                ty_c = 'limegreen'
            elif type_lat[key] == 42:
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
            if cos(lattice[key]) > 1e-3:
                color = 'w'
            elif cos(lattice[key]) < -1e-3:
                color = 'w'
            else:
                if sin(lattice[key]) > 1e-3:
                    color = 'w'
                else:
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
            plt.autoscale(enable = 'True', axis = 'both')
            plt.savefig(f'spin_ty_lat_[{self.a, self.h, self.T, num}].png')
            plt.close()

    def plot_all_ty_lat(self, type_lat, num):
        mag_H = round(np.linalg.norm(self.H), 1)
        #theta = round(np.degrees(self.r),1)
        
        unit = sp_loc(self.r)
        scale = 1.2
        length = 0.5*scale
        head = 0.45*scale
        width = 0.15*scale
        
        fig = plt.figure(figsize = [15, 15], dpi = 72)
        ax = fig.add_subplot(111)
  
        val_ang_map = {21: 0, 22: np.pi, 23: np.pi/2, 24: 3*np.pi/2}
        
        for key, value in type_lat.items():
            #draw type
            if value in ty_colors:
                ty_c = ty_colors[value]
                square = plt.Rectangle(xy = (key[0], key[1]-unit), width = sqrt(2)*unit, \
                                       height = sqrt(2)*unit, angle = 45, ec = 'white', fc = ty_c)
                ax.add_patch(square)
            
            #draw type II
            if str(value)[0] == '2':
                fc = 'white'
                ec =  'k'
                angle = val_ang_map[value]
                ax.arrow(key[0] - length*cos(angle), key[1] - length*sin(angle), \
                         2*length*cos(angle), 2*length*sin(angle), \
                         head_length = head, head_width = head, fc = fc, ec = ec, linestyle = '-', \
                         width = width, length_includes_head = True)
            
            #draw monopole
            if value in mono_properties:
                mono_c, mono_r = mono_properties[value]
                circle = plt.Circle(xy = (key[0], key[1]), radius = mono_r, fc = mono_c, ec = 'white')
                ax.add_patch(circle)
                
        #plt.xticks(fontsize = 24)
        #plt.yticks(fontsize = 24)
        if num == None:
            plt.title(f'{round(np.degrees(self.r),3)}°, H={round(self.H[0],3),round(self.H[1],3)}')
            plt.axis('equal')
            plt.show()
        else:
            plt.autoscale(enable = 'True', axis = 'both')
            plt.title(f'{round(np.degrees(self.r),3)}°, H={round(self.H[0],3),round(self.H[1],3)}')
            plt.savefig(f'obc,spin_ty_lat_[{self.a, self.h, self.T, mag_H, num}].png')
            plt.close()  
        
    def energy_landscape_line_h(self, spins, line):
        unit = sp_loc(self.r)
        font = {'size': 20}
        Es = []
        plt.figure(figsize = [8, 6], dpi = 200)
        for i in range(1,2*self.a-1):
            if i % 2 == 0:
                key = (i*unit, line*unit, self.h)
            else:
                key = (i*unit, line*unit, 0)
            Es.append(self.delta_E_local(key, spins))
        #print(Es)
        plt.plot(range(1,2*self.a-1), Es, marker = 's', markersize = 3, mec = 'k', mfc = 'mediumseagreen')
        plt.title(f'{round(np.degrees(self.r),3)}°')
        plt.xlabel('r[a]',fontproperties=font, rotation = 0, x = 0.95, labelpad = -15)
        plt.ylabel('E[w]',fontproperties=font, rotation = 0, y = 0.9, labelpad = -50)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.xlim(1, 39)
        plt.ylim(-5.2, 2)
        plt.show()
        
    def energy_landscape_line_hh(self, spins, line):
        unit = sp_loc(self.r)
        font = {'size': 20}
        Es = []
        plt.figure(figsize = [8, 6], dpi = 200)
        for i in range(1,2*self.a-1):
            if i % 2 == 0:
                key = (line*unit, i*unit, self.h)
            else:
                key = (line*unit, i*unit, 0)
            Es.append(self.delta_E_local(key, spins))
        #print(Es)
        plt.plot(range(1,2*self.a-1), Es, marker = 's', markersize = 3, mec = 'k', mfc = 'mediumseagreen')
        plt.title(f'{round(np.degrees(self.r),3)}°')
        plt.xlabel('r[a]',fontproperties=font, rotation = 0, x = 0.95, labelpad = -15)
        plt.ylabel('E[w]',fontproperties=font, rotation = 0, y = 0.9, labelpad = -50)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.xlim(1, 39)
        plt.ylim(-5.2, 2)
        plt.show()     
        
    def energy_landscape_line_d(self, spins):
        #diagonal
        unit = sp_loc(self.r)
        font = {'size': 20}
        Es = []
        plt.figure(figsize = [8, 6], dpi = 200)
        for i in range(2*self.a):
            key = (i*unit, i*unit, self.h)
            #Es.append(self.delta_E_local(key, spins))
            Es.append(self.delta_E(key, spins))
        print(Es)
        plt.plot(range(2*self.a), Es, marker = 'o', markersize = 4, \
                 mec = 'k', mfc = 'steelblue', color = 'darkgreen')
        plt.title(f'rotate={round(np.degrees(self.r),3)}°, H={round(self.H[0],3),round(self.H[1],3)}')
        plt.xlabel('r[a]',fontproperties=font, rotation = 0, x = 0.95, labelpad = -15)
        plt.ylabel('E[w]',fontproperties=font, rotation = 0, y = 0.9, labelpad = -50)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.xlim(0, 2*self.a-1)
        plt.ylim(-5.2, 2)
        plt.show() 






if __name__ == '__main__':    
    print(f"Process ID: {os.getpid()}")
    print("Code is running...")
    h = 0.0
    n = 25
    t = .2
    theta = np.radians(0)

    H_mag = 0
    H_ang = -1*np.pi/4
    H_vec = H_mag*np.array([np.cos(H_ang), np.sin(H_ang), 0])
    
    # lattice_X
    test = square_kiri(n, h, theta, t, H_vec) #class
    spins = test.spin_lattice() #initial spins
    
    length_ds = 5
    loc = (int(n/2)-int(length_ds/2),int(n/2)-1)
    
    '''
    spins = test.DS_2(spins, length_ds)
    type_lat = test.all_ty_for_fig_all(spins)
    bs = test.boundary(spins)
    test.plot_all_ty_lat(type_lat, 0)
    mc_steps = 5
    flips = 4*mc_steps*n**2
    for i in trange(flips):
        flip = test.flip_in_local(spins, bs)
        spins = flip[0]
    type_lat = test.all_ty_for_fig_all(spins)
    test.plot_all_ty_lat(type_lat, 1)   
    '''
       
    #produce gif 
    spins = test.DS_2(spins, length_ds)
    type_lat = test.all_ty_for_fig_all(spins)
    #print(type_lat)
    bs = test.boundary(spins)
    #energies = test.spin_energies(spins)
    #test.plot_lat_color(spins, energies, type_lat, None)
    
    
    #vertex = find_key_by_value(type_lat, 31)
    #print(vertex)
    #monopole_clu = test.get_spin_cluster(spins, vertex).keys()
    #print(monopole_clu)
    #print(test.get_spin_cluster(spins, vertex))
    

    #print(list(energies.values()))
    test.plot_lat(spins, None) #draw initial state fig
    
    #test.plot_lat_ty_mono(spins, type_lat, None)
    #test.plot_all_ty_lat(type_lat, 0)
    #test.energy_landscape_line_h(spins,int(1+2*loc[1]))
    #test.energy_landscape_line_hh(spins,int(1+2*loc[0]))
    #test.energy_landscape_line_d(spins)
    
    '''
    mc_steps = 5
    flips = 4*mc_steps*n**2
    fig_nums = [0]
    for i in trange(flips):
        flip = test.flip_in_local(spins, bs) #only local interactions considered, bs=frozen
        #flip = test.flip_in(spins, energies, bs)
            
        #flip = test.flip_local(spins) #only local interactions considered
        #flip = test.flip(spins) #all interactions considered
        spins = flip[0]
        f_or_n = flip[1]
    
        if f_or_n == 1:
        #if (i % 500) == 0:
            type_lat = test.all_ty_for_fig_all(spins)
            #energies = test.spin_energies(spins)
            #test.plot_lat_color(spins, energies, type_lat, i+1) #what you want to plot
            test.plot_all_ty_lat(type_lat, i+1)
            #test.plot_lat_ty_mono(spins, type_lat, i+1)
            fig_nums.append(i+1)
            #type_num = list(type_lat.values())
    
    with imageio.get_writer(uri=f'2-2, n={n}, H={H_mag}({round(np.degrees(H_ang))}), h={h}, t={t}, theta={np.degrees(theta)}°.gif', \
                            mode='I', fps=5, loop = 1) as writer:
        for i in fig_nums:
            img = Image.open(f'obc,spin_ty_lat_[{n, h, t, H_mag, i}].png')
            img = img.convert('P', palette=Image.ADAPTIVE, colors=256)  # 使用自适应调色板
            writer.append_data(imageio.imread(f'obc,spin_ty_lat_[{n, h, t, H_mag, i}].png'))
            os.remove(f'obc,spin_ty_lat_[{n, h, t, H_mag, i}].png')
    
    with open(f'2-2, n={n}, H={H_mag}({round(np.degrees(H_ang))}), h={h}, t={t}, theta={np.degrees(theta)}°, mcs={mc_steps}.pickle', 'wb') as f:
        pickle.dump(spins, f)
    '''
    
    