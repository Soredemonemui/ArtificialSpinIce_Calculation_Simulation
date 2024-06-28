# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 07:26:03 2024

@author: Collo
"""

from math import pi, sin, cos, sqrt
from funda_func import sp_loc, E, mag, find_key_by_value
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio.v2 as imageio
from PIL import Image
import os


line_num = 5
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

class SASI_line(object):

    def __init__(self, a, h, r, T, H):
        self.a = a #表示自旋的数量N
        self.h = h #lifting_height h
        self.r = r #角度rotation angle
        self.T = T #温度
        self.H = H #external magnetic field
        
    def spin_lattice(self):
        #X形状square ice
        all_spins = {}
        for k in range(self.a):
            loc_h = (k, 0, 0)
            all_spins[loc_h] = 0 + self.r
        for m in range(self.a-1):
            loc_u, loc_d = ((2*m+1)/2, 1/2, 0), ((2*m+1)/2, -1/2, 0) 
            all_spins[loc_u] = pi/2 - self.r
            all_spins[loc_d] = pi/2 - self.r
            
        for i in range(line_num):
            for k in range(self.a):
                loc_hu, loc_hd = (k, -(i+1), 0), (k, i+1, 0)
                all_spins[loc_hu] = 0 + self.r
                all_spins[loc_hd] = 0 + self.r
            for m in range(self.a-1):
                loc_u, loc_d = ((2*m+1)/2, 1/2 + (i+1), 0), ((2*m+1)/2, -1/2 - (i+1), 0)
                all_spins[loc_u] = pi/2 - self.r
                all_spins[loc_d] = pi/2 - self.r
        return all_spins

    def boundary(self, spins):
        b_spins = {}
        key_y = []
        for key in spins.keys():
            key_y.append(key[1])
        for key in spins.keys():
            if key[1] == max(key_y) or key[1] == min(key_y):
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

    def intro_mono(self, spins, x):
        #loc = location of vertex (i, j) = 0 ~ self.a
        #num = determining the spin in the vertex spin cluster = (00, 01, 10, 11)
        #key = (x*sp_loc(self.r), 0, self.h)
        key = (x, 0, 0)
        spins[key] += pi
        return spins

    def delta_E(self, key_spin, spins):
        energy = -np.dot(self.H, mag(spins[key_spin]))
        locs = list(spins.keys())        
        for loc in locs:
            if loc != key_spin:
                energy += E(np.array(loc), np.array(key_spin),
                            mag(spins[loc]), mag(spins[key_spin]))
        return energy/(sqrt(2)*sp_loc(self.r))**3
    
    def spin_energies(self, spins):
        energies = {}
        for key in spins.keys():
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

         
    def plot_lat(self, lattice, num):
        mag_H = round(np.linalg.norm(self.h), 1)
        #theta = round(np.degrees(self.r),1)

        scale = 1.0
        length = 0.4*scale
        head = 0.3*scale
        width = 0.1*scale
        
        plt.figure(figsize = [self.a, 6], dpi = 72)
        
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
            
    def plot_lat_color(self, lattice, energies, num):
        mag_H = round(np.linalg.norm(self.h), 1)
        #theta = round(np.degrees(self.r),1)
        e_max, e_min = 2.2, -5.5

        scale = 1.0
        length = 0.4*scale
        head = 0.35*scale
        width = 0.07*scale

        
        fig = plt.figure(figsize = [self.a, 6], dpi = 72)
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


        if num == None:
            #plt.colorbar(plt.cm.ScalarMappable(cmap='turbo'), label='Energy')
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # 更新颜色条的数据
            plt.title(f'{round(np.degrees(self.r),3)}°')
            plt.colorbar(sm, ax=ax, label='Energy')
            plt.axis('equal')
            plt.show()
        else:
            #plt.colorbar(plt.cm.ScalarMappable(cmap='turbo'), label='Energy')
            plt.axis('equal')
            plt.autoscale(enable = 'True', axis = 'both')
            plt.savefig(f'line,colorful_spin_lat_[{self.a, self.h, self.T, mag_H, num}].png')
            plt.close()
            
    def energy_landscape_line(self, spins):
        #diagonal
        unit = 1
        font = {'size': 20}
        Es = []
        plt.figure(figsize = [8, 6], dpi = 200)
        for i in range(self.a):
            #key = (i*unit, 0, self.h)
            key = (i*unit, 0, 0)
            Es.append(self.delta_E(key, spins))
        print(Es)
        plt.plot(range(self.a), Es, marker = 'o', markersize = 4, \
                 mec = 'k', mfc = 'steelblue', color = 'darkgreen')
        plt.title(f'rotate={round(np.degrees(self.r),3)}°, H={round(self.H[0],3),round(self.H[1],3)}')
        plt.xlabel('r[a]',fontproperties=font, rotation = 0, x = 0.95, labelpad = -15)
        plt.ylabel('E[w]',fontproperties=font, rotation = 0, y = 0.9, labelpad = -50)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.xlim(0, self.a-1)
        plt.ylim(-5.2, 2)
        plt.show()    

n = 50
H_mag = 0.0
H_ang = -1*np.pi/4
H_vec = H_mag*np.array([np.cos(H_ang), np.sin(H_ang), 0])
theta = np.radians(0)
ds_len= 5
h = 0
t = 0.2

if __name__ == '__main__':

    test = SASI_line(n, h, theta, t, H_vec)
    spins = test.spin_lattice()
    bs = test.boundary(spins)
    for i in range(-ds_len, ds_len+1):
        test.intro_mono(spins, int(n/2+i))
    #test.plot_lat(spins, None)
    #energies = test.spin_energies(spins)
    #test.plot_lat_color(spins, energies, 0)
    #test.plot_lat_color(spins, energies, None)
    test.energy_landscape_line(spins)
    
    '''
    mc_steps = 5
    flips = 3*mc_steps*n
    fig_nums = [0]
    for i in range(flips):
        #flip = test.flip_in(spins, bs) #only local interactions considered, bs=frozen
        flip = test.flip_in(spins, energies, bs)

        #flip = test.flip(spins) #all interactions considered
        spins, f_or_n = flip[0], flip[1]
    
        if f_or_n == 1:
        #if (i % 500) == 0:
            energies = test.spin_energies(spins)
            test.plot_lat_color(spins, energies, i+1) #what you want to plot
            #test.plot_lat(spins, i+1)
            fig_nums.append(i+1)
            
    with imageio.get_writer(uri=f'2-2, n={n}, H={H_mag}({round(np.degrees(H_ang))}), h={h}, t={t}, theta={np.degrees(theta)}°.gif', \
                            mode='I', fps=5, loop = 1) as writer:
        for i in fig_nums:
            img = Image.open(f'line,colorful_spin_lat_[{n, h, t, H_mag, i}].png')
            img = img.convert('P', palette=Image.ADAPTIVE, colors=256)  # 使用自适应调色板
            writer.append_data(imageio.imread(f'line,colorful_spin_lat_[{n, h, t, H_mag, i}].png'))
            os.remove(f'line,colorful_spin_lat_[{n, h, t, H_mag, i}].png')
    '''
    