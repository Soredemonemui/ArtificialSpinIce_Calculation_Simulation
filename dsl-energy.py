# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:53:08 2024

@author: Collo
"""

import sys
sys.path.append(r'D:\所有科研之类的东西\python_test\new_visual\square\tracking_monopoles')
from track_monopole import square_kiri
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


h = 0.0
n = 25
t = .2
theta = np.radians(0)

H_mag = 2.4
H_ang = -1*np.pi/4
H_vec = H_mag*np.array([np.cos(H_ang), np.sin(H_ang), 0])

def ds_energy(x, a, b, c):
    return a*x - b/x + c


# lattice_X
test = square_kiri(n, h, theta, t, H_vec) #class
#spins = test.spin_lattice() #initial spins
#E_TypeI = sum(test.spin_energies(spins).values())
E_TypeI = -204105.2802077355 #n=100,k=5
#E_TypeI = -203214.42427689128 #n=100,k=infty

#print(f'E_TypeI = {E_TypeI}w')
'''
start = time.time()
#prl fig3 (ds energy)
dsl = np.linspace(12, 98, 44)
E_DS_r = []
for i in dsl:
    length_ds = int(i)
    loc = (int(n/2)-int(length_ds/2),int(n/2)-1)
    spins = test.spin_lattice() # initialize
    for j in range(length_ds):
        spins = test.intro_mono(spins, (loc[0]+j,loc[1]), [1,1])
        spins = test.intro_mono(spins, (loc[0]+1+j,loc[1]), [0,1])
    spins = test.intro_mono(spins, (loc[0],loc[1]), [0,1])
    energies = test.spin_energies(spins)
    E_DS_r.append(sum(energies.values()) - E_TypeI)
print(f'E_DS_r in dsl={dsl}->{E_DS_r}')
end = time.time()
print(f'time_cost = {end-start}s')
'''

'''
x = np.array([2,4,6,8,10,20,30,40,50,60,70,80,90,100])
y = np.array([36.96465238917153, 51.989698426041286, 66.76024446290103, 
              81.53079049973167, 96.3013365365041, 170.15406672045356, 244.0067969043739,
              317.8595270883234, 391.71225727160345, 465.5649874548253, 539.4177176380763,
              613.270447821269, 687.1231780046073, 760.9759081880038])
popt, pcov = curve_fit(ds_energy, x, y)
print(popt)

plt.scatter(x, y, label = 'Data')
plt.plot(x, ds_energy(x, *popt), color = 'red', label = 'Fitted Data')
plt.legend()
plt.show()
'''
