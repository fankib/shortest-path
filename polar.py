# Some points in Free Gliding:


# compute some full speed glides:
t0 = 15*3600+10*60+40
h0 = 2486

t1 = 15*3600+18*60+59
h1 = 1758
d1 = 6600

t2 = 15*3600+25*60+59
h2 = 883
d2 = 7300+1000

def stats(h, d, t):
    s = d/t
    g = d/h
    print('from 0 to 1', s, 'ms', s*3.6, 'kmh', g, 'gr')

# diff 2 to 0:
#stats(h0-h2, d1+d2, t2-t0)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# logarithmic glide curve (maybe quadratic is better)
def func_exp(x, a, b, c):
    return -np.exp(a*(x-b))+c

def func_quad(x, a, b, c):
    return -(a*(x-b))**2 + c

def func_linear(x, a, b, c):
    return -a*x + c

func = func_exp
#func = func_quad
#func = func_linear

#xdata = np.linspace(0, 60, 100)


glide = np.array([11, 9.7, 9.1, 12.3, 8.6, 8.8, 11.1, 11, 10.2, 8.9, 9.3, 8.8, 13.6, 8.6, 9.2, 6.5])
speed = np.array([42, 54, 45, 51, 46, 47, 56, 44, 51, 49, 46, 50, 38, 54, 58, 63])

# sort by speed:
args = np.argsort(speed)
glide = glide[args]
speed = speed[args]


popt, pcov = curve_fit(func, speed, glide, bounds=([0.00, 35, 0], [1.0, 60, 15]))
print('popt', popt)

# print glides:
print('glides', [(s, func(s, *popt)) for s in [40, 45, 50, 55, 60]])


plt.figure()
plt.scatter(speed, glide)
xdata = np.linspace(30, 62, 1000)
plt.plot(xdata, func(xdata, *popt))

plt.gca().set_xlim(0, 65)
plt.gca().set_ylim(0, 15)

#plt.show()

# use this polars to compute 10 km speeds
plt.figure()
thermals = np.linspace(0.5, 3.5, 20)
#thermals = np.logspace(0.8, 2.5, 10)
for thermal in thermals:
    #gas = np.array([35, 40, 45, 50, 55, 60])
    gas = xdata
    glide = func(gas, *popt)

    t_glide = 10./gas*60
    height = 10000/glide
    t_height = height/thermal/60
    t_total = t_glide + t_height

    min_arg = np.argmin(t_total)
    min_x = gas[min_arg]
    min_y = t_total[min_arg]
    lim_y = t_total < min_y + 0.25

    plt.plot(gas[lim_y], t_total[lim_y], label=f'{thermal:.1f} m/s')
    plt.scatter(min_x, min_y, c='k', marker='x')

plt.gca().set_xlim(0, 65)
plt.grid()
plt.legend()
#plt.show()

# Show different Glide Ratios in Action:
plt.figure()

# Version A: turn and run
#for big_thermal in [1.0, 1.5, 2.0, 2.5, 3.0]:
for big_thermal in [1.0]:
    big_speed = 60.
    if big_thermal <= 1.0:    
        big_speed = 50. # 50km/h@1ms
    big_glide = func(big_speed, *popt)
    t_glide = 10./big_speed*60
    height = 10000/big_glide
    t_height = height/big_thermal/60
    t_total = t_glide + t_height
    plt.plot([0, 45], [t_total, t_total], c='k', label=f'{big_thermal:.2f} m/s')

thermals = np.linspace(0.1, big_thermal, 10)
glides = np.linspace(10, 40, 100)
gas = [40, 47.5][1] # Flugstil B
for thermal in thermals:
        t_glide = 10./gas*60
        height = 10000/glides
        t_height = height/thermal/60
        t_total = t_glide + t_height

        plt.plot(glides, t_total, linewidth=1, label=f'{thermal:.2f} m/s')

plt.gca().set_xlim(0, 45)
#plt.gca().ticklabel_format(style='plain')
plt.grid()
plt.legend()
plt.show()





