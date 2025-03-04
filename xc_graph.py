import numpy as np
import matplotlib.pyplot as plt

'''
The XC Magic is:

km = avg * time

e.g.

200 = x1 * x2

x1 = 200/x2

'''

target_distance = 200

plt.figure()
plt.title('The XC Formula')

x = np.linspace(0, 11) # the time

plt.plot(x, 100/x, label='100km')
plt.plot(x, 200/x, label='200km')
plt.plot(x, 300/x, label='300km')

plt.xlim(5, 11)
plt.ylim(15, 35)

plt.grid()
plt.legend()

plt.xlabel('time (h)')
plt.ylabel('avg speed (km/h)')

plt.show()