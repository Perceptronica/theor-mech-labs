import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

m1 = 5
m2 = 5
r = 0.25
l = 1
b = 0.125
k = 10
phi = np.pi/2
theta = np.pi/2
g = 9.81

step = 200
t = np.linspace(0, 2*np.pi, step)
phi = np.cos(6*t)
theta = np.sin(6*t)

fig, ax = plt.subplots()

ax.set_xlim(-2*l, 3*l)
ax.set_ylim(3*l, -2*l)
ax.set_aspect('equal', 'box')

# поворот на 90 градусов по часовой стрелке:
transform = Affine2D().rotate_deg_around(0, 0, 90) + ax.transData

ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.xaxis.tick_top()

A1x = 0
A1y = 0
B1x = -l
B1y = 0

Ax = l * np.cos(phi)
Ay = -l * np.sin(phi)
Bx = Ax
By = Ay - l

Ox = Bx
Oy = (Ay + By) / 2

Cx = Ox + b * np.cos(theta)
Cy = Oy + (-b * np.sin(theta))

pA1 = ax.plot(A1y, A1x, marker='o', color='black', transform=transform)[0]
pB1 = ax.plot(B1y, B1x, marker='o', color='black', transform=transform)[0]
pA =  ax.plot(Ax[0], Ay[0], marker='o', color='green', transform=transform)[0]
pB =  ax.plot(Bx[0], By[0], marker='o', color='red', transform=transform)[0]
pO =  ax.plot(Ox[0], Oy[0], marker='o', color='black', transform=transform)[0]
pC =  ax.plot(Cx[0], Cy[0], marker='o', color='orange', transform=transform)[0]

AA1 = ax.plot([A1x, Ax[0]], [A1y, Ay[0]], linewidth=3, color='green', transform=transform)[0]
BB1 = ax.plot([0, Bx[0]], [B1y - l, By[0]], linewidth=3, color='red', transform=transform)[0]
AB =  ax.plot([Ax[0], Bx[0]], [Ay[0], By[0]], linewidth=3, color='black', transform=transform)[0]
OC =  ax.plot([Ox[0], Cx[0]], [Oy[0], Cy[0]], color="orange", transform=transform)[0]

circle = plt.Circle((Cx[0], Cy[0]), r, fill=False, transform=transform, linewidth=2)
fig.add_artist(circle)

def run(i):
     pA.set_data(Ax[i], Ay[i])
     pB.set_data(Bx[i], By[i])
     pO.set_data(Ox[i], Oy[i])
     pC.set_data(Cx[i], Cy[i])
     AA1.set_data([A1x, Ax[i]], [A1y, Ay[i]])
     BB1.set_data([0, Bx[i]], [B1y - l, By[i]])
     AB.set_data([Ax[i], Bx[i]], [Ay[i], By[i]])
     OC.set_data([Ox[i], Cx[i]], [Oy[i], Cy[i]])
     circle.center = (Cx[i], Cy[i])
     return

anim = FuncAnimation(fig, run, interval=2, frames=step) 

plt.show()