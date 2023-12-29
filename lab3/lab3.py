import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

def SystDiffEq(y, t, m1, m2, r, theta, phi, g, b, l, k):
    # y = [phi, theta, phi', theta'] -> [phi', theta', phi'', theta'']
    dy = np.zeros_like(y)
    dy[0] = y[2]
    dy[1] = y[3]
    a11 = (m1 + m2) * l
    a12 = m1 * b * np.cos(y[1]-y[0])
    b1 = m1 * b * np.sin(y[1]-y[0]) * (y[3]**2) - \
         (m1 + m2) * g * np.sin(y[0])
    a21 = l * np.cos(y[1] - y[0])
    a22 = ((r**2)/(2*b) + b)
    b2 = -1*g*np.sin(y[1]) - ((k * y[3])/(m1 * b)) - \
         l*(y[2]**2)*np.sin(y[1] - y[0])
    detA = a11 * a22 - a12 * a21
    detA1 = b1 * a22 - a12 * b2
    detA2 = a11 * b2 - b1 * a21
    dy[2] = detA1/detA
    dy[3] = detA2/detA
    return dy

m1 = 5
m2 = 5
r = 0.25
l = 1
b = 0.125
k = 10
phi0 = np.pi/2
theta0 = np.pi/2
g = 9.81

step = 10000
t = np.linspace(0, 100, step)

y0 = [phi0, theta0, 0, 0]
Y = odeint(SystDiffEq, y0, t, (m1, m2, r, theta0, phi0, g, b, l, k))

phi = Y[:,0]
theta = Y[:,1]
phit = Y[:,2]
thetat = Y[:,3]

phitt = np.zeros_like(t)
thetatt = np.zeros_like(t)
NA = np.zeros_like(t)
NB = np.zeros_like(t)
N = np.zeros_like(t)

for i in range(len(t)):
    phitt[i] = SystDiffEq(Y[i], t[i], m1, m2, r, theta, phi, g, b, l, k)[2]
    thetatt[i] = SystDiffEq(Y[i], t[i], m1, m2, r, theta, phi, g, b, l, k)[3]
    NA[i] = NB[i] = ((m1 + m2)*(g * np.cos(phi[i]) + l * ((phit[i])**2))) + \
              m1*b*(thetatt[i]*np.sin(theta[i] - phi[i]) + \
              (thetat[i]**2)*np.cos(theta[i] - phi[i])) / 2

# вывод графиков зависимости:
fgrt = plt.figure()

phiplt = fgrt.add_subplot(5,1,1)
phiplt.plot(t, phi, color="red")
phiplt.set_title('Phi(t)')

thetaplt = fgrt.add_subplot(5,1,3)
thetaplt.plot(t, theta, color="orange")
thetaplt.set_title('Theta(t)')

Nplt = fgrt.add_subplot(5,1,5)
Nplt.plot(t, NA, color="blue")
Nplt.set_title('N(t)')

fgrt.show()

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

anim = FuncAnimation(fig, run, interval=1, frames=step) 

plt.show()