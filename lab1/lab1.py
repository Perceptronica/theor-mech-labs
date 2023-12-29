import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy import symbols, sin, cos, diff, lambdify

t = symbols('t', real=True)
r = 2 + sin(6*t)
phi = 7*t + 1.2*cos(6*t)

# Символьно вычисляем производные для скорости и ускорения
r_diff = r.diff(t)
phi_diff = phi.diff(t)
r_diff2 = r_diff.diff(t)
phi_diff2 = phi_diff.diff(t)

# Преобразование в полярные координаты
x = r * cos(phi)
y = r * sin(phi)

# Производные для скорости и ускорения в декартовых координатах
vx = r_diff * cos(phi) - r * sin(phi) * phi_diff
vy = r_diff * sin(phi) + r * cos(phi) * phi_diff

ax = r_diff2 * cos(phi) - 2 * r_diff * sin(phi) * phi_diff - r * cos(phi) * phi_diff**2 - r * sin(phi) * phi_diff2
ay = r_diff2 * sin(phi) + 2 * r_diff * cos(phi) * phi_diff - r * sin(phi) * phi_diff**2 + r * cos(phi) * phi_diff2

# Превращаем символьные выражения в функции численного вычисления
x_func = lambdify(t, x)
y_func = lambdify(t, y)
vx_func = lambdify(t, vx)
vy_func = lambdify(t, vy)
ax_func = lambdify(t, ax)
ay_func = lambdify(t, ay)

# Функция для обновления анимации
def update(num, data, lines):
    x, y, vx, vy, ax, ay = data
    time = np.linspace(0, num/10, num=num)
    lines[0].set_data(x[:num], y[:num])
    
    # Стрелка радиус-вектора
    lines[1].set_data([0, x[num-1]], [0, y[num-1]])
    
    # Стрелка вектора скорости
    lines[2].set_data([x[num-1], x[num-1] + vx[num-1]/50], [y[num-1], y[num-1] + vy[num-1]/50])
    
    # Стрелка вектора ускорения
    lines[3].set_data([x[num-1], x[num-1] + ax[num-1]/300], [y[num-1], y[num-1] + ay[num-1]/300])
    
    return lines

# Создаем массив времени с достаточным количеством точек
time_steps = np.linspace(0, 2*np.pi, 300)
x_data = x_func(time_steps)
y_data = y_func(time_steps)
vx_data = vx_func(time_steps)
vy_data = vy_func(time_steps)
ax_data = ax_func(time_steps)
ay_data = ay_func(time_steps)

# Инициализируем фигуру и оси для анимации
fig, ax = plt.subplots()
lines = [ax.plot([], [], 'b')[0],  # Траектория
         ax.plot([], [], 'r', lw=2)[0],  # Радиус-вектор
         ax.plot([], [], 'g', lw=2)[0],  # Вектор скорости
         ax.plot([], [], 'y', lw=2)[0]]  # Вектор ускорения

ax.set_aspect('equal', 'box')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Создаем анимацию
ani = animation.FuncAnimation(fig, update, frames=300, fargs=([x_data, y_data, vx_data, vy_data, ax_data, ay_data], lines), interval=40, blit=True)

# Показываем анимацию
plt.show()
