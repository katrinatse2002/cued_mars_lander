import numpy as np
import matplotlib.pyplot as plt

# mass, spring constant, initial position and velocity
G = 6.674*10e-11
M = 6.42*10e23
m = 100
GMm = G * M * m

h = 0
r0 = 6*10e6
e = 0.7

position = np.array([r0 + h, 0, 0])
velocity = np.array([0, e*np.sqrt(G*M/(r0+h)), 0])

# simulation time, timestep and time
t_max = 50000
dt = 1
t_array = np.arange(0, t_max, dt)

# initialise empty lists to record trajectories
pos_list = []

for i in range(len(t_array)):

    # append current state to trajectories
    pos_list.append(position)

    # calculate new position and velocity
    r = position - np.array([0, 0, 0])
    r_mag = np.linalg.norm(r)
    r_norm = r / r_mag

    F_mag = -GMm / r_mag**2
    F = r_norm * F_mag

    if i == 0:
        pos_prev = -(velocity*dt - position)
    else:
        pos_prev = pos_list[i-1]

    a = F / m
    pos_current = position
    position = 2*position - pos_prev + dt*dt*a
    velocity = (1/dt) * (position - pos_current)

# convert trajectory lists into arrays, so they can be sliced
# useful for Assignment 2
pos_x_array = np.array([pos_list[i][0] for i in range(len(pos_list))])
pos_y_array = np.array([pos_list[i][1] for i in range(len(pos_list))])
pos_z_array = np.array([pos_list[i][2] for i in range(len(pos_list))])

# plot the position-time graph
plt.title('Verlet')
plt.grid()
plt.plot(pos_x_array, pos_y_array, label='Position')
plt.legend()
plt.show()
