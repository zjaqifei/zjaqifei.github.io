#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 10  # length of the beam [m]
q = 5000  # distributed load [N/m]
E = 25e9  # elastic modulus [Pa]
b = 0.2  # width of the cross section [m]
h = 0.4  # height of the cross section [m]
k = 1000  # spring stiffness [N/m]
rho = 2500  # density [kg/m^3]
I = (b * h**3) / 12  # second moment of area [m^4]
a_sq = E * I / (rho * b * h)  # parameter a^2 [(m^2/s^2)]

# Discretization
nx = 300  # number of spatial points
nt = 300  # number of time steps
dx = L / nx  # spatial step [m]
dt = 0.1  # time step [s]
x = np.linspace(0, L, nx)  # spatial grid
w = np.zeros(nx)  # initial deflection
w_new = np.zeros(nx)  # deflection at the next time step

# Time evolution (simplified, for illustration)
for t in range(nt):
    # Update deflection
    for i in range(1, nx-1):
        # Using a simplified finite difference scheme
        w_new[i] = w[i] + dt**2 * (w[i-1] - 2*w[i] + w[i+1]) / dx**2 - dt**2 * q / (a_sq * rho * b * h)
    
    # Update boundary conditions
    w_new[0] = w_new[1]  # free-end
    w_new[-1] = w_new[-2]  # free-end
    
    # Update deflection
    w = np.copy(w_new)
    
    # Plotting every 20 steps
    if t % 20 == 0:
        plt.plot(x, w, label=f't = {t*dt:.1f}s')

# Plotting
plt.title('Simplified Deflection of a Beam (Approximate)')
plt.xlabel('Position along the beam (m)')
plt.ylabel('Deflection (m)')
plt.legend()
plt.grid(True)
plt.show()


# In[11]:


# Parameters
P = 0.01  # load magnitude [N]
v = 0.5  # load velocity [m/s]
x_0 = 0  # initial position of the load [m]
T = 15  # total observation time [s]
dt = 0.1  # time step [s]

# Time and spatial discretization
time = np.arange(0, T+dt, dt)  # time grid
x = np.linspace(0, L, nx)  # spatial grid

# Simple static response computation
def compute_deflection(x, x_P):
    """Compute the static deflection of a simply supported beam under a point load."""
    # Note: This is an exact solution for a simply supported beam under a point load
    #       and it may not be valid if the point load is not within the span of the beam.
    R = P/2  # Reaction force [N]
    M_P = -P * (L - x_P) / 2  # Bending moment at the load [Nm]
    w = np.zeros_like(x)  # Initialize deflection array
    mask = x <= x_P  # Mask array for selecting elements to the left of the load
    w[mask] = -(P * x[mask] * (L - x[mask])**2) / (2 * L)
    w[~mask] = -(P * (L - x[~mask]) * x[~mask]**2) / (2 * L)
    return w

# Compute and plot the deflection
plt.figure(figsize=(10, 6))
for t in time[::int(len(time)/5)]:  # Select some time points for visualization
    x_P = x_0 + v * t  # Update the load position
    if x_P <= L:  # Check if the load is within the span of the beam
        w = compute_deflection(x, x_P)  # Compute the deflection
        plt.plot(x, w, label=f't = {t:.1f}s')

# Plotting
plt.title('Approximate Dynamic Response of a Beam under a Moving Load')
plt.xlabel('Position along the beam (m)')
plt.ylabel('Deflection (m)')
plt.legend()
plt.grid(True)
plt.show()


# In[4]:


# Parameters for the dynamic load
f = 1  # frequency [Hz]
P_0 = 10000  # amplitude of the load [N]
x_P = L/2  # position of the load [m]

# Compute and plot the deflection under a sinusoidal load
plt.figure(figsize=(10, 6))
for t in time[::int(len(time)/5)]:  # Select some time points for visualization
    P = P_0 * np.sin(2 * np.pi * f * t)  # Compute the load magnitude
    w = compute_deflection(x, x_P)  # Compute the deflection
    plt.plot(x, w, label=f't = {t:.1f}s, P = {P:.0f}N')

# Plotting
plt.title('Approximate Response of a Beam under a Sinusoidal Load')
plt.xlabel('Position along the beam (m)')
plt.ylabel('Deflection (m)')
plt.legend()
plt.grid(True)
plt.show()


# In[5]:


# Parameters
L = 10  # length of the beam [m]
P = 10000  # load magnitude [N]
load_positions = [L/4, L/2, 3*L/4]  # positions of the load [m]

# Simple static response computation for a cantilever beam under a point load
def compute_cantilever_deflection(x, x_P):
    """Compute the static deflection of a cantilever beam under a point load."""
    # Note: This is an exact solution for a cantilever beam under a point load
    #       and it may not be valid if the point load is not within the span of the beam.
    w = np.zeros_like(x)  # Initialize deflection array
    mask = x >= x_P  # Mask array for selecting elements to the right of the load
    w[mask] = (P * (x[mask] - x_P)**2 * (3 * L - x[mask])) / (6 * E * I)
    return w

# Compute and plot the deflection
plt.figure(figsize=(10, 6))
for x_P in load_positions:  # Apply the load at different positions
    w = compute_cantilever_deflection(x, x_P)  # Compute the deflection
    plt.plot(x, w, label=f'x_P = {x_P}m')

# Plotting
plt.title('Static Response of a Cantilever Beam under a Point Load')
plt.xlabel('Position along the beam (m)')
plt.ylabel('Deflection (m)')
plt.legend()
plt.grid(True)
plt.show()


# In[6]:


# Simplified non-linear effect parameters
critical_deflection = 0.02  # [m] assumed critical deflection for non-linear effects
additional_stiffness_factor = 10  # assumed additional "stiffness" for non-linear effects

# Compute and plot the deflection under a point load at the midpoint
x_P = L/2  # position of the load [m]
w_linear = compute_deflection(x, x_P)  # Compute the linear deflection

# Simplified non-linear correction
# Note: This is a highly simplified and heuristic correction for illustrative purposes only.
w_nonlinear = w_linear.copy()
mask = np.abs(w_linear) > critical_deflection
w_nonlinear[mask] = w_linear[mask] * (1 + additional_stiffness_factor * 
                                      (np.abs(w_linear[mask]) - critical_deflection))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, w_linear, label='Linear')
plt.plot(x, w_nonlinear, label='Simplified Non-linear')
plt.title('Linear vs Simplified Non-linear Deflection of a Beam under a Point Load')
plt.xlabel('Position along the beam (m)')
plt.ylabel('Deflection (m)')
plt.legend()
plt.grid(True)
plt.show()


# In[7]:


# Parameters
b = 0.2  # width of the beam [m]
h = 0.4  # height of the beam [m]
A = b * h  # cross-sectional area [m^2]
I = (b * h**3) / 12  # second moment of area [m^4]
P_0 = 1000  # load magnitude [N]

# Compute the first natural frequency of a simply supported beam
f_1 = (1.875**2) / (2 * np.pi * L**2) * np.sqrt(E * I / (rho * A))

# Frequencies for the external load
frequencies = [0.5 * f_1, f_1, 2 * f_1]

# Time vector
time = np.linspace(0, 10, 1000)  # [s]

# Compute and plot the deflection under a sinusoidal load
plt.figure(figsize=(10, 6))
for f in frequencies:  # Apply the load at different frequencies
    deflections = []
    for t in time:  # Compute the deflection at each time point
        P = P_0 * np.sin(2 * np.pi * f * t)  # Compute the load magnitude
        w = compute_deflection(x, L/2)  # Compute the deflection
        deflections.append(max(w))  # Store the maximum deflection
    plt.plot(time, deflections, label=f'f = {f:.2f} Hz')

# Plotting
plt.title('Approximate Dynamic Response of a Beam under a Sinusoidal Load')
plt.xlabel('Time (s)')
plt.ylabel('Maximum Deflection (m)')
plt.legend()
plt.grid(True)
plt.show()

# Display the first natural frequency
f_1


# In[8]:


import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 1000  # kg
k = 200000  # N/m
c = 100  # Ns/m
F0 = 1000  # N
omega = 1.0  # rad/s

# Time vector
dt = 0.01  # Time step
t = np.arange(0, 10, dt)  # Total time

# External load
F = F0 * np.sin(omega * t)

# Initial conditions
u = np.zeros_like(t)
u_dot = np.zeros_like(t)
u_ddot = np.zeros_like(t)

# Newmark-beta parameters
beta = 1/4
gamma = 1/2

# Time integration
for i in range(len(t)-1):
    # Predictor
    u[i+1] = u[i] + dt*u_dot[i] + 0.5*dt**2*(1-2*beta)*u_ddot[i]
    u_dot[i+1] = u_dot[i] + (1-gamma)*dt*u_ddot[i]
    
    # Corrector
    u_ddot[i+1] = (F[i+1] - c*u_dot[i+1] - k*u[i+1]) / m
    u[i+1] = u[i] + dt*u_dot[i] + 0.5*dt**2*((1-2*beta)*u_ddot[i] + 2*beta*u_ddot[i+1])
    u_dot[i+1] = u_dot[i] + dt*((1-gamma)*u_ddot[i] + gamma*u_ddot[i+1])

# Plotting
plt.plot(t, u, label='Displacement (m)')
plt.plot(t, F/F0, label='Normalized Force (N)')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m) / Force (N)')
plt.legend()
plt.grid(True)
plt.title('Response of a SDOF system to sinusoidal force')
plt.show()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 1000  # kg
k = 200000  # N/m
c = 100  # Ns/m
F0 = 1000  # N
omega = 1.0  # rad/s

# Time vector
dt = 0.01  # Time step
t = np.arange(0, 10, dt)  # Total time

# External load
F = F0 * np.sin(omega * t)

# Initial conditions
u = np.zeros_like(t)
u_dot = np.zeros_like(t)
u_ddot = np.zeros_like(t)

# Newmark-beta parameters
beta = 1/4
gamma = 1/2

# Time integration
for i in range(len(t)-1):
    # Predictor
    u[i+1] = u[i] + dt*u_dot[i] + 0.5*dt**2*(1-2*beta)*u_ddot[i]
    u_dot[i+1] = u_dot[i] + (1-gamma)*dt*u_ddot[i]
    
    # Corrector
    u_ddot[i+1] = (F[i+1] - c*u_dot[i+1] - k*u[i+1]) / m
    u[i+1] = u[i] + dt*u_dot[i] + 0.5*dt**2*((1-2*beta)*u_ddot[i] + 2*beta*u_ddot[i+1])
    u_dot[i+1] = u_dot[i] + dt*((1-gamma)*u_ddot[i] + gamma*u_ddot[i+1])

# Plotting
plt.plot(t, u, label='Displacement (m)')
plt.plot(t, F/F0, label='Normalized Force (N)')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m) / Force (N)')
plt.legend()
plt.grid(True)
plt.title('Response of a SDOF system to sinusoidal force')
plt.show()


# In[6]:


import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 1000  # kg
k = 200000  # N/m
c = 100  # Ns/m
F0 = 1000  # N
omega = 1.0  # rad/s

# Time vector
dt = 0.01  # Time step
t = np.arange(0, 10, dt)  # Total time

# External load
F_x = F0 * np.sin(omega * t)
F_y = F0 * np.cos(omega * t)

# Initial conditions
u_x = np.zeros_like(t)
u_dot_x = np.zeros_like(t)
u_ddot_x = np.zeros_like(t)

u_y = np.zeros_like(t)
u_dot_y = np.zeros_like(t)
u_ddot_y = np.zeros_like(t)

# Newmark-beta parameters
beta = 1/4
gamma = 1/2

# Time integration
for i in range(len(t)-1):
    # Predictor
    u_x[i+1] = u_x[i] + dt*u_dot_x[i] + 0.5*dt**2*(1-2*beta)*u_ddot_x[i]
    u_dot_x[i+1] = u_dot_x[i] + (1-gamma)*dt*u_ddot_x[i]
    
    u_y[i+1] = u_y[i] + dt*u_dot_y[i] + 0.5*dt**2*(1-2*beta)*u_ddot_y[i]
    u_dot_y[i+1] = u_dot_y[i] + (1-gamma)*dt*u_ddot_y[i]
    
    # Corrector
    u_ddot_x[i+1] = (F_x[i+1] - c*u_dot_x[i+1] - k*u_x[i+1]) / m
    u_x[i+1] = u_x[i] + dt*u_dot_x[i] + 0.5*dt**2*((1-2*beta)*u_ddot_x[i] + 2*beta*u_ddot_x[i+1])
    u_dot_x[i+1] = u_dot_x[i] + dt*((1-gamma)*u_ddot_x[i] + gamma*u_ddot_x[i+1])
    
    u_ddot_y[i+1] = (F_y[i+1] - c*u_dot_y[i+1] - k*u_y[i+1]) / m
    u_y[i+1] = u_y[i] + dt*u_dot_y[i] + 0.5*dt**2*((1-2*beta)*u_ddot_y[i] + 2*beta*u_ddot_y[i+1])
    u_dot_y[i+1] = u_dot_y[i] + dt*((1-gamma)*u_ddot_y[i] + gamma*u_ddot_y[i+1])

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, u_x, label='Displacement x (m)')
plt.plot(t, F_x/F0, label='Normalized Force x (N)')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m) / Force (N)')
plt.legend()
plt.grid(True)
plt.title('Response of a 2D Bridge Model in x direction')

plt.subplot(2, 1, 2)
plt.plot(t, u_y, label='Displacement y (m)')
plt.plot(t, F_y/F0, label='Normalized Force y (N)')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m) / Force (N)')
plt.legend()
plt.grid(True)
plt.title('Response of a 2D Bridge Model in y direction')

plt.tight_layout()
plt.show()


# In[ ]:




