# zjaqifei.github.io
# README: Numerical Simulation Process

## Introduction

This document discusses the deflection behavior of bridges under various load and boundary conditions using numerical simulations. Here's a detailed description of each experiment.

---

## Experiment 1: Basic Simulation
### Parameters:
- Beam length: \( L = 10 \) m
- Distributed load: \( q = 5000 \) N/m
- Elastic modulus: \( E = 25 \times 10^9 \) Pa
- Cross-sectional width: \( b = 0.2 \) m
- Cross-sectional height: \( h = 0.4 \) m
- Elastic bearing stiffness: \( k = 1000 \) N/m
- Density: \( \rho = 2500 \) kg/m³
- Observation range: \( 0 \leq x \leq L \)
- Initial conditions: stationary beam i.e., \( w(x,0) = 0 \) and given velocity condition.

### Steps:
1. Spatial and time discretization
2. Initialization
3. Iteration using finite difference method
4. Observation and analysis

---

## Experiment 2: Moving Point Load
To simulate the dynamic load caused by vehicles on bridges, we consider a moving point load on the beam.

### Steps:
1. Define the load with characteristics such as load magnitude \( P \), velocity \( v \), and initial position \( x_0 \).
2. Update the load position at each time step with \( x_P(t) = x_0 + v \cdot t \).
3. Solve for the response.
4. Analyze the results, especially focusing on key performance metrics like maximum deflection and deflection rate.

### Experiment 2.1: Single Moving Point Load

#### Assumptions and Parameters:
- Load magnitude: \( P = 10000 \) N
- Load velocity: \( v = 1 \) m/s
- Initial load position: \( x_0 = 0 \) m
- Beam length: \( L = 10 \) m
- Distributed load: \( q = 0 \) (only dynamic load considered)
- Observation time: \( T = 15 \) s

### Experiment 2.2: Periodically Varying Load
Explore the beam's response under a sinusoidally varying dynamic load.

#### Parameters and Assumptions:
- Load frequency: \( f = 1 \) Hz
- Load amplitude: \( P_0 = 10000 \) N
- Load position: \( x_P = L/2 \) m (placed at beam's midpoint)
- Observation time: \( T = 10 \) s
- Initial condition: stationary beam

---

## Experiment 3: Changing Boundary Conditions
Study the response of a cantilever beam (one end fixed, the other free) under a point load.

#### Parameters:
- Load magnitude: \( P = 10000 \) N
- Load position: \( x_P = L/2 \) m (placed at beam's midpoint)
- Beam length: \( L = 10 \) m

---

## Experiment 4: Nonlinear Cases

### Nonlinear Elastic Response
Consider the nonlinear elastic behavior of the material using models like the Ramberg-Osgood model.

#### Parameters and Assumptions:
- Beam length: \( L = 10 \) m
- Load magnitude: \( P = 10000 \) N
- Load position: \( x_P = L/2 \) m (placed at beam's midpoint)

### Nonlinear Elastic Model:
The Ramberg-Osgood model is given by:
\[ \sigma = E \epsilon + \alpha E (\epsilon^n) \]
Where:
- \( \sigma \) is the stress
- \( \epsilon \) is the strain
- \( E \) is the initial elastic modulus
- \( \alpha \) and \( n \) are material parameters

---

## Experiment 5: Resonance Experiments

### Experiment 5.1
For a simple beam, the first natural frequency can be approximated using a theoretical expression.

#### Parameters:
- Beam length: \( L = 10 \) m
- Elastic modulus: \( E = 25 \times 10^9 \) Pa
- Cross-sectional width: \( b = 0.2 \) m
- Cross-sectional height: \( h = 0.4 \) m
- Material density: \( \rho = 2500 \) kg/m³
- Load amplitude: \( P_0 = 1000 \) N
- Load frequency \( f \) varies, including the structure's first natural frequency.
