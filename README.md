# zjaqifei.github.io
# README: Numerical Simulation Process and the research on Runge phenomenan

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
- - Elastic modulus: \( E = 25 \times 10^9 \) Pa
- Cross-sectional width: \( b = 0.2 \) m
- Cross-sectional height: \( h = 0.4 \) m
- Material density: \( \rho = 2500 \) kg/m³
- Load amplitude: \( P_0 = 1000 \) N
- Load frequency \( f \) varies, including the structure's first natural frequency.

## introduction on the code for the runge phenomenan
## README

### Overview

The provided code aims to address the Runge Phenomenon, which refers to the oscillations near the endpoints of an interval when interpolating with high-degree polynomials. The code showcases different interpolation methods, including polynomial interpolation, least squares polynomial fitting, and Mock-Chebyshev interpolation, to approximate the Runge function.

The Runge function is defined as:
\[ f(x) = \frac{1}{1 + 25x^2} \]

### Dependencies
- `numpy`
- `matplotlib.pyplot`
- `sklearn.preprocessing`
- `sklearn.linear_model`
- `sklearn.pipeline`
- `scipy.optimize`
- `scipy.interpolate`
- `numpy.polynomial`

### Details

1. **Basic Polynomial Interpolation**:
   - The Runge function is visualized using equidistant nodes and polynomial interpolation.
   - The oscillations near the endpoints are observed as the polynomial degree increases.

2. **Visualizing the Runge Phenomenon**:
   - The Runge function and its polynomial interpolations for various numbers of points are plotted to showcase the oscillations when using equidistant nodes.

3. **Polynomial Regression with Regularization**:
   - Ridge (L2), Lasso (L1), and Elastic Net regularization are applied to polynomial regression to address the oscillations. This section compares the effects of different regularization techniques on the polynomial fit.

4. **External Fake Constraints (EFC) Interpolation**:
   - EFC is introduced to adjust the curvature of the polynomial, thereby reducing oscillations due to the Runge Phenomenon. The results of Lagrange interpolation are compared with EFC interpolation.

5. **Optimized EFC Interpolation**:
   - Optimization techniques are applied to find the best EFC positions to minimize the error between the EFC polynomial and the true Runge function.

6. **Mock-Chebyshev Subset Interpolation**:
   - Mock-Chebyshev nodes are introduced as an improvement over equidistant nodes. The subset of nodes selected from equidistant nodes closely resembles the Chebyshev nodes.
   - The code demonstrates the difference between interpolations using equidistant nodes, Chebyshev-Lobatto nodes, and mock-Chebyshev nodes.

7. **Improved Mock-Chebyshev Interpolation**:
   - An advanced method for selecting mock-Chebyshev nodes is introduced to provide a better approximation of the Runge function.

8. **Constrained Mock-Chebyshev Least Squares Approximation**:
   - A constrained least squares approximation is used to fit the polynomial, ensuring that the polynomial passes through selected mock-Chebyshev nodes.

### Usage

To visualize and understand the effects of different interpolation methods on the Runge function, simply execute the provided code. Ensure that the required libraries are installed.

### Conclusion

The provided code offers various methods to address the Runge Phenomenon, showcasing the challenges and solutions related to polynomial interpolation. Different interpolation and approximation techniques are explored to provide a comprehensive understanding of the problem and its solutions.


## introduction on the latter part
Sure! Here's a more detailed breakdown of the provided code:

1. **Tikhonov Regularized Interpolation**:
    - This segment deals with a polynomial interpolation of the Runge function. The Runge function is known to cause oscillations near the boundaries when interpolated with high-degree polynomials on equidistant nodes. This is often referred to as Runge's phenomenon.
    - To mitigate this phenomenon, Tikhonov regularization is introduced. Regularization adds a penalty to the polynomial's coefficients, favoring smoother curves and helping to stabilize the interpolation.
    - The method involves creating a Vandermonde matrix, a difference matrix, and then using a least squares solver (`lstsq`) to compute the polynomial coefficients that best fit the data with the regularization constraint.

2. **Intelligent Genetic Algorithm (IGA) on Runge Function**:
    - Here, a standard Genetic Algorithm (GA) approach is used with an adaptive mechanism inspired by the behavior of intelligent creatures.
    - The GA involves basic operations like initialization, selection (choosing parents based on fitness), crossover (combining genes of parents), and mutation (randomly altering genes) to evolve a population of solutions.
    - The fitness function in this context measures how well a polynomial, defined by its coefficients (or genes), approximates the Runge function.

3. **Filtered Trigonometric Interpolant**:
    - This method provides a way to approximate the Runge function by filtering its trigonometric interpolant.
    - The approach involves dividing the interval into three sections: left, middle, and right. The middle section is approximated using a filtered trigonometric interpolant, while the left and right sections use polynomial interpolation.
    - This combined approach helps provide a good approximation over the entire interval.

4. **Hybrid Particle Swarm Optimization (PSO) and Genetic Algorithm (GA)**:
    - This section introduces a hybrid approach that combines the PSO and GA algorithms.
    - PSO is inspired by the social behavior of birds flocking or fish schooling. It involves particles "flying" through the solution space and being influenced by their personal best positions and the global best position.
    - After the PSO step, the GA operations (crossover and mutation) are applied to further refine the solutions.
    - This hybrid approach aims to combine the exploration capabilities of PSO with the exploitation capabilities of GA.

5. **Intelligent Genetic Algorithm (IGA) to Find Best Interpolation Node**:
    - This approach uses a GA to determine the best interpolation nodes for the Runge function.
    - The method involves encoding a potential solution (interpolation node) as a binary chromosome, which is then decoded to obtain the actual node position.
    - The fitness function measures the difference between the Runge function value and its interpolation value at a particular node.

6. **Another Genetic Algorithm Approach for the Runge Function**:
    - This segment represents a different approach to using a GA for the Runge function.
    - The method also involves encoding and decoding, similar to the previous approach, but the operations and implementations differ slightly.

7. **SVD Truncated Interpolation on Uniform Grid**:
    - This method uses Singular Value Decomposition (SVD) to obtain a stable polynomial interpolation of the Runge function.
    - By truncating (or ignoring) small singular values, the method avoids the ill-conditioning often associated with high-degree polynomial interpolation on uniform grids.

8. **SVD Truncated Interpolation using Chebyshev Nodes**:
    - Similar to the previous method, this segment uses SVD for interpolation, but it employs Chebyshev nodes instead of uniform nodes.
    - Chebyshev nodes are specific points in the interval that help mitigate the oscillations associated with Runge's phenomenon. When combined with SVD truncation, this method provides a robust and accurate approximation of the Runge function.

The overall code aims to demonstrate various approaches to approximating the Runge function, each with its advantages and challenges. The visualizations provided in the code help in understanding the behavior and performance of each method.

