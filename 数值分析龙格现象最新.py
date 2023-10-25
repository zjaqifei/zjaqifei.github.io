#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Define the Runge function
def runge(x):
    return 1.0 / (1 + 25 * x**2)

# Generate data points
x = np.linspace(-1, 1, 1000)
y = runge(x)

# Equidistant nodes for polynomial interpolation
nodes = np.linspace(-1, 1, 11)
values = runge(nodes)

# Polynomial interpolation
p = np.polyfit(nodes, values, len(nodes)-1)
y_poly = np.polyval(p, x)

# Plot the Runge function and its polynomial interpolation
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Runge function', color='blue')
plt.plot(x, y_poly, label='Polynomial interpolation', color='red', linestyle='--')
plt.scatter(nodes, values, color='green', marker='o', label='Equidistant nodes')
plt.legend()
plt.title('Runge Phenomenon using Polynomial Interpolation')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()


# In[10]:


import numpy as np
import matplotlib.pyplot as plt

# Define the Runge function
def runge_function(x):
    return 1.0 / (1 + 25 * x**2)

# Create a dense set of x values for plotting the true function
x_dense = np.linspace(-1, 1, 1000)
y_dense = runge_function(x_dense)

# Plot the Runge function and its polynomial interpolations for various numbers of points
plt.figure(figsize=(12, 8))
plt.plot(x_dense, y_dense, label='Runge Function', color='black', linewidth=2)

for num_points in [5, 10, 15, 20]:
    x_points = np.linspace(-1, 1, num_points)
    y_points = runge_function(x_points)
    
    # Get polynomial interpolation
    coeffs = np.polyfit(x_points, y_points, num_points-1)
    y_poly = np.polyval(coeffs, x_dense)
    
    plt.plot(x_dense, y_poly, label=f'Interpolation with {num_points} points')

plt.ylim([-1, 1.5])
plt.title('Runge Phenomenon Visualization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()


# In[6]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso, ElasticNet

# Define a function to fit and predict using polynomial regression
def polynomial_regression(x_train, y_train, x_test, degree, alpha=0):
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
    model.fit(x_train, y_train)
    return model.predict(x_test)
# Define the Runge function and generate data
x_sample = np.linspace(-1, 1, 11).reshape(-1, 1)
y_sample = runge(x_sample)



# Reshape the data for prediction
x_reshape = x.reshape(-1, 1)

# Without regularization
y_pred_no_reg = polynomial_regression(x_sample, y_sample, x_reshape, 10)

# With L2 regularization (Ridge regression)
y_pred_with_reg = polynomial_regression(x_sample, y_sample, x_reshape, 10, alpha=0.01)








# Define function to fit and predict using different regularizations
def polynomial_regression_reg(x_train, y_train, x_test, degree, reg_type="ridge", alpha=0.01):
    if reg_type == "ridge":
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
    elif reg_type == "lasso":
        model = make_pipeline(PolynomialFeatures(degree), Lasso(alpha=alpha, max_iter=10000))
    elif reg_type == "elasticnet":
        model = make_pipeline(PolynomialFeatures(degree), ElasticNet(alpha=alpha, max_iter=10000))
    else:
        raise ValueError("Invalid regularization type")
    
    model.fit(x_train, y_train)
    return model.predict(x_test)

# With L1 regularization (Lasso regression)
y_pred_lasso = polynomial_regression_reg(x_sample, y_sample, x_reshape, 10, reg_type="lasso", alpha=0.01)

# With Elastic Net regularization
y_pred_elasticnet = polynomial_regression_reg(x_sample, y_sample, x_reshape, 10, reg_type="elasticnet", alpha=0.01)

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(x, y, label='Runge function', color='blue')
plt.scatter(x_sample, y_sample, color='green', marker='o', label='Sampled points')
plt.plot(x, y_pred_no_reg, label='No regularization', color='red', linestyle='--')
plt.plot(x, y_pred_with_reg, label='Ridge (L2)', color='orange', linestyle='--')
plt.plot(x, y_pred_lasso, label='Lasso (L1)', color='purple', linestyle='-.')
plt.plot(x, y_pred_elasticnet, label='Elastic Net', color='cyan', linestyle=':')
plt.legend()
plt.title('Effect of Different Regularizations on Runge Phenomenon')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()


# In[7]:


#基于虚假性约束的龙格现象的研究
#定义龙格函数 

#生成等间距节点。
#基于等间距节点，计算Lagrange插值多项式。
#添加外部伪约束（EFC）来调整多项式的曲率，从而减少龙格现象带来的振荡。
#将Lagrange插值多项式与添加了EFC的多项式进行对比。
import numpy as np
import matplotlib.pyplot as plt

# Define the Runge function
def runge(x):
    return 1.0 / (1 + 25 * x**2)

# Lagrange interpolation
def lagrange_interpolation(x, y, x_val):
    n = len(x)
    result = 0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (x_val - x[j]) / (x[i] - x[j])
        result += term
    return result

# External Fake Constraints (EFC) Interpolation
# For simplicity, we will add EFC to the polynomial by adding points with second derivative 0.
# This is a basic method and does not guarantee the best results.
def efc_interpolation(x, y, x_val, efc_points):
    x_efc = np.append(x, efc_points)
    y_efc = np.append(y, [0] * len(efc_points))
    return lagrange_interpolation(x_efc, y_efc, x_val)

# Create equidistant nodes
x_nodes = np.linspace(-1, 1, 5)
y_nodes = runge(x_nodes)

# Evaluate the functions
x_vals = np.linspace(-1.3, 1.3, 400)
y_actual = runge(x_vals)
y_lagrange = [lagrange_interpolation(x_nodes, y_nodes, x_val) for x_val in x_vals]
y_efc = [efc_interpolation(x_nodes, y_nodes, x_val, [-1.1, 1.1]) for x_val in x_vals]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_actual, label="Runge Function", color="black")
plt.plot(x_vals, y_lagrange, label="Lagrange Interpolation", linestyle="--", color="blue")
plt.plot(x_vals, y_efc, label="EFC Interpolation", linestyle="-.", color="red")
plt.legend()
plt.title("Comparison of Runge Function, Lagrange Interpolation, and EFC Interpolation")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()


# In[1]:


import numpy as np
from scipy.optimize import minimize

# Define Runge function
def runge(x):
    return 1.0 / (1 + 25 * x**2)

# Define the polynomial function based on coefficients
def polynomial(x, coeffs):
    return sum([c * x**i for i, c in enumerate(coeffs)])

# Define the objective function for optimization
def objective_function(coeffs, x_vals, y_vals, efc_positions):
    # Calculate the polynomial values at the given x positions
    poly_vals = np.array([polynomial(x, coeffs) for x in x_vals])
    
    # First term: Match the function values
    match_error = np.sum((poly_vals - y_vals)**2)
    
    # Second term: EFC constraints
    efc_error = np.sum([(polynomial(x, coeffs) - runge(x))**2 for x in efc_positions])
    
    return match_error + efc_error

# Generate equispaced nodes
num_nodes = 5
x_nodes = np.linspace(-1, 1, num_nodes)
y_nodes = runge(x_nodes)

# Optimization
best_error = float('inf')
best_efc_positions = None
best_coeffs = None

# Search for EFC positions
for num_efc in range(2, 11):
    efc_positions_left = np.linspace(-1.1, -2, num_efc//2)
    efc_positions_right = np.linspace(1.1, 2, num_efc//2)
    efc_positions = np.concatenate((efc_positions_left, efc_positions_right))
    
    # Solve the optimization problem
    initial_guess = [0] * (num_nodes + num_efc)
    result = minimize(objective_function, initial_guess, args=(x_nodes, y_nodes, efc_positions))
    
    if result.fun < best_error:
        best_error = result.fun
        best_efc_positions = efc_positions
        best_coeffs = result.x

best_error, best_efc_positions, best_coeffs



import matplotlib.pyplot as plt

# Generate a dense set of x values for plotting
x_dense = np.linspace(-1.2, 1.2, 400)
y_runge = runge(x_dense)
y_poly = [polynomial(x, best_coeffs) for x in x_dense]

plt.figure(figsize=(10,6))
plt.plot(x_dense, y_runge, label='Runge Function', color='blue')
plt.plot(x_dense, y_poly, label='EFC Polynomial', linestyle='dashed', color='red')
plt.scatter(x_nodes, y_nodes, color='green', label='Equispaced Nodes')
plt.scatter(best_efc_positions, runge(best_efc_positions), color='black', marker='x', label='EFC Positions')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Runge Function vs EFC Polynomial')
plt.legend()
plt.grid(True)
plt.show()


# In[13]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from scipy.interpolate import CubicSpline, lagrange
from numpy.polynomial.chebyshev import chebinterpolate

# Define the Runge function
def runge(x):
    return 1.0 / (1 + 25 * x**2)

# Define equispaced nodes and Chebyshev nodes
num_points = 11
x_equispaced = np.linspace(-1, 1, num_points)
x_chebyshev = np.cos((2 * np.arange(1, num_points + 1) - 1) * np.pi / (2 * num_points))
y_equispaced = runge(x_equispaced)
y_chebyshev = runge(x_chebyshev)

# Define a dense grid for plotting
x_dense = np.linspace(-1.1, 1.1, 400)
y_true = runge(x_dense)

# L1, L2, and Elastic Net Interpolation
l1 = Lasso(alpha=0.01).fit(x_equispaced[:, np.newaxis], y_equispaced)
l2 = Ridge(alpha=0.01).fit(x_equispaced[:, np.newaxis], y_equispaced)
elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.5).fit(x_equispaced[:, np.newaxis], y_equispaced)

y_l1 = l1.predict(x_dense[:, np.newaxis])
y_l2 = l2.predict(x_dense[:, np.newaxis])
y_elastic_net = elastic_net.predict(x_dense[:, np.newaxis])

# Chebyshev Interpolation
coeff_chebyshev = chebinterpolate(runge, num_points - 1)
y_chebyshev_interp = np.polynomial.chebyshev.chebval(x_dense, coeff_chebyshev)

# Spline Interpolation
spline = CubicSpline(x_equispaced, y_equispaced)
y_spline = spline(x_dense)

# Plotting the results
plt.figure(figsize=(14, 8))

plt.plot(x_dense, y_chebyshev_interp, label='Chebyshev Interpolation', linestyle='-')
plt.plot(x_dense, y_spline, label='Spline Interpolation', linestyle='--', color='black')
plt.scatter(x_equispaced, y_equispaced, color='red', s=50, zorder=5, label='Equispaced Nodes')
plt.scatter(x_chebyshev, y_chebyshev, color='green', s=50, zorder=5, label='Chebyshev Nodes')
plt.plot(x, y, label='Runge function', color='blue')
plt.plot(x, y_poly, label='Polynomial interpolation', color='red', linestyle='--')
#
# Generate data points
x = np.linspace(-1, 1, 1000)
y = runge(x)

# Equidistant nodes for polynomial interpolation
nodes = np.linspace(-1, 1, 11)
values = runge(nodes)

# Polynomial interpolation
p = np.polyfit(nodes, values, len(nodes)-1)
y_poly = np.polyval(p, x)


##
plt.legend()
plt.title('Comparison of Interpolation Methods with Runge Function')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Define the Runge function
def runge_function(x):
    return 1.0 / (1 + 25 * x**2)

# Define the Lagrange interpolating polynomial
def lagrange_interpolation(x, x_values, y_values):
    n = len(x_values)
    result = 0.0
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if i != j:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += term
    return result

def three_interval_method(x, x_values, y_values, epsilon):
    n = len(x_values)
    if x < -1 + epsilon:
        # Linear interpolation on the left interval
        return y_values[0] + (y_values[1] - y_values[0]) / (x_values[1] - x_values[0]) * (x - x_values[0])
    elif x > 1 - epsilon:
        # Linear interpolation on the right interval
        return y_values[-2] + (y_values[-1] - y_values[-2]) / (x_values[-1] - x_values[-2]) * (x - x_values[-2])
    else:
        # Lagrange interpolation on the central interval
        return lagrange_interpolation(x, x_values, y_values)

# Generate equidistant nodes and their function values
num_points = 11
x_values = np.linspace(-1, 1, num_points)
y_values = runge_function(x_values)

# Generate a dense set of x values for plotting
x_dense = np.linspace(-1.05, 1.05, 400)
y_true = runge_function(x_dense)
y_approx = [three_interval_method(x, x_values, y_values, 0.3) for x in x_dense]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_true, label="True Runge function", color="blue")
plt.plot(x_dense, y_approx, label="Three-interval method", color="red", linestyle="--")
plt.scatter(x_values, y_values, color="green", marker="o", label="Interpolation nodes")
plt.legend()
plt.title("Runge Function and Three-interval Interpolation")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()


# In[3]:


# Use Chebyshev nodes for the central interval
def chebyshev_nodes(n):
    return np.cos(np.pi * (2 * np.arange(1, n + 1) - 1) / (2 * n))

# Define the interpolation using the three-interval method
def improved_three_interval_method(x, x_values, y_values, epsilon):
    n = len(x_values)
    if x < -1 + epsilon:
        # Quadratic interpolation on the left interval
        a0, a1, a2 = np.polyfit(x_values[:3], y_values[:3], 2)
        return a0 * x**2 + a1 * x + a2
    elif x > 1 - epsilon:
        # Quadratic interpolation on the right interval
        a0, a1, a2 = np.polyfit(x_values[-3:], y_values[-3:], 2)
        return a0 * x**2 + a1 * x + a2
    else:
        # Lagrange interpolation on the central interval using Chebyshev nodes
        central_x_values = chebyshev_nodes(n - 2)  # Excluding the two end points
        central_y_values = runge_function(central_x_values)
        return lagrange_interpolation(x, central_x_values, central_y_values)

# Generate a dense set of x values for plotting
y_approx_improved = [improved_three_interval_method(x, x_values, y_values, 0.3) for x in x_dense]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_true, label="True Runge function", color="blue")
plt.plot(x_dense, y_approx_improved, label="Improved Three-interval method", color="red", linestyle="--")
plt.scatter(x_values, y_values, color="green", marker="o", label="Interpolation nodes")
plt.legend()
plt.title("Runge Function and Improved Three-interval Interpolation")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()


# In[4]:


from numpy.polynomial import Polynomial

# Least-squares polynomial fitting
def least_squares_fit(x_values, y_values, degree):
    coeffs = np.polyfit(x_values, y_values, degree)
    return Polynomial(coeffs[::-1])

# Mock-Chebyshev subset interpolation
def mock_chebyshev_subset(N, P):
    full_grid = np.linspace(-1, 1, P)
    chebyshev_lobatto = np.cos(np.pi * np.linspace(0, 1, N))
    # Find the nearest points in the full grid to the Chebyshev–Lobatto nodes
    indices = np.array([np.argmin(np.abs(full_grid - point)) for point in chebyshev_lobatto])
    return full_grid[indices]

# Define the Runge function again for clarity
def runge_function(x):
    return 1.0 / (1 + 25 * x**2)

# Generate data points
P = 20
x_values = np.linspace(-1, 1, P)
y_values = runge_function(x_values)

# Perform least-squares polynomial fitting
degree = 10
lsq_poly = least_squares_fit(x_values, y_values, degree)

# Mock-Chebyshev subset interpolation
subset_points = mock_chebyshev_subset(11, P)
y_subset = runge_function(subset_points)
mock_cheb_poly = Polynomial.fit(subset_points, y_subset, degree)

# Plot the results
plt.figure(figsize=(12, 7))
x_dense = np.linspace(-1.1, 1.1, 400)
plt.plot(x_dense, runge_function(x_dense), label="True Runge function", color="blue")
plt.plot(x_dense, lsq_poly(x_dense), label="Least-squares fit", linestyle="--", color="red")
plt.plot(x_dense, mock_cheb_poly(x_dense), label="Mock-Chebyshev interpolation", linestyle="-.", color="green")
plt.scatter(x_values, y_values, color="black", marker="o", label="Sample points")
plt.legend()
plt.title("Runge Function, Least-Squares Fit, and Mock-Chebyshev Interpolation")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()


# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Define the Runge function
def runge_function(x):
    return 1.0 / (1 + 25 * x**2)

# Polynomial interpolation function
def interpolate_polynomial(x, y, x_new):
    return np.polyval(np.polyfit(x, y, len(x)-1), x_new)

# Create the evenly spaced grid and the Chebyshev-Lobatto grid
x_evenly_spaced = np.linspace(-1, 1, 11)
y_evenly_spaced = runge_function(x_evenly_spaced)

x_chebyshev = np.cos(np.linspace(0, np.pi, 11))
y_chebyshev = runge_function(x_chebyshev)

# Mock-Chebyshev subset: simply select every second point from the evenly spaced grid as a mock
x_mock_chebyshev = x_evenly_spaced[::2]
y_mock_chebyshev = y_evenly_spaced[::2]

# Interpolate the functions on a dense grid
x_dense = np.linspace(-1.1, 1.1, 400)
y_true = runge_function(x_dense)

y_interp_evenly_spaced = interpolate_polynomial(x_evenly_spaced, y_evenly_spaced, x_dense)
y_interp_chebyshev = interpolate_polynomial(x_chebyshev, y_chebyshev, x_dense)
y_interp_mock_chebyshev = interpolate_polynomial(x_mock_chebyshev, y_mock_chebyshev, x_dense)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_true, label="True Runge function", color="black")
plt.plot(x_dense, y_interp_evenly_spaced, label="Interpolation on evenly spaced grid", linestyle="--")
plt.plot(x_dense, y_interp_chebyshev, label="Interpolation on Chebyshev-Lobatto grid", linestyle=":")
plt.plot(x_dense, y_interp_mock_chebyshev, label="Interpolation on mock-Chebyshev grid", linestyle="-.")
plt.legend()
plt.title("Comparison of Interpolation Methods to Address Runge Phenomenon")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt

# 定义 Runge 函数
def runge_function(x):
    return 1.0 / (1.0 + 25 * x**2)

# Improved Mock-Chebyshev Interpolation
def improved_mock_chebyshev_nodes(M, N_subset):
    """生成改进的Mock-Chebyshev节点"""
    chebyshev_lobatto = np.cos(np.pi * np.arange(N_subset) / (N_subset - 1))
    equispaced = np.linspace(-1, 1, M)
    selected_indices = [np.argmin(np.abs(equispaced - node)) for node in chebyshev_lobatto]
    return np.array(equispaced)[selected_indices]

# 使用改进方法选择 mock-Chebyshev 节点
num_nodes = 11
degree = 10
x = np.linspace(-1, 1, num_nodes)
x_subset_improved = improved_mock_chebyshev_nodes(len(x), degree)
y_subset_improved = runge_function(x_subset_improved)

# 使用选定的子集进行插值
p_subset_improved = np.polyfit(x_subset_improved, y_subset_improved, degree - 1)

# 在密集网格上评估多项式
x_dense = np.linspace(-1, 1, 400)
y_true_dense = runge_function(x_dense)
y_approx_improved = np.polyval(p_subset_improved, x_dense)

# 为 Improved Mock-Chebyshev Interpolation 作图
plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_true_dense, label="True Runge function", color="black")
plt.plot(x_dense, y_approx_improved, label=f"Improved Mock-Chebyshev (degree {degree})", linestyle="--")
plt.scatter(x, runge_function(x), color='red', marker='x', label="Equispaced nodes")
plt.legend()
plt.title("Improved Mock-Chebyshev Interpolation")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

# Constrained Mock-Chebyshev Least Squares Approximation
from scipy.linalg import lstsq

def constrained_least_squares(x, y, x_subset, y_subset, degree):
    A = np.vander(x, degree + 1)
    A_subset = np.vander(x_subset, degree + 1)
    # Create the constraints matrix
    constraints = np.zeros((A_subset.shape[0], A.shape[1]))
    constraints[:A_subset.shape[0], :A_subset.shape[1]] = A_subset
    # Create the combined matrix and target vector
    combined_matrix = np.vstack([A, constraints])
    combined_target = np.hstack([y, y_subset])
    # Solve the constrained least squares problem
    coeffs, _, _, _ = lstsq(combined_matrix, combined_target)
    return coeffs

# 获取约束多项式的系数
coeffs_constrained = constrained_least_squares(x, runge_function(x), x_subset_improved, y_subset_improved, degree)

# 评估约束多项式
y_approx_constrained = np.polyval(coeffs_constrained, x_dense)

# 为 Constrained Mock-Chebyshev Least Squares Approximation 作图
plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_true_dense, label="True Runge function", color="black")
plt.plot(x_dense, y_approx_constrained, label=f"Constrained Mock-Chebyshev Least Squares (degree {degree})", linestyle="--")
plt.scatter(x, runge_function(x), color='red', marker='x', label="Equispaced nodes")
plt.legend()
plt.title("Constrained Mock-Chebyshev Least Squares Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

# 定义 Runge 函数
def runge_function(x):
    return 1.0 / (1.0 + 25 * x**2)

# Tikhonov 正则化插值
def tikhonov_interpolation(x, y, degree, lambda_):
    # 创建 Vandermonde 矩阵
    A = np.vander(x, degree + 1)
    
    # 创建差分矩阵 L
    L = np.diff(np.eye(degree + 1), 2, axis=0)
    
    # 使用 lstsq 解决正则化问题
    coeffs, _, _, _ = lstsq(np.vstack([A, lambda_ * L]), np.hstack([y, np.zeros(L.shape[0])]))
    
    return coeffs

# 生成数据点
num_points = 11
x = np.linspace(-1, 1, num_points)
y = runge_function(x)

# 使用 Tikhonov 正则化进行插值
degree = 10
lambda_ = 0.01
coeffs = tikhonov_interpolation(x, y, degree, lambda_)

# 在密集网格上评估多项式
x_dense = np.linspace(-1, 1, 400)
y_true_dense = runge_function(x_dense)
y_approx = np.polyval(coeffs, x_dense)

# 作图
plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_true_dense, label="True Runge function", color="black")
plt.plot(x_dense, y_approx, label=f"Tikhonov Regularized Interpolation (degree {degree}, lambda {lambda_})", linestyle="--")
plt.scatter(x, y, color='red', marker='x', label="Data points")
plt.legend()
plt.title("Tikhonov Regularized Interpolation for Runge Function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()


# In[8]:


import numpy as np

# 定义Runge函数
def runge_function(x):
    return 1.0 / (1.0 + 25 * x**2)

# 初始化种群
def initialize_population(population_size, solution_size):
    return np.random.rand(population_size, solution_size)

# 适应度评估
def fitness(solution):
    # 使用solution参数评估Runge函数并返回其适应度
    # 适应度可以是Runge函数与插值之间的差异
    # 这只是一个示例，具体细节可能需要更多的工作
    interpolated_value = np.polyval(solution, np.linspace(-1, 1, len(solution)))
    true_value = runge_function(np.linspace(-1, 1, len(solution)))
    return np.linalg.norm(interpolated_value - true_value)

# 选择父代
def select_parents(population, fitnesses):
    parents_idx = np.argsort(fitnesses)[:2]
    return population[parents_idx]

# 交叉
def crossover(parent1, parent2):
    crossover_point = np.random.randint(len(parent1))
    child = np.hstack((parent1[:crossover_point], parent2[crossover_point:]))
    return child

# 变异
def mutate(child):
    mutation_point = np.random.randint(len(child))
    child[mutation_point] += np.random.randn()
    return child

# IGA算法主体
def IGA(population_size=200, generations=300, solution_size=10):
    population = initialize_population(population_size, solution_size)
    
    for generation in range(generations):
        fitnesses = [fitness(solution) for solution in population]
        
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = select_parents(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = np.array(new_population)
    
    best_solution = population[np.argmin(fitnesses)]
    return best_solution

best_solution = IGA()


# In[9]:


import numpy as np
import matplotlib.pyplot as plt

# 定义Runge函数
def runge_function(x):
    return 1.0 / (1.0 + 25 * x**2)

# 滤波的三角形插值
def filtered_trigonometric_interpolant(x, data, a, b):
    N = len(data) - 1
    k = np.arange(-N/2, N/2 + 1)
    c = np.fft.fftshift(np.fft.fft(data) / N)
    filter_function = np.exp(-np.abs(k/a)**b)
    filtered_c = c * filter_function
    f_hat = np.fft.ifft(np.fft.ifftshift(filtered_c) * N)
    return np.interp(x, np.linspace(-1, 1, N+1), f_hat.real)

# 主函数
def interpolate_runge(N):
    x = np.linspace(-1, 1, N+1)
    y = runge_function(x)
    
    # 子区间定义
    boundary_layer_width = N**-0.5
    x_middle = np.linspace(-1 + boundary_layer_width, 1 - boundary_layer_width, int(N - 2*boundary_layer_width*N))
    y_middle = filtered_trigonometric_interpolant(x_middle, y, a=2.0, b=2.0)
    
    x_left = np.linspace(-1, -1 + boundary_layer_width, int(boundary_layer_width*N))
    x_right = np.linspace(1 - boundary_layer_width, 1, int(boundary_layer_width*N))
    p_left = np.polyfit(x_left, runge_function(x_left), len(x_left)-1)
    p_right = np.polyfit(x_right, runge_function(x_right), len(x_right)-1)
    y_left = np.polyval(p_left, x_left)
    y_right = np.polyval(p_right, x_right)
    
    x_combined = np.concatenate([x_left, x_middle, x_right])
    y_combined = np.concatenate([y_left, y_middle, y_right])
    return x_combined, y_combined

N = 100
x, y_approx = interpolate_runge(N)
y_true = runge_function(x)
plt.plot(x, y_true, label='True Runge Function')
plt.plot(x, y_approx, label='Approximation', linestyle='--')
plt.legend()
plt.show()


# In[12]:


import numpy as np

# 目标函数
def objective_function(x):
    return 1.0 / (1 + 25 * x**2)

# 粒子群优化部分
class Particle:
    def __init__(self, dim):
        self.position = np.random.uniform(-1, 1, dim)
        self.velocity = np.random.uniform(-0.1, 0.1, dim)
        self.best_position = np.copy(self.position)
        self.best_score = objective_function(self.position)

def pso_update(particle, g_best_position, w=0.5, c1=1.5, c2=1.5):
    inertia = w * particle.velocity
    personal_attraction = c1 * np.random.random() * (particle.best_position - particle.position)
    global_attraction = c2 * np.random.random() * (g_best_position - particle.position)
    
    particle.velocity = inertia + personal_attraction + global_attraction
    particle.position += particle.velocity
    
    score = objective_function(particle.position)
    if score < particle.best_score:
        particle.best_score = score
        particle.best_position = particle.position

# 遗传算法部分
def crossover(parent1, parent2):
    idx = np.random.randint(0, len(parent1))
    child1 = np.concatenate((parent1[:idx], parent2[idx:]))
    child2 = np.concatenate((parent2[:idx], parent1[idx:]))
    return child1, child2

def mutate(child, mutation_rate=0.1):
    if np.random.random() < mutation_rate:
        idx = np.random.randint(0, len(child))
        child[idx] = np.random.uniform(-1, 1)
    return child

# 主函数
def hybrid_pso_ga(pop_size, dim, generations):
    particles = [Particle(dim) for _ in range(pop_size)]
    
    for generation in range(generations):
        # PSO部分
        g_best_position = min(particles, key=lambda x: x.best_score).best_position
        for particle in particles:
            pso_update(particle, g_best_position)
        
        # GA部分
        particles.sort(key=lambda x: x.best_score)
        for i in range(pop_size // 2):
            if np.random.random() < 0.9:  # crossover rate
                child1_position, child2_position = crossover(particles[i].position, particles[i+1].position)
                child1_position = mutate(child1_position)
                child2_position = mutate(child2_position)
                particles[i].position = child1_position
                particles[i+1].position = child2_position

    best_particle = min(particles, key=lambda x: x.best_score)
    return best_particle.best_position, best_particle.best_score

# 测试
if __name__ == "__main__":
    pop_size = 30
    dim = 1  # dimension of the problem
    generations = 100
    best_position, best_score = hybrid_pso_ga(pop_size, dim, generations)
    print(f"Best position: {best_position}, Best score: {best_score}")


# In[13]:


import numpy as np
import matplotlib.pyplot as plt

# 目标函数
def runge(x):
    return 1.0 / (1 + 25 * x**2)

# 编码
def encode(x_bounds, chrom_length):
    return np.random.randint(0, 2, chrom_length)

# 解码
def decode(chromosome, x_bounds, chrom_length):
    x = int(chromosome, 2)
    return x_bounds[0] + x * (x_bounds[1] - x_bounds[0]) / (2**chrom_length - 1)

# 选择操作
def select(population, fitnesses):
    idx = np.random.choice(np.arange(len(fitnesses)), size=2, p=fitnesses/np.sum(fitnesses))
    return population[idx]

# 交叉操作
def crossover(parents):
    if np.random.rand() > 0.9:
        return parents
    idx = np.random.randint(1, len(parents[0]))
    children = [np.hstack([parents[0, :idx], parents[1, idx:]]),
                np.hstack([parents[1, :idx], parents[0, idx:]])]
    return children

# 变异操作
def mutate(child):
    if np.random.rand() > 0.05:
        return child
    idx = np.random.randint(len(child))
    new_child = np.copy(child)
    new_child[idx] = 1 - new_child[idx]
    return new_child

# 适应度函数
def fitness(target_func, chromosome, x_bounds, chrom_length):
    x = decode(''.join([str(g) for g in chromosome]), x_bounds, chrom_length)
    return -abs(target_func(x))

# IGA算法
def IGA(target_func, x_bounds, chrom_length, pop_size, generation):
    population = np.array([encode(x_bounds, chrom_length) for _ in range(pop_size)])

    for _ in range(generation):
        fitnesses = np.array([fitness(target_func, ind, x_bounds, chrom_length) for ind in population])
        new_population = []
        for _ in range(pop_size // 2):
            parents = select(population, fitnesses)
            children = crossover(parents)
            children = np.vstack([mutate(child) for child in children])
            new_population.extend(children)
        population = np.array(new_population)
    
    best_idx = np.argmax(fitnesses)
    return decode(''.join([str(g) for g in population[best_idx]]), x_bounds, chrom_length)

# 使用IGA找到最佳插值节点
x_bounds = [-1, 1]
chrom_length = 10
pop_size = 100
generation = 100
best_x = IGA(runge, x_bounds, chrom_length, pop_size, generation)

# 使用找到的节点进行插值
x = np.linspace(-1, 1, 1000)
y_true = runge(x)
y_interp = runge(best_x)

plt.figure(figsize=(10,6))
plt.plot(x, y_true, label='True Function')
plt.scatter([best_x], [y_interp], color='red', s=100, zorder=5, label='IGA Interpolation Point')
plt.legend()
plt.title('IGA for Runge Function')
plt.show()


# In[14]:


import numpy as np
import matplotlib.pyplot as plt

# 目标函数
def runge(x):
    return 1.0 / (1 + 25 * x**2)

# 初始化种群
def initialize_population(pop_size, chrom_length):
    return np.random.uniform(-1, 1, (pop_size, chrom_length))

# 适应度函数
def fitness(individual):
    x = decode(individual)
    return -runge(x)

# 解码
def decode(individual):
    return np.sum(individual * 2 ** np.arange(individual.size)[::-1]) / (2**individual.size - 1) * 2 - 1

# 选择操作
def select(population, fitnesses):
    idx = np.random.choice(np.arange(population.shape[0]), size=2, p=fitnesses/fitnesses.sum(axis=0))
    return population[idx]

# 交叉操作
def crossover(parents):
    crossover_point = np.random.randint(0, parents.shape[1])
    child1 = np.hstack([parents[0, :crossover_point], parents[1, crossover_point:]])
    child2 = np.hstack([parents[1, :crossover_point], parents[0, crossover_point:]])
    return child1, child2

# 变异操作
def mutate(child, mutation_rate=0.1):
    if np.random.rand() < mutation_rate:
        mutation_point = np.random.randint(child.size)
        child[mutation_point] = 1 - child[mutation_point]
    return child

# IGA主程序
def IGA(target_func, x_bounds, chrom_length, pop_size, generation):
    population = initialize_population(pop_size, chrom_length)
    for _ in range(generation):
        fitnesses = np.array([fitness(ind) for ind in population])
        new_population = []
        for _ in range(pop_size // 2):
            parents = select(population, fitnesses)
            children = crossover(parents)
            children = np.vstack([mutate(child) for child in children])
            new_population.append(children)
        population = np.vstack(new_population)
    best_individual = population[np.argmax(fitnesses)]
    best_x = decode(best_individual)
    return best_x

# 参数设置
x_bounds = [-1, 1]
chrom_length = 10
pop_size = 1000
generation = 2000

# 使用IGA找到最佳插值节点
best_x = IGA(runge, x_bounds, chrom_length, pop_size, generation)

# 使用找到的节点进行插值
x = np.linspace(-1, 1, 1000)
y_true = runge(x)
y_interp = np.interp(x, [-1, best_x, 1], [runge(-1), runge(best_x), runge(1)])

plt.figure(figsize=(10,6))
plt.plot(x, y_true, label='True Function')
plt.plot(x, y_interp, '--', label='IGA Interpolated Function')
plt.legend()
plt.title('IGA for Runge Function')
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Runge函数定义
def runge(x):
    return 1.0 / (1 + 25 * x**2)

# Legendre多项式插值矩阵
def legendre_matrix(x, degree):
    N = len(x)
    P = np.zeros((N, degree + 1))
    P[:, 0] = 1
    if degree > 0:
        P[:, 1] = x
    for n in range(1, degree):
        P[:, n+1] = ((2*n+1) * x * P[:, n] - n * P[:, n-1]) / (n+1)
    return P

# 使用SVD截断解决插值问题
def svd_truncated_interpolation(x, y, degree, threshold=1e-10):
    L = legendre_matrix(x, degree)
    U, s, Vt = np.linalg.svd(L, full_matrices=False)
    inv_s = np.array([1/si if si > threshold else 0 for si in s])
    return Vt.T @ (inv_s * (U.T @ y))

# 定义均匀网格
x_uniform = np.linspace(-1, 1, 11)
y_uniform = runge(x_uniform)

# 使用不同的截断阈值
thresholds = [1e-2, 1e-5, 1e-10, 1e-15]
colors = ['r', 'g', 'b', 'm']

x_fine = np.linspace(-1, 1, 400)
y_fine = runge(x_fine)

plt.figure(figsize=(10, 6))
plt.plot(x_fine, y_fine, 'k-', label="Runge Function")
plt.plot(x_uniform, y_uniform, 'ko', label="Uniform points")

# 对于每个阈值，进行SVD截断插值，并绘制结果
for thresh, color in zip(thresholds, colors):
    coefs = svd_truncated_interpolation(x_uniform, y_uniform, 10, threshold=thresh)
    L_fine = legendre_matrix(x_fine, 10)
    y_approx = L_fine @ coefs
    plt.plot(x_fine, y_approx, color, label=f"Threshold = {thresh}")

plt.legend()
plt.title("SVD Truncated Interpolation on Uniform Grid")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()


# In[ ]:


def generate_chebyshev_nodes(N, a=-1, b=1):
    """Generate Chebyshev nodes in the interval [a, b]."""
    k = np.arange(1, N + 1)
    x_cheb = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2*k - 1) * np.pi / (2*N))
    return x_cheb

# Generate Chebyshev nodes
N = 10
x_cheb = generate_chebyshev_nodes(N)

# Compute the Vandermonde matrix using Chebyshev nodes
V_cheb = np.vander(x_cheb, increasing=True)

# Apply SVD on the Vandermonde matrix
U_cheb, Sigma_cheb, Vt_cheb = np.linalg.svd(V_cheb, full_matrices=False)

# Perform the truncated SVD interpolation for various truncation thresholds
thresholds = [1e-1, 1e-2, 1e-3, 1e-4]
interpolations_cheb = {}

for threshold in thresholds:
    # Compute the approximate inverse using truncated SVD
    Sigma_inv_truncated_cheb = np.array([1/s if s > threshold else 0 for s in Sigma_cheb])
    V_approx_inv_cheb = np.dot(Vt_cheb.T, np.dot(np.diag(Sigma_inv_truncated_cheb), U_cheb.T))
    
    # Compute the interpolation coefficients
    f_cheb_values = runge(x_cheb)
    coeffs_cheb = np.dot(V_approx_inv_cheb, f_cheb_values)
    
    # Evaluate the interpolation on a dense grid
    dense_grid = np.linspace(-1, 1, 1000)
    poly_values_cheb = np.array([np.polyval(coeffs_cheb, x_val) for x_val in dense_grid])
    interpolations_cheb[threshold] = poly_values_cheb

# Plot the original Runge function and the interpolations
plt.figure(figsize=(10, 6))
plt.plot(dense_grid, runge(dense_grid), 'k', label='Original Runge Function')
colors = ['r', 'b', 'g', 'y']
for threshold, color in zip(thresholds, colors):
    plt.plot(dense_grid, interpolations_cheb[threshold], color, label=f'Truncation Threshold: {threshold}')

plt.title('Runge Function and its Truncated SVD Interpolations using Chebyshev Nodes')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

