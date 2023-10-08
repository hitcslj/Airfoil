import numpy as np
import matplotlib.pyplot as plt

def bezier_curve(control_points, t):
    n = len(control_points) - 1
    result = np.zeros_like(control_points[0])
    for i in range(n + 1):
        result += control_points[i] * binomial_coefficient(n, i) * (1 - t)**(n - i) * t**i
    return result

def binomial_coefficient(n, k):
    return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))

def ffd(deformed_points, control_points, lattice_points):
    n = len(deformed_points)
    m = len(deformed_points[0])
    for i in range(n):
        for j in range(m):
            deformed_points[i][j] = bezier_curve(control_points, lattice_points[i][j])

# Example usage
control_points = np.array([[0, 0], [1, 2], [3, 1], [4, 3]])
lattice_points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
deformed_points = np.copy(control_points)

ffd(deformed_points, control_points, lattice_points)

# Plotting the results
plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Original')
plt.plot(deformed_points[:, 0], deformed_points[:, 1], 'bo-', label='Deformed')
plt.plot(lattice_points[:, 0], lattice_points[:, 1], 'go', label='Lattice')
plt.legend()
plt.show()