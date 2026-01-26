import numpy as np
from scipy.integrate import odeint


def generate_sine_array(n_points=10000):
    t = np.linspace(0, 1000, n_points)
    return np.sin(t)


def generate_gaussian_array(n_points=10000):
    return np.random.normal(0, 1, n_points)


def generate_lorenz_array(n_points=10000, sigma=10.0, rho=28.0, beta=8.0/3.0):
    def lorenz_system(state, t, sigma, rho, beta):
        x, y, z = state
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return [dx_dt, dy_dt, dz_dt]
    
    initial_state = [1.0, 1.0, 1.0]
    t = np.linspace(0, 100, n_points)
    
    trajectory = odeint(lorenz_system, initial_state, t, args=(sigma, rho, beta))
    x_component = trajectory[:, 0]
    
    return x_component
