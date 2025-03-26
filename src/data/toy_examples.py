import numpy as np
import torch 

# create a simple dataset that all the functions can use
def simple_3d_sde_dataset():
    # Define the SDE functions
    def FX(x, y, z):
        return -y - z

    def FY(x, y, z):
        return x + 0.2 * y

    def FZ(x, y, z):
        return 0.2 + z * (x - 5.7)

    # Generate a simple 3D dataset
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    z = np.zeros_like(t)

    # Initial conditions
    x[0], y[0], z[0] = 1.0, 1.0, 1.0

    # Euler-Maruyama method to solve the SDE
    for i in range(1, len(t)):
        x[i] = x[i-1] + FX(x[i-1], y[i-1], z[i-1]) * dt
        y[i] = y[i-1] + FY(x[i-1], y[i-1], z[i-1]) * dt
        z[i] = z[i-1] + FZ(x[i-1], y[i-1], z[i-1]) * dt

    return t, x, y, z

def simple_2d_sde_dataset():
    # Parameters
    tmax = 10000
    dt = 1.0
    a = 1.0
    DX = 0.0004
    DY = 0.0001
    sigX = (2 * DX * dt) ** 0.5
    sigY = (2 * DY * dt) ** 0.5

    # Generate dWX and dWY
    dWX = torch.normal(mean=0.0, std=sigX, size=(tmax,))
    dWY = torch.normal(mean=0.0, std=sigY, size=(tmax,))

    # Initialize X, Y, dX, and dY
    X = torch.zeros(tmax)
    Y = torch.zeros(tmax)
    dX = torch.zeros(tmax)
    dY = torch.zeros(tmax)

    # Drift functions FX and FY
    def FX(x, y):
        return dt * a * ((x - y) - x**3)

    def FY(x, y):
        return dt * a * (0.2 * (y + x) - y**3)

    # Simulate the SDE
    for t in range(tmax - 1):
        dX[t] = FX(X[t], Y[t]) + dWX[t]
        dY[t] = FY(X[t], Y[t]) + dWY[t]
        X[t + 1] = X[t] + dX[t]
        Y[t + 1] = Y[t] + dY[t]

    t = np.arange(tmax)

    # stack the data
    X = np.vstack((X.numpy(), Y.numpy())).T

    return X, t