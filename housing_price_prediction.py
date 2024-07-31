import ssl
import certifi
from urllib.request import urlopen

import numpy as np
from scipy.integrate import odeint
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sage.all import *

# Fix for SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Fetch the California housing dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Use only a subset of the data (e.g., 1000 samples)
subset_size = 1000
indices = np.random.choice(np.arange(X.shape[0]), size=subset_size, replace=False)
X_subset = X[indices]
y_subset = y[indices]

# Feature Engineering: Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_subset)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Normalize the target variable
y_mean = y_subset.mean()
y_std = y_subset.std()
y_scaled = (y_subset - y_mean) / y_std

# Add an intercept term
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

# ODE Approach
def model(theta, t, X, y):
    m, n = X.shape
    gradient = np.zeros(n)
    for i in range(m):
        error = np.dot(X[i], theta) - y[i]
        gradient += error * X[i]
    return -gradient / m

# Initial conditions: Assume starting with zero parameters
initial_theta = np.zeros(X_scaled.shape[1])

# Increase the number of time points for better convergence (e.g., from 0 to 500 in steps of 1)
t = np.linspace(0, 500, 500)

# Solve the ODE system
theta_solution = odeint(model, initial_theta, t, args=(X_scaled, y_scaled))

# Use the final parameter values for prediction
final_theta_ode = theta_solution[-1]

# Predict house prices
predictions_scaled_ode = X_scaled.dot(final_theta_ode)
predictions_ode = predictions_scaled_ode * y_std + y_mean

# Calculate loss for ODE approach
loss_ode = [np.mean((X_scaled.dot(theta) - y_scaled)**2) for theta in theta_solution]

# Gradient Descent Approach
learning_rate = 0.01
n_iterations = 1000
theta_gd = np.zeros(X_scaled.shape[1])
loss_gd = []

for iteration in range(n_iterations):
    gradients = 2 / subset_size * X_scaled.T.dot(X_scaled.dot(theta_gd) - y_scaled)
    theta_gd = theta_gd - learning_rate * gradients
    loss_gd.append(np.mean((X_scaled.dot(theta_gd) - y_scaled)**2))

# Predict house prices using gradient descent
predictions_scaled_gd = X_scaled.dot(theta_gd)
predictions_gd = predictions_scaled_gd * y_std + y_mean

# Plot loss curves
loss_plot_ode = list_plot(list(zip(t, loss_ode)), color='blue', legend_label='ODE Loss')
loss_plot_gd = list_plot(list(zip(range(n_iterations), loss_gd)), color='red', legend_label='GD Loss')
loss_combined_plot = loss_plot_ode + loss_plot_gd
loss_combined_plot.set_legend_options(loc='upper right')
loss_combined_plot.axes_labels(['Iterations', 'Loss'])
title_text_loss = text("Loss Curves: ODE vs Gradient Descent", (250, max(loss_ode) * 0.9), fontsize=10, horizontal_alignment='center')
loss_combined_plot = loss_combined_plot + title_text_loss

# Plot predictions
actual_plot = list_plot([(i, y_subset[i]) for i in range(len(y_subset))], color='blue', legend_label='Actual Prices', size=10)
predicted_plot_ode = list_plot([(i, predictions_ode[i]) for i in range(len(y_subset))], color='red', legend_label='Predicted Prices (ODE)', size=10)
predicted_plot_gd = list_plot([(i, predictions_gd[i]) for i in range(len(y_subset))], color='green', legend_label='Predicted Prices (GD)', size=10)

# Combine both plots
combined_plot_ode = actual_plot + predicted_plot_ode
combined_plot_gd = actual_plot + predicted_plot_gd

combined_plot_ode.set_legend_options(loc='upper right')
combined_plot_ode.axes_labels(['Sample Index', 'House Price'])
title_text_ode = text("Actual vs Predicted House Prices (ODE)", (len(y_subset) / 2, max(y_subset) * 1.1), fontsize=10, horizontal_alignment='center')
final_plot_ode = combined_plot_ode + title_text_ode

combined_plot_gd.set_legend_options(loc='upper right')
combined_plot_gd.axes_labels(['Sample Index', 'House Price'])
title_text_gd = text("Actual vs Predicted House Prices (GD)", (len(y_subset) / 2, max(y_subset) * 1.1), fontsize=10, horizontal_alignment='center')
final_plot_gd = combined_plot_gd + title_text_gd

# Show the plots
loss_combined_plot.show()
final_plot_ode.show()
final_plot_gd.show()
