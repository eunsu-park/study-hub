"""Exercises for Lesson 10: Bayesian Time Series — Topic: Probabilistic_Programming"""
import numpy as np

# Exercise 1: Implement a Kalman filter for a local level model.
def exercise_kalman_filter():
    # TODO: Generate random walk data with Gaussian noise
    # TODO: Implement predict/update Kalman filter steps
    # TODO: Plot filtered estimates vs true state
    pass

# Exercise 2: Bayesian seasonal decomposition using Fourier features.
def exercise_seasonal_decomposition():
    # TODO: Generate data: trend + weekly seasonality + noise
    # TODO: Build design matrix with Fourier features (period=7)
    # TODO: Fit Bayesian regression and extract components
    pass

# Exercise 3: Bayesian changepoint detection.
def exercise_changepoint():
    # TODO: Generate piecewise-constant data with 2 changepoints
    # TODO: Build PyMC model with discrete changepoint parameters
    # TODO: Run MCMC and estimate changepoint locations
    pass

# Exercise 4: Forecast with uncertainty using a Bayesian AR model.
def exercise_ar_forecast():
    # TODO: Generate AR(2) data
    # TODO: Fit Bayesian AR(2) model
    # TODO: Generate 20-step-ahead forecasts with prediction intervals
    pass

if __name__ == "__main__":
    for ex in [exercise_kalman_filter, exercise_seasonal_decomposition,
               exercise_changepoint, exercise_ar_forecast]:
        print(f"\n{ex.__name__}")
        ex()
