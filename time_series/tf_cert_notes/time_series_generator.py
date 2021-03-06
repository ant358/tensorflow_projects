# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def autocorrelation(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    φ = 0.8
    ar = rnd.randn(len(time) + 1)
    for step in range(1, len(time) + 1):
        ar[step] += φ * ar[step - 1]
    return ar[1:] * amplitude


def impulses(time, num_impulses, amplitude=1, seed=None):
    rnd = np.random.RandomState(seed)
    impulse_indices = rnd.randint(len(time), size=10)
    series = np.zeros(len(time))
    for index in impulse_indices:
        series[index] += rnd.rand() * amplitude
    return series


def generate_series(time=np.arange(1460), 
                    baseline=10, 
                    amplitude=40, 
                    slope=0.05, 
                    noise_level=5):
    """Generates a time series with a trend, seasonality, noise, and impulses
                    
    Arguments: 
        time {np.array} -- time array
        baseline {int} -- baseline value
        amplitude {int} -- amplitude of the seasonal pattern
        slope {float} -- slope of the trend
        noise_level {int} -- noise level

    Returns:
        np.array -- time series
    
    """
    return baseline + trend(time, slope) +\
                      seasonality(time, period=365, amplitude=amplitude) +\
                      white_noise(time, noise_level, seed=42)


if __name__ == '__main__':
    time = np.arange(4 * 365 + 1)
    series = generate_series(time=time)

    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()


# %%
