from scipy.signal import butter, filtfilt
import numpy as np
import os
import matplotlib.pyplot as plt


## Filters
def butter_bandpass(data, lowcut, highcut, fs, order):
    data = data.astype(np.float64)
    b, a = butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass(data, cutoff, fs, order):
    normal_cutoff = cutoff/(0.5*fs)
    data = data.astype(np.float64)
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


## Signal types

#  Lowpass filtered signal comparison - amplitude
def lowpass_signal(x, cf=10.0):
    return butter_lowpass(x, cf, 400, 3)

#  1st derivative comparison - velocity
def first_derivative(x, fs=400):
    dt = 1/fs
    # x = butter_lowpass(x, 10.0, 400, 3)
    return np.gradient(x, dt)

def second_derivative(x, fs=400):
    dt = 1/fs
    # x = butter_lowpass(x, 10.0, 400, 3)
    x = np.gradient(x, dt)
    return np.gradient(x, dt)


## Directory functions
def create_directory(path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            pass


## Plot functions
def enter_key(event):
    if event.key == 'enter':
        plt.close('all')