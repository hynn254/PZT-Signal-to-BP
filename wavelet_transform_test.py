import pywt
import numpy as np
import pandas as pd
from wavelet_transform import wavelet_denoising_1D

# Make sine signal for example
fs = 1000
duration = 1.0
freq = 5

t = np.linspace(0, duration, int(fs*duration), endpoint=False)
y = np.sin(2 * np.pi * freq * t)

# Add noise
np.random.seed(0)   # Fix noise
noise_std = 0.2
noise = np.random.normal(0, noise_std, size=len(y))

y_noise = y + noise


######## Using pywt library
# # Check 'coif' wave list
# a = pywt.wavelist('coif')
# print(a)

# # Check specification about 'coif9' function
# wavelet = pywt.Wavelet('coif9') 
# print(wavelet)

# phi, psi, x = wavelet.wavefun(level=4)
# # phi : wavelet function 
# # psi : scaling function 


######## Same as reference
y_denoised = wavelet_denoising_1D(y_noise, 'coif9', 4, 'hard')


######## Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(y)
plt.title('Original signal')

plt.subplot(1, 3, 2)
plt.plot(y_noise)
plt.title('Noisy signal')

plt.subplot(1, 3, 3)
plt.plot(y_denoised)
plt.title('Wavelet denoised')

plt.tight_layout()
plt.show()


