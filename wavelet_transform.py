import pywt
import numpy as np

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
# # Check coif wave list
# a = pywt.wavelist('coif')
# print(a)

# # Use coif9(mother wavelet) with decomposition level = 4
# wavelet = pywt.Wavelet('coif9') 
# print(wavelet)

# # phi : wavelet function 
# # psi : scaling function 
# phi, psi, x = wavelet.wavefun(level=4)


######## Implement paper

## Decomposition
# single level transform -> pywt.dwt
# multi-level 1D transform(decomposition) -> pywt.wavedec
# multi-level 2D -> pywt.wavedec2  
# coeffs = cA_n, cD_n, cD_n-1, ..., cD1
# cA : approximation coefficients(low freq), cD : detail coefficients(high freq)
coeffs = pywt.wavedec(data=y_noise, wavelet='coif9', level=4)

cA4 = coeffs[0] 
cDs = coeffs[1:]
cD1 = cDs[-1]    
print(coeffs)

## Thresholding : The universal threshold (Donoho–Johnstone)
N = len(y_noise)

sigma = 1.482579 * np.median(np.abs(y_noise - np.median(y_noise)))
# sigma = np.median(np.abs(cD1))  # if cD1 is almost zero-mean noise

threshold = sigma * np.sqrt(2 * np.log(N))
cDs_thresh = [pywt.threshold(cD, threshold, mode='hard') for cD in cDs]

## Inverse (= Get Denoised Signal)
coeffs_denoised = [cA4] + cDs_thresh
# single level inverse -> pywt.idwt
# multi-level inverse(reconstruction) -> pywt.waverec
y_denoised = pywt.waverec(coeffs_denoised, wavelet='coif9')


######## Plot
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


