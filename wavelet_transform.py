import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, data, fs, order):
    b, a = butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
    y = filtfilt(b, a, data)
    return y

# Read PZT sensor signal
file_name = '20260211_JHJ'
pzt_signal = pd.read_csv(f'BP-piezo/data/{file_name}.csv')['1'].to_numpy()[1:]
pzt_signal = np.asarray(pzt_signal, dtype=np.float64)

# fs = 1/0.0025s = 400 Hz
# pzt_denoised = butter_bandpass(0.3, 5.0, pzt_signal, 400, 3)

######## Implement paper

## Decomposition
# single level transform -> pywt.dwt
# multi-level 1D transform(decomposition) -> pywt.wavedec
# multi-level 2D -> pywt.wavedec2  
# coeffs = cA_n, cD_n, cD_n-1, ..., cD1
# cA : approximation coefficients(low freq), cD : detail coefficients(high freq)
coeffs = pywt.wavedec(data=pzt_signal, wavelet='coif9', level=4)

cA4 = coeffs[0] 
cDs = coeffs[1:]
cD1 = cDs[-1]    
print(coeffs)

## Thresholding : The universal threshold (Donoho–Johnstone)
N = len(pzt_signal)

sigma = 1.482579 * np.median(np.abs(pzt_signal - np.median(pzt_signal)))
# sigma = np.median(np.abs(cD1))  # if cD1 is almost zero-mean noise

threshold = sigma * np.sqrt(2 * np.log(N))
cDs_thresh = [pywt.threshold(cD, threshold, mode='hard') for cD in cDs]

## Inverse (= Get Denoised Signal)
coeffs_denoised = [cA4] + cDs_thresh
# single level inverse -> pywt.idwt
# multi-level inverse(reconstruction) -> pywt.waverec
# pzt_denoised_wv = pywt.waverec(coeffs_denoised, wavelet='coif9')
pzt_denoised_2 = butter_bandpass(0.3, 5.0, pzt_signal, 400, 2)
pzt_denoised_3 = butter_bandpass(0.3, 5.0, pzt_signal, 400, 3)
pzt_denoised_4 = butter_bandpass(0.3, 5.0, pzt_signal, 400, 4)
pzt_denoised_5 = butter_bandpass(0.3, 5.0, pzt_signal, 400, 5)

######## Plot
plt.figure(figsize=(20, 5))
plt.subplot(1, 5, 1)
plt.plot(pzt_signal)
plt.title('Original signal')

# plt.subplot(1, 3, 2)
# plt.plot(pzt_denoised_wv)
# plt.title('Wavelet denoised')

plt.subplot(1, 5, 2)
plt.plot(pzt_denoised_2)
plt.title('Bandpass filtering - 2nd')

plt.subplot(1, 5, 3)
plt.plot(pzt_denoised_3)
plt.title('Bandpass filtering - 3rd')

plt.subplot(1, 5, 4)
plt.plot(pzt_denoised_4)
plt.title('Bandpass filtering - 4th')

plt.subplot(1, 5, 5)
plt.plot(pzt_denoised_5)
plt.title('Bandpass filtering - 5th')

plt.tight_layout()
plt.show()
