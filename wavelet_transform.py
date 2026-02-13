import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filters import butter_bandpass


def wavelet_denoising_1D(data, mother_wavelet: str, level: int, thrs_mode: str):
    '''
    Denoising the signal using wavelet transform
    
    @param data: The input 1D data
    @param mother_wavelet: The wavelet function for mother wavelet (std) - 'coif9', ...
    @param level: The decomposition level (int)
    @param thrs_mode: The mode of thresholding (str) - 'soft', 'hard', ..
    @return data_denoised: Denoised data using wavelet transform 

    1. Transform
    * single level transform -> pywt.dwt
    * multi-level 1D transform(decomposition) -> pywt.wavedec (V)
    * multi-level 2D -> pywt.wavedec2  

    2. Thresholding
    * the universal threshold (Donoho–Johnstone)

    3. Inverse (= Get Denoised Signal)
    * single level inverse -> pywt.idwt
    * multi-level inverse(reconstruction) -> pywt.waverec (V)
    '''

    ## 1. Transform
    coeffs = pywt.wavedec(data=data, wavelet=mother_wavelet, level=level)
    # coeffs = cA_n, cD_n, cD_n-1, ..., cD1
    # cA : approximation coefficients(low freq), cD : detail coefficients(high freq)

    cA = coeffs[0]
    cDs = coeffs[1:]

    ## 2. Thresholding 
    N = len(data)

    sigma = 1.482579 * np.median(np.abs(pzt_signal - np.median(pzt_signal)))
    # sigma = np.median(np.abs(cD1))  # if cD1 is almost zero-mean noise

    threshold = sigma * np.sqrt(2 * np.log(N))
    cDs_thresh = [pywt.threshold(cD, threshold, mode=thrs_mode) for cD in cDs]

    ## 3. Inverse 
    coeffs_denoised = [cA] + cDs_thresh
    data_denoised = pywt.waverec(coeffs_denoised, wavelet=mother_wavelet)

    return data_denoised



######## Read PZT sensor signal
# fs = 1/0.0025s = 400 Hz
file_name = '20260211_JHJ'
pzt_signal = pd.read_csv(f'BP-piezo/data/{file_name}.csv')['1'].to_numpy()[1:]
pzt_signal = np.asarray(pzt_signal, dtype=np.float64)


######## Implement paper 
# Only wavelet transform (OW)
pzt_denoised_ow = wavelet_denoising_1D(pzt_signal, 'coif9', 4, 'hard')

# Wavelet transform + Bandpass filtering (WB)
pzt_denoised_wb = butter_bandpass(0.3, 5.0, pzt_denoised_ow, 400, 3)

# Only bandpass filtering (OB)
pzt_denoised_2 = butter_bandpass(0.3, 5.0, pzt_signal, 400, 2)
pzt_denoised_3 = butter_bandpass(0.3, 5.0, pzt_signal, 400, 3)
pzt_denoised_4 = butter_bandpass(0.3, 5.0, pzt_signal, 400, 4)
pzt_denoised_5 = butter_bandpass(0.3, 5.0, pzt_signal, 400, 5)


######## Plot
plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
plt.plot(pzt_signal)
plt.title('Original signal')

plt.subplot(1, 3, 2)
plt.plot(pzt_denoised_3)
plt.title('Wavelet denoised')

# plt.subplot(1, 5, 2)
# plt.plot(pzt_denoised_2)
# plt.title('Bandpass filtering - 2nd')

# plt.subplot(1, 5, 3)
# plt.plot(pzt_denoised_3)
# plt.title('Bandpass filtering - 3rd')

# plt.subplot(1, 5, 4)
# plt.plot(pzt_denoised_4)
# plt.title('Bandpass filtering - 4th')

# plt.subplot(1, 5, 5)
# plt.plot(pzt_denoised_5)
# plt.title('Bandpass filtering - 5th')

plt.tight_layout()
plt.show()
