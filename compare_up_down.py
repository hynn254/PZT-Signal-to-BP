import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, data, fs, order):
    b, a = butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
    y = filtfilt(b, a, data)
    return y

name = 'JHJ'
date = '20260212'

file_name_st = f'{date}_{name}_steady'
pzt_signal_st = pd.read_csv(f'BP-piezo/data/{file_name_st}.csv')['1'].to_numpy()[1:]
pzt_signal_st = np.asarray(pzt_signal_st, dtype=np.float64)
filtered_st = butter_bandpass(0.3, 5.0, pzt_signal_st, 400, 2)

file_name_ex = f'{date}_{name}_exercise'
pzt_signal_ex = pd.read_csv(f'BP-piezo/data/{file_name_ex}.csv')['1'].to_numpy()[1:]
pzt_signal_ex = np.asarray(pzt_signal_ex, dtype=np.float64)
filtered_ex = butter_bandpass(0.3, 5.0, pzt_signal_ex, 400, 2)


plt.figure(figsize=(12, 5))
plt.plot(filtered_st, label='steady')
plt.plot(filtered_ex, label='exercise')
plt.legend()
# plt.subplot(1, 2, 1)
# # plt.plot(pzt_signal_st)
# plt.plot(filtered_st)

# plt.subplot(1, 2, 2)
# # plt.plot(pzt_signal_ex)
# plt.plot(filtered_ex)
plt.show()