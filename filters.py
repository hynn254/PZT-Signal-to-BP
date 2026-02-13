from scipy.signal import butter, filtfilt

def butter_bandpass(data, lowcut, highcut, fs, order):
    b, a = butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
    y = filtfilt(b, a, data)
    return y