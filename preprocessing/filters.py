from scipy.signal import butter, filtfilt

def butter_bandpass(data, lowcut, highcut, fs, order):
    b, a = butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass(data, cutoff, fs, order):
    normal_cutoff = cutoff/(0.5*fs)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
