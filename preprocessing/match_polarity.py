import numpy as np
from pathlib import Path
import pandas as pd
from utils import first_derivative, create_directory, enter_key
from scipy.stats import skew
import matplotlib.pyplot as plt


## Function for matching polarity
def match_polarity_(signal):
    first_dev = first_derivative(signal)
    skewness = skew(first_dev)
    
    if skewness >= 0:
        matched_signal = signal
    else:
        matched_signal = -1 * signal
    
    return matched_signal, skewness



## Use preprocessed signal (lowpass filtering - wavelet denoising)
folder = 'BP-piezo/data/processed/ref/Mn8'
date = '20260219'

pzt_files = list(Path(f'{folder}/{date}').glob('*.csv'))

for file in pzt_files:
    signal = pd.read_csv(file)['Volt'].to_numpy().astype(np.float64)
    matched_signal, skewness = match_polarity_(signal)

    # Check skewness
    print(f'{file.stem}: {skewness}')

    # Save matched signal to CSV file
    create_directory(f'{folder}/{date}/matched')
    matched_df = pd.DataFrame({'Volt': matched_signal})
    matched_df.to_csv(f'{folder}/{date}/matched/{file.name}')

    # Plot the original signal and result
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(signal)
    plt.title('Before')
    plt.subplot(1, 2, 2)
    plt.plot(matched_signal)
    plt.title('After')
    plt.gcf().canvas.mpl_connect('key_press_event', enter_key)
    plt.show()
    plt.close('all') 
    
    