import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import create_directory, enter_key

def gaussian_noise(signal, sigma, save_path):
    # noise = np.random.randn(len(signal))
    # noise = np.random.normal(0, sigma * np.std(signal), size=len(signal))
    noise = np.random.normal(0, sigma, size=len(signal))
    # noised_signal = signal + sigma * noise
    noised_signal = signal + noise

    y_min = np.min(signal) - 0.005
    y_max = np.max(signal) + 0.005

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(signal)
    plt.ylim(y_min, y_max)
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.plot(noised_signal)
    plt.title('Noised')
    plt.ylim(y_min, y_max)
    plt.savefig(f'{save_path}.png')

    plt.gcf().canvas.mpl_connect('key_press_event', enter_key)
    plt.show()
    # plt.close('all')

    return noised_signal


def shifting(signal, shift_rate, save_path):
    shifted_signal = np.roll(signal, int(len(signal) * shift_rate))
    
    y_min = np.min(signal) - 0.005
    y_max = np.max(signal) + 0.005

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(signal)
    plt.ylim(y_min, y_max)
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.plot(shifted_signal)
    plt.title('Shifted')
    plt.ylim(y_min, y_max)
    plt.savefig(f'{save_path}.png')

    plt.gcf().canvas.mpl_connect('key_press_event', enter_key)
    plt.show()
    # plt.close('all')

    return shifted_signal


def gaussian_snr(signal, snr_db, save_path):
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / snr_linear
    sigma = np.sqrt(noise_power)
    noise = np.random.normal(0, sigma, signal.shape)

    noised_signal = signal + noise

    y_min = np.min(signal) - 0.005
    y_max = np.max(signal) + 0.005

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(signal)
    plt.ylim(y_min, y_max)
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.plot(noised_signal)
    plt.title('Noised')
    plt.ylim(y_min, y_max)
    plt.savefig(f'{save_path}.png')

    plt.gcf().canvas.mpl_connect('key_press_event', enter_key)
    plt.show()
    # plt.close('all')
    return noised_signal

data_path = 'BP-piezo/data/processed'
date = '20260318' 
subject = 'KYG'   

pzt_files = list(Path(f'{data_path}/ref/{date}/{subject}/matched').glob('*.csv'))  
folder_path = f'{data_path}/augmentation'
result_path = f'BP-piezo/results/augmentation'

for pzt_file in pzt_files:
    signal = pd.read_csv(pzt_file)['Volt'].to_numpy()
    
    file_path = f'{date}/{subject}'
    gn_file_path = f'{folder_path}/gaussian/{file_path}'
    sh_file_path = f'{folder_path}/shifting/{file_path}'
    gn_snr_file_path = f'{folder_path}/gaussian_20dB/{file_path}'
    
    gn_savefig_path = f'{result_path}/gaussian/{file_path}'
    sh_savefig_path = f'{result_path}/shifting/{file_path}'
    gn_snr_savefig_path = f'{result_path}/gaussian_20dB/{file_path}'
    
    create_directory(gn_file_path)
    create_directory(sh_file_path)
    create_directory(gn_snr_file_path)
    create_directory(gn_savefig_path)
    create_directory(sh_savefig_path)
    create_directory(gn_snr_savefig_path)

    # gn_signal = gaussian_noise(
    #     signal=signal, 
    #     sigma=0.001, 
    #     save_path=f'{gn_savefig_path}/{pzt_file.stem}'
    #     )
    
    # sh_signal = shifting(
    #     signal=signal, 
    #     shift_rate=0.02, 
    #     save_path=f'{sh_savefig_path}/{pzt_file.stem}'
    #     )

    gn_snr_signal = gaussian_snr(
        signal=signal, 
        snr_db=20, 
        save_path=f'{gn_snr_savefig_path}/{pzt_file.stem}'
        )
    
    # gn_signal_df = pd.DataFrame({'Volt': gn_signal})
    # gn_signal_df.to_csv(f'{gn_file_path}/{pzt_file.stem}.csv', index=False)

    # sh_signal_df = pd.DataFrame({'Volt': sh_signal})
    # sh_signal_df.to_csv(f'{sh_file_path}/{pzt_file.stem}.csv')

    gn_snr_signal_df = pd.DataFrame({'Volt': gn_snr_signal})
    gn_snr_signal_df.to_csv(f'{gn_snr_file_path}/{pzt_file.stem}.csv', index=False)
