from pathlib import Path
from scipy.signal import envelope, hilbert, find_peaks
from utils import sort_scope_key
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import create_directory


folder = 'BP-piezo/data/processed/ref'
date_train = '20260318'
date_test = '20260319'
subject = 'JHJ'
save_path = f'BP-piezo/results/peak_valley/{date_train}/{subject}'
create_directory(save_path)

pzt_files_train = Path(f'{folder}/{date_train}/{subject}/matched')
pzt_files_train = sorted(pzt_files_train.glob('*.csv'), key=sort_scope_key)

pzt_files_test = Path(f'{folder}/{date_test}/{subject}/matched')
pzt_files_test = sorted(pzt_files_test.glob('*.csv'), key=sort_scope_key)

gt_path = f'BP-piezo/data/GT'
   
def match_bp_(info, pzt_files, train_mode, gt_path=gt_path):
    date, subject = info[0], info[1]
    gt_file = pd.read_csv(f'{gt_path}/{date}.csv')
    SBP = gt_file.loc[gt_file['name']==subject, 'SBP'].values[0]
    DBP = gt_file.loc[gt_file['name']==subject, 'DBP'].values[0]

    meds_p = []
    meds_n = []

    for pzt_file in pzt_files:
        signal = pd.read_csv(pzt_file)['Volt'].to_numpy()

        peaks_p, _ = find_peaks(signal, prominence=0.01)
        peaks_n, _ = find_peaks(-1 * signal, prominence=0.01)
    
        med_p = np.median(signal[peaks_p])
        med_n = np.median(signal[peaks_n])

        meds_p.append(med_p)
        meds_n.append(med_n)

        # signal_envlp_p = np.abs(hilbert(signal))
        
        plt.plot(signal, color='black')
        # plt.plot(signal_envlp_p)
        plt.plot(peaks_p, signal[peaks_p], 'o', color='red')
        plt.plot(peaks_n, signal[peaks_n], 'o', color='blue')
        plt.savefig(f'{save_path}/{pzt_file.stem}.png')
        # plt.show()
        plt.close('all')

    mean_n, mean_p = np.mean(meds_n), np.mean(meds_p)
    if train_mode:
        x = np.array([mean_n, mean_p])
        y = np.array([DBP, SBP])
        poly = np.polyfit(x, y, 1)
        print(poly)
        print(f'GT: {SBP}, {DBP}')
        print(x * poly[0] + poly[1])

        return poly
    else:
        return mean_n, mean_p, SBP, DBP


poly = match_bp_([date_train, subject], train_mode=True, pzt_files=pzt_files_train)
mean_n, mean_p, sbp, dbp = match_bp_([date_test, subject], train_mode=False, pzt_files=pzt_files_test)

estimated_sbp = mean_p * poly[0] + poly[1]
estimated_dbp = mean_n * poly[0] + poly[1]
print(f'TEST BP: {dbp}, {sbp}')
print(f'ESTIMATED BP: {estimated_dbp}, {estimated_sbp}')
