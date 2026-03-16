from scipy.integrate import cumulative_trapezoid
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend


date = '20260309'
data_folder =f'BP-piezo/data/processed/ref/{date}'
pzt_files = list(Path(data_folder).glob('*.csv'))


def integrate_signal(signal, fs=400):
    integrated = cumulative_trapezoid(signal, dx=1/fs, initial=0)
    return integrated

test_file = f'BP-piezo/data/processed/ref/{date}/JHJ_st3.csv'
signal = pd.read_csv(test_file).to_numpy().flatten()
integrated = integrate_signal(signal)

# plt.subplot(2, 1, 1)
plt.plot(signal)
# plt.subplot(2, 1, 2)
# plt.plot(integrated)
plt.show()

# def integrate_by_cycle(signal, cycle_length, fs=400, offset=100):
#     n_cycles = len(signal) // cycle_length - 1  # 마지막 사이클 버림
#     integrated_cycles = []
    
#     for i in range(n_cycles):
#         start = i * cycle_length + offset
#         end = (i + 1) * cycle_length + offset
#         cycle = signal[start:end]
#         integrated = cumulative_trapezoid(cycle, dx=1/fs, initial=0)
#         integrated_cycles.append(integrated)
    
#     return np.array(integrated_cycles)

def integrate_by_cycle(signal, cycle_length, fs=400):
    n_cycles = len(signal) // cycle_length
    integrated_cycles = []
    
    for i in range(n_cycles):
        cycle = signal[i*cycle_length : (i+1)*cycle_length]
        integrated = cumulative_trapezoid(cycle[100:], dx=1/fs, initial=0)
        integrated_cycles.append(integrated)
    
    return np.array(integrated_cycles)

integrated_cycles = integrate_by_cycle(signal, cycle_length=250)

fig, axes = plt.subplots(4, 2, figsize=(12, 10))
axes_flat = axes.flatten()

# 적분하고 detrend해봤는데 그래도 이건 굳이 할 필요 없어 보임,,. 원래 파형 특성이 다 사라지는 듯 
for i, cycle in enumerate(integrated_cycles):
    axes_flat[i].plot(detrend(cycle))
    axes_flat[i].set_title(f'Cycle {i+1}', fontsize=9)
    axes_flat[i].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()