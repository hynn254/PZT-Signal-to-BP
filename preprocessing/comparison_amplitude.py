import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import first_derivative
import re
from pathlib import Path

#### USE MATCHED PROCESSED SIGNAL 
name = 'JHJ'
date = '20260214'
sensor = 'Mn8'
num = '2'
method = 'first_derivative'

#### 1) Compare one-by-one (5s length)
file_name_st = f'{name}_st{num}'
pzt_signal_st = pd.read_csv(f'BP-piezo/data/processed/ref/{sensor}/{date}/matched/{file_name_st}.csv')['Volt'].to_numpy()
pzt_signal_st = np.asarray(pzt_signal_st, dtype=np.float64)

file_name_ex = f'{name}_ex{num}'
pzt_signal_ex = pd.read_csv(f'BP-piezo/data/processed/ref/{sensor}/{date}/matched/{file_name_ex}.csv')['Volt'].to_numpy()
pzt_signal_ex = np.asarray(pzt_signal_ex, dtype=np.float64)

plt.figure(figsize=(12, 5))
plt.plot(pzt_signal_st, label='steady')
plt.plot(pzt_signal_ex, label='exercise')
plt.legend()
plt.title('Lowpass filtered - steady vs. exercise')
# plt.savefig(f'BP-piezo/results/st-ex-comparison/{method}/{date}_{sensor}_{name}_all_{num}.png')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(pzt_signal_st)
plt.subplot(1, 2, 2)
plt.plot(pzt_signal_ex)
plt.show()


#### 2) Plot ALL signal successively in one figure

## Sort files
pzt_files = list(Path(f'BP-piezo/data/processed/ref/{sensor}/{date}/matched').glob('*.csv'))  # When using data that records date and sensor type
# pzt_files = list(Path(f'BP-piezo/data/processed/ref/{date}').glob('*.csv'))             # When using data that only records date

def sort_key(file):
    name, state = file.stem.split("_")   # JHJ_st1
    case = re.match(r"[a-z]+", state).group()  # st
    case_order = {"st": 0, "ex": 1, "re": 2}

    if re.search(r"\d+", state).group():
        num = int(re.search(r"\d+", state).group())
    else:
        num = 1     # No number = Measure only once 
    
    return (name, case_order[case], num)

sorted_pzt_files = sorted(pzt_files, key=sort_key)

## Plot fucntion
def plot_by_case(sorted_files, compare_method=None, title=""):
    colors = {"st": 'silver', "ex": 'coral', "re": 'seagreen'}

    start = 0
    merged_signal = []

    plt.figure(figsize=(12, 5))

    for pzt_file in sorted_files:
        name, rest = pzt_file.stem.split("_")
        case = re.match(r"[a-z]+", rest).group()

        signal = pd.read_csv(pzt_file)['Volt'].to_numpy().astype(np.float64)

        if compare_method is not None:
            signal = compare_method(signal)
        else:
            signal = signal

        end = start + len(signal)

        plt.plot(range(start, end), signal, color='k')
        plt.axvspan(start, end, color=colors[case], alpha=0.1, linewidth=0)

        merged_signal.append(signal)
        start = end

    plt.title(f'({name}) {title}')
    plt.show()

    return np.concatenate(merged_signal)


origin_merged = plot_by_case(sorted_pzt_files, None, title='Original signal')
first_merged = plot_by_case(sorted_pzt_files, first_derivative, title='1st derivative signal')

