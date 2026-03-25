import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

date = '20260318'
base_path = Path(f'BP-piezo/data/processed/ref/{date}')

# 사람별 폴더 자동 탐지
person_folders = sorted([p for p in base_path.iterdir() if p.is_dir()])

# scope_1_1.csv 형식 숫자 기준 정렬
def sort_key(file):
    numbers = list(map(int, re.findall(r'\d+', file.stem)))  # [1, 1], [2, 1], ...
    return numbers

fig, axes = plt.subplots(len(person_folders), 1, figsize=(16, 4 * len(person_folders)))
if len(person_folders) == 1:
    axes = [axes]

for ax, folder in zip(axes, person_folders):
    files = sorted((folder / 'matched').glob('*.csv'), key=sort_key)
    
    segments = []
    for f in files:
        signal = pd.read_csv(f)['Volt'].to_numpy().astype(np.float64)
        segments.append(signal)
    
    merged = np.concatenate(segments)
    ax.plot(merged, color='k', linewidth=0.5)
    ax.set_title(folder.name)
    ax.set_ylim(-0.05, 0.05)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Volt')

plt.tight_layout()
plt.show()