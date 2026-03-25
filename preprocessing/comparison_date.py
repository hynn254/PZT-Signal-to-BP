import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

base_path = Path('BP-piezo/data/processed/ref')

# 날짜 폴더 자동 탐지
date_folders = sorted([p for p in base_path.iterdir() if p.is_dir()])

target_dates = ['20260318', '20260319', '20260320']  # 원하는 날짜로 수정
date_folders = [d for d in date_folders if d.name in target_dates]

# 파일 정렬 키
def sort_key(file):
    numbers = list(map(int, re.findall(r'\d+', file.stem)))
    return numbers

# 전체 날짜에 걸쳐 등장하는 피험자 목록 수집
all_persons = set()
for date_folder in date_folders:
    for p in date_folder.iterdir():
        if p.is_dir():
            all_persons.add(p.name)
all_persons = sorted(all_persons)

target_persons = ['KYG']
all_persons = [p for p in all_persons if p in target_persons]

fig, axes = plt.subplots(len(all_persons), 1, figsize=(16, 4 * len(all_persons)))
if len(all_persons) == 1:
    axes = [axes]

for ax, person in zip(axes, all_persons):
    for date_folder in date_folders:
        person_path = date_folder / person / 'matched'
        if not person_path.exists():
            continue

        files = sorted(person_path.glob('*.csv'), key=sort_key)
        segments = []
        for f in files:
            signal = pd.read_csv(f)['Volt'].to_numpy().astype(np.float64)
            segments.append(signal)

        if not segments:
            continue

        merged = np.concatenate(segments)
        ax.plot(merged, linewidth=1.0, label=date_folder.name)

    ax.set_title(person)
    ax.set_ylim(-0.05, 0.05)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Volt')
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()