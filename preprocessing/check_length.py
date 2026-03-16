from pathlib import Path
import numpy as np
import pandas as pd


folder = 'BP-piezo/data/raw/'
date = '20260316'
subject = 'OSJ'

pzt_files = list(Path(f'{folder}/{date}/{subject}').glob('*.csv'))

for pzt_file in pzt_files:
    signal = pd.read_csv(pzt_file)['1'][1:].to_numpy()

    if signal.size < 2000:
        print(pzt_file.name)
        Path(pzt_file).unlink()