import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def gaussian_noise(signal):
    white_noise = np.random.randn(len(signal))
    


date = '20260319' 
subject = 'JHJ'
pzt_files = list(Path(f'BP-piezo/data/processed/ref/{date}/{subject}/matched').glob('*.csv'))  

