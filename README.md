# PZT-Signal-to-BP
Blood pressure estimation codes using PZT sensor signal preprocessing and deep learning

## Data Directory Structure
```
BP-piezo/
│
├── data/
│ ├── *GT/
│ ├── *processed/
│ ├── *raw/
│ └── *make_gt.py
```

- `GT/`: Ground truth measurements - name, SBP, DBP, case, sensor 
- `processed/`: Processed PZT signals used for analysis and model input.
- `raw/`: Raw PZT signals collected from experiments.
- `make_gt.py`: Python script for organizing BP labels and saving them as a CSV file.


### data/raw/
#### Naming Format
- Date: Experiment date
- Sensor type: Sensor type used in the experiment
  - Mn0, Mn8: Sensors with the same production method, but different type
  - small(Not used in folder name, but in GT--sensor): Sensor with the different production method, and it's smaller than Mn0/Mn8
- {name}: Subject identifier
- {case}: Experimental condition
  - st: steady state
  - ex: exercise (to increase BP)
  - re: rest (post-exercise steady state)
- {num}: Segment index (5-second window, sequentially ordered)
  - Segments were manually inspected and low-quality ones were excluded, resulting in exactly 20 segments per acquisition.


#### Type 1. Date/{name}_{case}{num}.csv
```
YYYYMMDD/
├── {name}_{case}{num}.csv
├── ...
```
- 20260211
- 20260212
- 20260219
- 20260309
- 20260313


#### Type 2. Date/Sensor type/{name}_{case}{num}.csv
```
YYYYMMDD/
├── Mn0
│ ├── {name}_{case}{num}.csv
│ ├── ...
├── Mn8
│ ├── {name}_{case}{num}.csv
│ ├── ...
```
- 20260214
- 20260303 


#### Type 3. Date/Name/scope_{num}_1.csv (⭐ Used for model training - except 20260316)
```
YYYYMMDD/
├── JHJ
│ ├── scope_{num}_1.csv
│ ├── ...
├── PDK
│ ├── scope_{num}_1.csv
│ ├── ...
```
- 20260316
- 20260318
- 20260319
- 20260320
- 20260323
- 20260324


### data/processed/
```
processed/
├── *ref/
│ ├── *matched/
│ ├── ... (pzt signal files)
├── *augmentation
│ ├── *gaussian/
│ ├── *gaussian_20dB/
│ └── *shifting/
```
- `ref`: Reference preprocessing pipeline; Low-pass filtering → Wavelet transform
- `matched`: Version with preprocessing(low-pass → wavelet) and signal polarity matching completed by `match_polarity.py` 
- `augmentation`: Augmented data by `augment_data.py`
  - `gaussian`: Noise injection - Fix std (0.001)
  - `gaussian_20dB`: Noise injection - Fix noise (20dB)
  - `shifting`: Signal shift (20%)

 
## Reference
Li, M., Aoyama, J., Inayoshi, K., & Zhang, H. (2025). Wearable PZT piezoelectric sensor device for accurate arterial pressure pulse waveform measurement. Advanced Electronic Materials, 11(9), 2400852.


