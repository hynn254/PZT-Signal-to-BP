# PZT-Signal-to-BP
Blood pressure estimation codes using PZT sensor signal preprocessing and deep learning

## Data Directory Structure
BP-piezo/
│
├── data/
│ ├── raw/
│ ├── processed/
│ └── make_gt.py

- `raw/`: Raw PZT signals collected from experiments.
- `processed/`: Processed PZT signals used for analysis and model input.
- `make_gt.py`: Python script for organizing BP labels and saving them as a CSV file.

### data/raw/

raw/
│
├── 20260211/
│ ├── JHJ_st.csv
│ └── PDK_st.csv
├── 20260212/
│ ├── JHJ_st.csv
│ ├── JHJ_ex.csv
│ ├── ...
├── **GT**/
│ ├── 20260212.csv
│ ├── ...
├── **Mn0**/
│ ├── 20260214/
│ ├── ...
├── **Mn8**/
│ ├── 20260214/
│ ├── ...

- Data are organized by experiment date (`YYYYMMDD`).
- The sensor type is not encoded in date-based folders.
- Data were collected using two sensor types: `Mn0` and `Mn8`.
- Ground truth measurements are stored separately in the `GT` and organized by date.
#### File Naming Format
{name}_{case}.csv
- {name}: Subject identifier
- {case}: Experimental condition
  1) st: steady state
  2) ex: exercise (to increase BP)
  3) re: rest (post-exercise steady state)

### data/processed/

processed/
├── **ref**/
│ ├── 20260211/
│ ├── 20260212/
│ ├── Mn0/
│ └── Mn8/
├── ...

- `ref`: Reference preprocessing pipeline; Low-pass filtering → Wavelet transform

_Reference_
Li, M., Aoyama, J., Inayoshi, K., & Zhang, H. (2025). Wearable PZT piezoelectric sensor device for accurate arterial pressure pulse waveform measurement. Advanced Electronic Materials, 11(9), 2400852.


