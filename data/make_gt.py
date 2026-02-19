import pandas as pd
import os

date = str(input("Date (ex. 20260101): "))
path = f'BP-piezo/data/raw/{date}.csv'

while True:
    name = str(input("Name (ex. JHJ): "))
    sbp = int(input("SBP: "))
    dbp = int(input("DBP: "))
    case_num = int(input("Case | 1. steady, 2. exercise, 3. rest(steady after exercise): "))
    sensor_num = int(input("Sensor | 1. Mn0, 2. Mn8: "))

    if case_num == 1:
        case = 'steady'
    elif case_num == 2:
        case = 'exercise'
    elif case_num == 3:
        case = 'rest'
    else:
        print('Invalid case number')
        continue

    if sensor_num == 1:
        sensor = 'Mn0'
    elif sensor_num == 2:
        sensor = 'Mn8'
    else:
        print('Invalid case number')
        continue

    df = pd.DataFrame({'name': [name], 'SBP': [sbp], 'DBP': [dbp], 'case': [case], 'sensor': [sensor]})    
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)     # 추가할 때마다 데이터 덧붙여서 쓰기, 파일이 존재하면 헤더 반복적으로 쓰기 X
    print('Saved')

    go = input("More? (Y/N): ").upper()
    if go != 'Y':
        break




