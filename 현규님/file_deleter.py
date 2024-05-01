import os

file_list = os.listdir('D:/aihub/136-1.극한 소음 음성 인식 데이터/01-1.정식개방데이터/Training/01.원천데이터')

for file in file_list:
    if file.endswith('VN.wav'):
        os.remove(f'D:/aihub/136-1.극한 소음 음성 인식 데이터/01-1.정식개방데이터/Training/01.원천데이터/{file}')
    else:
        continue

file_list = os.listdir('D:/aihub/136-1.극한 소음 음성 인식 데이터/01-1.정식개방데이터/Training/02.라벨링데이터')

for file in file_list:
    if file.endswith('json'):
        os.remove(f'D:/aihub/136-1.극한 소음 음성 인식 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/{file}')
    else:
        continue