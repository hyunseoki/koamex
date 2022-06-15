from zipfile import ZipFile
import glob
import os
import xml.etree.ElementTree as ET
import glob
import os
import pandas as pd
import numpy as np


###################### File Extracting from zip #######################
# BASE_PATH = r'./data/raw_data/raw_data'
# assert os.path.isdir(BASE_PATH), BASE_PATH
# fns = glob.glob(BASE_PATH + '/*.zip')

# idx = 0
# nb_zip = len(fns)
    
# for fn in fns:
#     with ZipFile(fn, 'r') as zip:        
#         folder_name = fn.split('.zip')[0]
#         folder_name = folder_name.replace(BASE_PATH + '/', '')
#         zip.extractall(folder_name)
        
#         idx += 1
#         print(f'[{idx}/{nb_zip}] {folder_name} is extracted')

# print(f'extracting done')

####################### dataset.csv #######################

def parsing(base_folders, df, phase):
    for base in base_folders:
        xml_fn = os.path.join(base, 'annotations.xml')
        tree = ET.parse(xml_fn)
        root = tree.getroot()

        for node in root[2:]:
            task_n = base.split('/')[-1]
            image_fn = node.attrib['name']
            img_w = node.attrib['width']
            img_h = node.attrib['height']

            key_points1 = None
            key_points2 = None

            for sub_node in node:
                if sub_node.attrib['label'] == 'point1':
                    key_points1 = sub_node.attrib['points'] ## 15 pairs
                elif sub_node.attrib['label'] == 'point2':
                    key_points2 = sub_node.attrib['points'] ## 3 pairs
                else:
                    raise NameError()

            if (key_points1 != None) and (key_points2 != None):
                key_points1 = key_points1.split(';')
                key_points2 = key_points2.split(';')

                if len(key_points1)==30 and len(key_points2)==6:
                    row = dict()
                    row['phase'] = phase
                    row['task'] = task_n
                    row['fn'] = image_fn
                    row['img_h'] = img_h
                    row['img_w'] = img_w

                    for idx, point in enumerate(key_points1):
                        p_y, p_x = np.array(point.split(','), dtype='float').astype('int')
                        row[f'keypoint1_{idx}_x'] = p_x
                        row[f'keypoint1_{idx}_y'] = p_y

                    for idx, point in enumerate(key_points2):
                        p_y, p_x = np.array(point.split(','), dtype='float').astype('int')
                        row[f'keypoint2_{idx}_x'] = p_x
                        row[f'keypoint2_{idx}_y'] = p_y          

                df = df.append(row, ignore_index=True)

    return df

TRAIN_BASE = [
     './data/raw_data/task_2_eos 4-2020_10_19_01_02_11-cvat for images 1.1',
     './data/raw_data/task_3_eos 5-1-2020_10_19_00_29_57-cvat for images 1.1',
     './data/raw_data/task_5_eos 5-2-2020_10_19_00_31_53-cvat for images 1.1',
     './data/raw_data/task_6_eos 6-1-2020_10_19_00_37_05-cvat for images 1.1',
     './data/raw_data/task_7_eos 6-2-2020_10_19_00_38_57-cvat for images 1.1',
     './data/raw_data/task_8_eos 7-1-2020_10_19_00_41_04-cvat for images 1.1',
     './data/raw_data/task_9_eos 7-2-2020_10_19_00_43_00-cvat for images 1.1',
     './data/raw_data/task_10_eos 7-3-2020_10_19_00_44_35-cvat for images 1.1',
     './data/raw_data/task_11_eos 8-1-2020_10_19_00_46_07-cvat for images 1.1',
     './data/raw_data/task_12_eos 8-2-2020_10_19_00_48_30-cvat for images 1.1',
     './data/raw_data/task_13_eos 8-3-2020_10_19_00_49_43-cvat for images 1.1',
     './data/raw_data/task_14_eos 9-1-2020_10_19_00_51_17-cvat for images 1.1',
     './data/raw_data/task_15_eos 9-2-2020_10_19_02_01_15-cvat for images 1.1',
     './data/raw_data/task_16_eos 9-3-2020_10_19_01_01_10-cvat for images 1.1',    
     './data/raw_data/task_20_eos 12-1-2020_10_19_01_38_42-cvat for images 1.1',
]

VALID_BASE = [
     './data/raw_data/task_17_eos 10-1-2020_10_19_01_52_10-cvat for images 1.1',
     './data/raw_data/task_18_eos 10-2-2020_10_19_01_07_45-cvat for images 1.1',    
]

TEST_BASE = [
    # './data/raw_data/task_19_eos 11-1-2020_10_19_01_45_57-cvat for images 1.1',
]

cols = ['phase', 'task', 'fn', 'img_h', 'img_w']

for i in range(30):
    cols.append(f'keypoint1_{i}_y')
    cols.append(f'keypoint1_{i}_x')
        
for i in range(6):
    cols.append(f'keypoint2_{i}_y')
    cols.append(f'keypoint2_{i}_x')

dataset_csv = pd.DataFrame(columns=cols)

dataset_csv = parsing(
    base_folders=TRAIN_BASE,
    df=dataset_csv,
    phase='train'
)

dataset_csv = parsing(
    base_folders=VALID_BASE,
    df=dataset_csv,
    phase='valid'
)

dataset_csv = parsing(
    base_folders=TEST_BASE,
    df=dataset_csv,
    phase='test'
)

# dataset_csv.drop([284, 585, 695, 726], axis=0, inplace=True)

for fn in ['030028100001.png', '029319890001.png', '027399950001.png', '029980630001.png']:
    dataset_csv.drop(dataset_csv[dataset_csv['fn'].str.contains(fn)].index, inplace=True)

dataset_csv.to_csv('./data/data_split.csv', index=False, sep=',')

train_case, val_case, test_case = len(dataset_csv[dataset_csv['phase'] == 'train']), len(dataset_csv[dataset_csv['phase'] == 'valid']), len(dataset_csv[dataset_csv['phase'] == 'test'])
print(f'train_case : {train_case}, val_case : {val_case}, test_case : {test_case}') ## train_case : 887, val_case : 139, test_case : 68