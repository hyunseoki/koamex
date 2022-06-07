from zipfile import ZipFile
import glob
import os
import xml.etree.ElementTree as ET
import glob
import os
import pandas as pd

###################### File Extracting from zip #######################
# BASE_PATH = r'./data/raw_data'
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

################## dataset.csv ################

def parsing(base_folders, df, phase):
    for base in base_folders:
        xml_fn = os.path.join(base, 'annotations.xml')
        tree = ET.parse(xml_fn)
        root = tree.getroot()

        for node in root[2:]:
            image_fn = os.path.join(base, 'images', node.attrib['name'])
            key_points1 = None
            key_points2 = None

            for sub_node in node:
                if sub_node.attrib['label'] == 'point1':
                    key_points1 = sub_node.attrib['points'] ## 15 pairs
                elif sub_node.attrib['label'] == 'point2':
                    key_points2 = sub_node.attrib['points'] ## 3 pairs
                else:
                    raise NameError()

            if (key_points1 != None) and (key_points2 != None) and \
               (len(key_points1.split(';')) == 30) and (len(key_points2.split(';')) == 6):
                df = df.append(
                    {
                        'phase': phase,
                        'fn': image_fn,
                        'keypoints1': key_points1,
                        'keypoints2': key_points2,               
                    }, ignore_index=True,
                )
        
    return df

TRAIN_BASE = [
     './data/task_2_eos 4-2020_10_19_01_02_11-cvat for images 1.1',
     './data/task_3_eos 5-1-2020_10_19_00_29_57-cvat for images 1.1',
     './data/task_5_eos 5-2-2020_10_19_00_31_53-cvat for images 1.1',
     './data/task_6_eos 6-1-2020_10_19_00_37_05-cvat for images 1.1',
     './data/task_7_eos 6-2-2020_10_19_00_38_57-cvat for images 1.1',
     './data/task_8_eos 7-1-2020_10_19_00_41_04-cvat for images 1.1',
     './data/task_9_eos 7-2-2020_10_19_00_43_00-cvat for images 1.1',
     './data/task_10_eos 7-3-2020_10_19_00_44_35-cvat for images 1.1',
     './data/task_11_eos 8-1-2020_10_19_00_46_07-cvat for images 1.1',
     './data/task_12_eos 8-2-2020_10_19_00_48_30-cvat for images 1.1',
     './data/task_13_eos 8-3-2020_10_19_00_49_43-cvat for images 1.1',
     './data/task_14_eos 9-1-2020_10_19_00_51_17-cvat for images 1.1',
     './data/task_15_eos 9-2-2020_10_19_02_01_15-cvat for images 1.1',
     './data/task_16_eos 9-3-2020_10_19_01_01_10-cvat for images 1.1',    
     './data/task_20_eos 12-1-2020_10_19_01_38_42-cvat for images 1.1',
]

VALID_BASE = [
     './data/task_17_eos 10-1-2020_10_19_01_52_10-cvat for images 1.1',
     './data/task_18_eos 10-2-2020_10_19_01_07_45-cvat for images 1.1',    
]

TEST_BASE = [
    './data/task_19_eos 11-1-2020_10_19_01_45_57-cvat for images 1.1',
]

cols = ['phase', 'fn', 'key_points1', 'key_points2']
dataset_csv = pd.DataFrame()

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

dataset_csv.to_csv('./data/data_split.csv', index=False, sep=',')
train_case, valid_case, test_case = len(dataset_csv[dataset_csv['phase'] == 'train']), len(dataset_csv[dataset_csv['phase'] == 'valid']), len(dataset_csv[dataset_csv['phase'] == 'test'])
print(f'train_case : {train_case}, valid_case : {valid_case}, test_case : {test_case}') ## train_case : 892, valid_case : 139, test_case : 68