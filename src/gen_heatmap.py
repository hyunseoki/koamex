import os
import numpy as np
from multiprocessing import Process
import numpy as np
from math import exp, ceil
import pandas as pd
from datetime import datetime
import albumentations as A
import cv2


def rescale(img, kp, target_height=640, target_width=480):
    transform = A.Compose(
        [A.Resize(height=target_height, width=target_width, interpolation=cv2.INTER_AREA, p=1),],
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)        
    )

    augmented = transform(image=img, keypoints=kp)
    transformed_img, transformed_kp = augmented['image'], augmented['keypoints']

    return transformed_img, transformed_kp


def gen_heatmap(p_x, p_y, h, w, scale=10.0): ##basic_scale=30
    scaledGaussian = lambda x : exp(-(1/2)*(x**2)) ## Rescale Gaussian Distribution Between 0 and 1

    gray_img = np.zeros((h,w), np.uint8)
    pos = np.array([int(p_x), int(p_y)])
    max_distance = np.linalg.norm([h,w]) / 2

    for i in range(h):
        for j in range(w):    
            distanceFromCenter = scale * np.linalg.norm(np.array([i,j]) - pos) / max_distance
            scaledGaussianProb = scaledGaussian(distanceFromCenter)
            gray_img[i,j] = np.clip(scaledGaussianProb * 255, 0, 255)

    return gray_img


def work(label_df, base_dir, dst_dir, target_height=640, target_width=480, scale=30.0):
    for file_idx in range(len(label_df)):
        start_time = datetime.now()
        
        df_row = label_df.iloc[file_idx]

        task_n = df_row['task']
        img_fn = os.path.join(base_dir, df_row['task'], 'images', df_row['fn'])

        img_save_dir = os.path.join(dst_dir, task_n, 'images')
        mask_save_dir = os.path.join(dst_dir, task_n, 'mask', os.path.basename(img_fn).split('.')[0])
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        if not os.path.exists(mask_save_dir):
            os.makedirs(mask_save_dir)

        img_save_fn = os.path.join(img_save_dir, os.path.basename(df_row['fn']))
        mask_save_fn = os.path.basename(df_row['fn']).replace('png', 'npy')
        mask_save_fn = os.path.join(mask_save_dir, mask_save_fn)
        
        # if os.path.isfile(img_save_fn):
        #     print(f'[info msg] {img_save_fn} is existing, process skipped')
        #     continue

        criteria = int(df_row['img_h'] * 0.4)        
        img = cv2.imread(img_fn, 0)
        assert type(img) == np.ndarray, img_fn
        img = img[criteria:]

        keypoints = list()
        for keypoint_idx in range(30):
            keypoints.append([df_row[f'keypoint1_{keypoint_idx}_x'] - criteria, df_row[f'keypoint1_{keypoint_idx}_y']]) 
        for keypoint_idx in range(6):
            keypoints.append([df_row[f'keypoint2_{keypoint_idx}_x'] - criteria, df_row[f'keypoint2_{keypoint_idx}_y']]) 

        img, keypoints = rescale(img=img, kp=keypoints, target_height=target_height, target_width=target_width)
        cv2.imwrite(img_save_fn, img)
        
        mask_list = list()
        for idx, kp in enumerate(keypoints):
            mask = gen_heatmap(kp[0], kp[1], target_height, target_width, scale)
            mask_list.append(mask)
            cv2.imwrite(os.path.join(mask_save_dir, f'{idx}.png'), mask)       

        with open(mask_save_fn, 'wb') as f:
            np.save(f, np.array(mask_list))

        print(f'[info msg] {img_save_fn} is saved (time : {datetime.now() - start_time} is taken)')

if __name__ == '__main__':
    class Config():
        scale = 5 
        base_path = r'./data/raw_data'
        dst_path = rf'./data/resized_scale{scale}'
        label_fn = os.path.join(r'./data', 'data_split_small.csv')       

        assert os.path.isdir(base_path)
        assert os.path.isfile(label_fn)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

    label_df = pd.read_csv(Config.label_fn)

    # procs_nb = 1
    procs_nb = 18
    load_sz = ceil(len(label_df) / procs_nb)

    procs = list()
    
    startTime = datetime.now()
    for i in range(procs_nb):
        target_df = label_df.iloc[load_sz * i:load_sz * (i + 1)]
        proc = Process(target=work, args=(target_df, Config.base_path, Config.dst_path, 640, 480, Config.scale))       
        print(f'[info msg] process[{i}] is assigned to label_df[{load_sz * i}:{min(load_sz * (i + 1), len(label_df))}]')
        procs.append(proc)
        proc.start()
        
        
    for proc in procs:
        proc.join()

    print(f'[info msg] time : {datetime.now() - startTime} is taken')
    