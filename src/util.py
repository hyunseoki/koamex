import os
import random
import numpy as np
import torch
import argparse
import copy
import cv2


KEYPOINT1_NAMES = {
    0: 'Rt_hip_center',
    1: 'Rt_femur_neck_center',
    
    2: 'Lt_hip_center',
    3: 'Lt_femur_neck_center', 

    4: 'Rt_upper_rect_1',
    5: 'Rt_upper_rect_2',
    6: 'Rt_upper_rect_3',
    7: 'Rt_upper_rect_4',
    
    8: 'Lt_upper_rect_1',
    9: 'Lt_upper_rect_2',
    10: 'Lt_upper_rect_3',
    11: 'Lt_upper_rect_4',
    
    12: 'Rt_proximal_tibia_lateral_plateau',
    13: 'Rt_center_of_tibial_spine',
    14: 'Rt_proximal_tibia_medial_plateau',
    
    15: 'Lt_proximal_tibia_lateral_plateau',
    16: 'Lt_center_of_tibial_spine',
    17: 'Lt_proximal_tibia_medial_plateau',

    18: 'Rt_lower_rec_1',
    19: 'Rt_lower_rec_2',
    20: 'Rt_lower_rec_3',
    21: 'Rt_lower_rec_4',
    
    22: 'Lt_lower_rec_1',
    23: 'Lt_lower_rec_2',
    24: 'Lt_lower_rec_3',
    25: 'Lt_lower_rec_4',
 
    26: 'Rt_talus_center_1',
    27: 'Rt_talus_center_2',
    
    28: 'Lt_talus_center_1',
    29: 'Lt_talus_center_2', 
}

KEYPOINT2_NAMES = {
    0: 'Rt_distal_lateral_femoral_condyle',
    1: 'Rt_distal_femoral_condyle_center',
    2: 'Rt_distal_medial_femoral_condyle',
    3: 'Lt_distal_lateral_femoral_condyle',
    4: 'Lt_distal_femoral_condyle_center',
    5: 'Lt_distal_medial_femoral_condyle',
}

def draw_keypoint(img,
                  keypoints,
                  keypoints_color=(255,0,0),
                  keypoints_scale=1,
                  text_color=(0,255,0),
                  text_scale=1,
                  text=False):
    dst = copy.copy(img)

    for idx, point in enumerate(keypoints):
        cv2.circle(
            img=dst,
            center=tuple((int(point[1]), int(point[0]))),
            radius=int(keypoints_scale),
            color=keypoints_color,
            thickness=int(keypoints_scale),
            lineType=cv2.LINE_AA,    
        )
        
        if text:
            if idx > 29:
                org = tuple((int(point[1]), int(point[0] + 15)))
            else:
                org = tuple((int(point[1]), int(point[0])))
            cv2.putText(
                img=dst,
                text=str(idx),
                org=org,
                fontFace=0,
                fontScale=text_scale,
                thickness=int(text_scale*2),
                color=text_color,        
            )

    return dst


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')