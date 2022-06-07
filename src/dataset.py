import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import cv2
import os
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def keypoint2box(keypoint):
    x1, y1 = min(keypoint[:, 0]), min(keypoint[:, 1])
    x2, y2 = max(keypoint[:, 0]), max(keypoint[:, 1])

    return [x1, y1, x2, y2]


class KeypointDataset(torch.utils.data.Dataset):
    def __init__(self, label_df, transforms=None, phase='train'):
        self.label_df = label_df
        self.transforms = transforms
        self.phase = phase

    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, index):
        img_fn = self.label_df.iloc[index]['fn']

        assert os.path.isfile(img_fn), img_fn
        # image = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(img_fn)

        keypoints1 = self.label_df.iloc[index]['keypoints1'].split(';')
        keypoints1 = np.array([[float(p.split(',')[0]), float(p.split(',')[1])] for p in keypoints1], dtype=np.int64)    
            
        box1 = keypoint2box(keypoints1)
        keypoints1 = keypoints1.tolist()   
        bboxes = [box1]

        # keypoints2 = self.label_df.iloc[index]['keypoints2'].split(';')
        # keypoints2 = np.array([[float(p.split(',')[0]), float(p.split(',')[1])] for p in keypoints2], dtype=np.int64)

        # x3, y3 = min(keypoints2[:, 0]), min(keypoints2[:, 1])
        # x4, y4 = max(keypoints2[:, 0]), max(keypoints2[:, 1])        

        # boxes = np.array([[x1, y1, x2, y2], [x3, y3, x4, y4]], dtype=np.int64)
        
        # keypoints2 = keypoints2.tolist()
        
        # labels = ['keypoints1', 'keypoints2']
        # keypoints1.append('keypoints1')
        # keypoints2.append('keypoints2')
        # keypoints = [keypoints1, keypoints2]

        # bboxes = [
        #     [x1, y1, x2, y2, 'keypoints1'],
        #     [x3, y3, x4, y4, 'keypoints2'],
        # ]
        
        labels = np.array([1])
        # labels= ['keypoints1']
        targets ={
            'image': image,
            'bboxes': bboxes,
            'labels': labels,
            'keypoints': keypoints1,
        }


        # targets ={
        #     'image': image,
        #     'bboxes': boxes,
        #     'labels': labels,
        #     'keypoints': keypoints1
        # }

        if self.transforms is not None:
            targets = self.transforms(**targets)

        image = targets['image']
        image = image / 255.0

        targets = {
            'labels': torch.as_tensor(targets['labels'], dtype=torch.int64),
            'boxes': torch.as_tensor(targets['bboxes'], dtype=torch.float32),
            'keypoints': torch.as_tensor(
                np.concatenate([targets['keypoints'], np.ones((30, 1))], axis=1)[np.newaxis], dtype=torch.float32
            )
        }

        return image, targets


def draw(img, points, color=(255,0,0), bboxes=None, keypoint_names=None):
    for idx, point in enumerate(points):      
        img = cv2.circle(
            img=img,
            center=point,
            radius=25,
            color=color,
            thickness=15,
            lineType=cv2.LINE_AA,    
        )
    
        if keypoint_names != None:
            cv2.putText(
                img=img,
    #                 text=f'{idx}: {keypoint_names[idx]}', 
                text=f'{idx}', 
                org=tuple(point), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=3.0,
                color=(255, 255, 0),
                thickness=3
            )

    for idx, bbox in enumerate(bboxes):
        p0 = (int(bbox[0]), int(bbox[1]))
        p1 = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(image, p0, p1, (255, 0, 0), thickness=3)

    return img


def get_train_transforms():
    return A.Compose(
        [
            ToTensorV2(p=1.0),
            # A.Rotate(limit=30, always_apply=True)
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=True),
    )


def get_valid_transforms():
    return A.Compose(
        [
            ToTensorV2(p=1.0),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=True),
    )


def get_test_transforms():
    return A.Compose(
        [
            ToTensorV2(p=1.0),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=True),
    )


def get_model():
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection import KeypointRCNN

    backbone = resnet_fpn_backbone(
        backbone_name='resnet50',
        pretrained=True,
    )

    model = KeypointRCNN(
        backbone,
        num_classes=1,
        num_keypoints=30,
    )

    model.train()

    return model


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    from util import (
        KEYPOINT1_NAMES, KEYPOINT2_NAMES
    )
    
    label_df = pd.read_csv(r'data\data_split.csv')
    train_df = label_df[label_df['phase']=='train']

    dataset = KeypointDataset(
        label_df=train_df,
        transforms=get_train_transforms(),
    )

    image, targets = dataset[0]

    for k, v in targets.items():
        print(f'k: {k}, v: {v}')
    
    model = get_model()
    model([image], [targets])

    # image =  image.detach().cpu().numpy().transpose(1, 2, 0)
    # keypoints = targets['keypoints'].detach().cpu().numpy()[0][:, :2].astype(np.int64).tolist()
    # boxes = targets['boxes'].detach().cpu().numpy().astype(np.int64).tolist()
 
    # image = draw(
    #     img=image,
    #     points=keypoints,
    #     color=(0, 255, 0),
    #     bboxes=boxes,
    #     keypoint_names=KEYPOINT1_NAMES
    # )

    # plt.imshow(image, 'gray')
    # plt.show()
