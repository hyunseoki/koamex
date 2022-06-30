import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import cv2
import os
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class KeypointDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, label_df, num_kp=30, transforms=None):
        self.base_path = base_path
        self.label_df = label_df
        self.transforms = transforms
        self.num_kp = num_kp

    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, index):
        df_row = self.label_df.iloc[index]

        task_n = df_row['task']
        img_fn = os.path.join(self.base_path, task_n, 'images', df_row['fn'])
        assert os.path.isfile(img_fn), f'wrong img_fn ({img_fn})'
        mask_fns = [os.path.join(self.base_path, task_n, 'mask', df_row['fn'].split('.')[0], f'{idx}.png') for idx in range(self.num_kp)]
        
        image = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
        mask = [cv2.imread(mask_fn, cv2.IMREAD_GRAYSCALE) for mask_fn in mask_fns]

        assert type(image) == np.ndarray, img_fn
        assert type(mask[0]) == np.ndarray
     
        if self.transforms != None:
            mask = [m for m in mask]
            transformed = self.transforms(image=image, masks=mask)
            image, mask = transformed['image'], transformed['masks']

        image = image / 255.0

        for i in range(self.num_kp):
            # mask[i] = np.power(mask[i], 8)
            if mask[i].max() == 0:
                print(os.path.basename(img_fn))
            mask[i] = mask[i] / mask[i].max()
        
        mask = np.array(mask, dtype=np.float32)
        
        if type(image) == torch.Tensor:
            mask = torch.tensor(mask, dtype=torch.float32)

        sample = dict()
        
        sample['id'] = os.path.join(task_n, os.path.basename(img_fn))
        sample['input'] = image
        sample['target'] = mask

        return sample


def get_train_transforms():
    return A.Compose(
        [
            A.RandomBrightnessContrast(p=0.3),
            A.Affine(
                scale=(0.9, 1.1),
                rotate=(5),
                translate_percent=(0.05, 0.05),
                cval=0,
                cval_mask=0,
                p=0.3,
            ),
            ToTensorV2(p=1.0),
        ],
    )


def get_valid_transforms():
    return A.Compose(
        [
            ToTensorV2(p=1.0),
        ],
    )


def get_test_transforms():
    return A.Compose(
        [
            ToTensorV2(p=1.0),
        ],
    )


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt    
    import copy
    
    def draw_keypoint(img, keypoints, scale=1):
        dst = copy.copy(img)

        for idx, point in enumerate(keypoints):
            cv2.circle(
                img=dst,
                center=tuple((int(point[1]), int(point[0]))),
                radius=int(20 * scale),
                color=(255,0,0),
                thickness=int(30 * scale),
                lineType=cv2.LINE_AA,    
            )
            
            cv2.putText(
                img=dst,
                text=str(idx),
                org=tuple((int(point[1]), int(point[0]))),
                fontFace=0,
                fontScale=int(5*scale),
                thickness=int(10*scale),
                color=(0,255,0),        
            )

        return dst

    label_df = pd.read_csv(r'F:\hyunseoki\koamex\data\data_split.csv')[:1]
    train_df = label_df[label_df['phase']=='train']

    dataset = KeypointDataset(
        base_path=r'F:\hyunseoki\koamex\data\resized_scale5',
        label_df=train_df,
        transforms=get_train_transforms(),
        # transforms=None,
    )
    
    sample = dataset[0]
    image = cv2.cvtColor(sample['input'].squeeze().numpy(), cv2.COLOR_GRAY2BGR)
    mask = cv2.cvtColor(sample['target'][0].squeeze().numpy(), cv2.COLOR_GRAY2BGR)
    
    plt.subplot(1,2,1)
    plt.imshow(image, 'gray')
    plt.subplot(1,2,2)
    plt.imshow(mask, 'gray')
    plt.show()
    

    # keypoints = list()
    # for idx in range(30):
    #     m = mask[idx]
    #     idx = (m==torch.max(m)).nonzero().numpy()[0]
    #     keypoints.append(idx)

    # mask = draw_keypoint(
    #     image,
    #     keypoints,
    #     scale=0.2,
    # )

    # plt.imshow(mask)
    # plt.show()

    # image, targets = dataset[0]

    # for k, v in targets.items():
    #     print(f'k: {k}, v: {v}')
    
    # model = get_model()
    # model([image], [targets])

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
