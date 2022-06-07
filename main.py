import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import pandas as pd
import torch
from datetime import datetime
from src import(
    KeypointDataset,
    seed_everything,
    get_train_transforms,
    get_valid_transforms,
    ModelTrainer,

)

def get_model():
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection import KeypointRCNN

    backbone = resnet_fpn_backbone('resnet50', pretrained=True)

    model = KeypointRCNN(
        backbone,
        num_classes=1,
        num_keypoints=30,
    )

    model.train()

    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    save_folder = str(os.path.join(r'./checkpoint', datetime.now().strftime("%m%d%H%M%S")))
    
    seed_everything(42)

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default=device)    

    parser.add_argument('--save_folder', type=str, default=save_folder)
    parser.add_argument('--comments', type=str, default=None)

    args = parser.parse_args()
    
    print('=' * 50)
    print('[info msg] arguments')
    for key, value in vars(args).items():
        print(key, ":", value)
    print('=' * 50)  

    label_df = pd.read_csv(r'data\data_split.csv')
    train_df = label_df[label_df['phase']=='train']
    valid_df = label_df[label_df['phase']=='valid']

    train_dataset = KeypointDataset(
        label_df=train_df,
        transforms=get_train_transforms(),
    )

    valid_dataset = KeypointDataset(
        label_df=valid_df,
        transforms=get_valid_transforms(),
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=4, ##workstation
        num_workers=2, ##server
        collate_fn=collate_fn,
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=4, ##workstation
        num_workers=2, ##server
        collate_fn=collate_fn,
    )

    model = get_model()
    model.to(args.device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2) #learning rate decay

    trainer = ModelTrainer(
        model=model,
        train_loader=train_data_loader,
        valid_loader=valid_data_loader,
        loss_func=loss,
        optimizer=optimizer,
        device=args.device,
        save_dir=args.save_folder,
        mode='max', 
        scheduler=scheduler, 
        num_epochs=args.epochs,
    )

    trainer.train()
    
    with open(os.path.join(trainer.save_dir, 'config.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('{} : {}\n'.format(key, value)) 


if __name__ == '__main__':
    main()