import os
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
    UNet,
    L1_loss,
    L2_loss,
    mean_radial_error,
    MeanRadialError,
    str2bool,
)

import warnings
warnings.filterwarnings("ignore")

def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    save_folder = str(os.path.join(r'./checkpoint', datetime.now().strftime("%m%d%H%M%S")))
    base_path = './data/resized_scale5'    
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=device)

    parser.add_argument('--model', type=str, default='custom')
    parser.add_argument('--base_path', type=str, default=base_path)
    parser.add_argument('--save_folder', type=str, default=save_folder)

    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--T0', type=int, default=50)
    
    parser.add_argument('--num_kp', type=int, default=30)
    parser.add_argument('--comments', type=str, default=None)

    args = parser.parse_args()
    
    print('=' * 50)
    print('[info msg] arguments')
    for key, value in vars(args).items():
        print(key, ":", value)
    print('=' * 50)  

    assert os.path.isdir(args.base_path), f'wrong path({args.base_path})'

    label_df = pd.read_csv(r'./data/data_split.csv')

    train_df = label_df[label_df['phase']=='train']
    valid_df = label_df[label_df['phase']=='valid']

    train_dataset = KeypointDataset(
        base_path=args.base_path,
        label_df=train_df,
        num_kp=args.num_kp,
        transforms=get_train_transforms(),
    )

    valid_dataset = KeypointDataset(
        base_path=args.base_path,
        label_df=valid_df,
        num_kp=args.num_kp,
        transforms=get_valid_transforms(),
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=4, ##workstation
        # num_workers=2, ##server
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=4, ##workstation
        # num_workers=2, ##server
    )    

    if args.model == 'custom':
        model = UNet(
            in_channels=1,
            n_filters=32,
            out_channels=args.num_kp,
        )
        print('custom_unet is created')
    else:
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            encoder_depth=5,
            decoder_attention_type='scse',
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=args.num_kp,                      # model output channels (number of classes in your dataset)
            activation='softmax',
        )
        print('smp.Unet is created')
    
    model.to(args.device)        
    loss = [L1_loss, L2_loss, MeanRadialError(device=args.device, scale=1e-4)]
    metric = mean_radial_error
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T0)

    trainer = ModelTrainer(
        model=model,
        train_loader=train_data_loader,
        valid_loader=valid_data_loader,
        loss_func=loss,
        metric_func=metric,
        optimizer=optimizer,
        device=args.device,
        save_dir=args.save_folder,
        mode='min', 
        scheduler=scheduler, 
        num_epochs=args.epochs,
        snapshot_period=args.T0,
        use_wandb=False,
    )

    if trainer.use_wandb:
        trainer.initWandb(
            project_name='koamex',
            run_name=args.comments,
            args=args,
        )

    trainer.train()
    
    with open(os.path.join(trainer.save_dir, 'config.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('{} : {}\n'.format(key, value)) 


if __name__ == '__main__':
    main()