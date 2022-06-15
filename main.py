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
    UNet,
    L1_loss,
    L2_loss,
    mean_radial_error,
    MeanRadialError,
    str2bool,
)

def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    save_folder = str(os.path.join(r'./checkpoint', datetime.now().strftime("%m%d%H%M%S")))
    base_path = './data/resized_scale5'
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default=device) 

    parser.add_argument('--model', type=str, default='custom')
    parser.add_argument('--base_path', type=str, default=base_path)
    parser.add_argument('--save_folder', type=str, default=save_folder)

    parser.add_argument('--use_mre_as_loss', type=str2bool, default=False)
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
        base_path=args.base_path,
        label_df=train_df,
        transforms=get_train_transforms(),
    )

    valid_dataset = KeypointDataset(
        base_path=args.base_path,
        label_df=valid_df,
        transforms=get_valid_transforms(),
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=4, ##workstation
        num_workers=2, ##server
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=4, ##workstation
        num_workers=2, ##server
    )    

    if args.model == 'custom':
        model = UNet(
            in_channels=1,
            n_filters=32,
            out_channels=30,
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
            classes=30,                      # model output channels (number of classes in your dataset)
            activation='softmax',
        )
        print('smp.Unet is created')
    model.to(args.device)

    # loss = torch.nn.CrossEntropyLoss()
    # loss = [L1_loss, L2_loss]
    if args.use_mre_as_loss:
        mre = MeanRadialError(device=args.device)
        loss = [L2_loss, mre]
        print('[info msg] mre is used')
    else:    
        loss = [L2_loss]
        print('[info msg] mre is not used')

    metric = mean_radial_error
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.2) #learning rate decay

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
        use_wandb=True,
    )

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