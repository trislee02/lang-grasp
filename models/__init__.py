import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from .lgrasp_net import LGraspNet
from .lgrasp_module import LGraspModule
import wandb


def make_model(args):
    net = LGraspNet(
            labels=[''],
            backbone=args.backbone,
            features=args.num_features,
            crop_size=224,
            arch_option=args.arch_option,
            block_depth=args.block_depth,
            activation=args.activation,
        )
    
    return net

def make_trainer(args):
    args.gpus = 1
    args.accelerator = "gpu"
    # args.strategy = "ddp" # Disable if there is a single GPU or there is not built-in NCCL support
    args.benchmark = True
    args.sync_batchnorm = True
    args.max_epochs = args.epochs

    args.gradient_clip_val=0.5

    wandb.login()

    args.logger = pl.loggers.WandbLogger(project='lgrasp')

    # Check overfit on small set of data
    if args.overfit_check:
        args.overfit_batches=0.01
    
    # args.default_root_dir = args.checkpoint_dir
    # acc_checkpoint = pl.callbacks.ModelCheckpoint(
    #     dirpath=args.checkpoint_dir,
    #     filename='model-{epoch:02d}-{val_accuracy:.2f}',
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_on_train_epoch_end=False,
    #     verbose=True,
    #     save_top_k=3,
    # )

    loss_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_on_train_epoch_end=False,
        verbose=True,
        save_top_k=1,
        save_last=True,
        save_weights_only=True
    )

    args.callbacks = [loss_checkpoint]

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    trainer = pl.Trainer.from_argparse_args(args)
    return trainer