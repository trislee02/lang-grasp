import os
import pytorch_lightning as pl
from .models.lseg_net import LSegNet
from .lgrasp_module import LGraspModule



def make_model(args):
    net = LSegNet(
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
    args.gpus = -1
    args.accelerator = "gpu"
    args.strategy = "ddp"
    args.benchmark = True
    args.sync_batchnorm = True
    args.max_epochs = args.epochs
    # args.default_root_dir = args.checkpoint_dir
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        monitor='val_accuracy',
        mode='max',
        save_on_train_epoch_end=False,
        verbose=True,
        save_top_k=3,
    )

    args.callbacks = [checkpoint_callback]

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    trainer = pl.Trainer.from_argparse_args(args)
    return trainer