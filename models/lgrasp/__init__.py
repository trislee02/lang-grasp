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

    trainer = pl.Trainer.from_argparse_args(args)
    return trainer