import argparse
import datetime
import json
import logging
import os
import sys

import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.utils.data
from torchsummary import summary

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow
from utils import count_parameters, parameters_grad

from models import make_model, make_trainer
from models import LGraspModule

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # Network
    parser.add_argument('--network', type=str, default='grconvnet3',
                        help='Network name in inference/models')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')
    # LGraspNet
    parser.add_argument('--backbone', type=str, default="clip_vitl16_384",
                        help="backbone network")
    parser.add_argument('--num-features', type=int, default=256,
                        help="number of features that go from encoder to decoder")
    parser.add_argument('--arch-option', type=int, default=0,
                        help="which kind of architecture to be used")
    parser.add_argument('--block-depth', type=int, default=0,
                        help="how many blocks should be used")
    parser.add_argument("--activation", choices=['lrelu', 'tanh'], default="lrelu",
                        help="use which activation to activate the block")


    # Datasets
    parser.add_argument('--dataset', type=str,
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # Training
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--base-lr', type=float, default=4e-3,
                        help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000,
                        help='Batches per Epoch')
    # parser.add_argument('--optim', type=str, default='adam',
    #                     help='Optmizer for the training. (adam or SGD)')
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="accumulate N batches for gradient computation")
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/',
                        help='Path to save model checkpoints')
    parser.add_argument('--overfit-check', action='store_true', default=False,
                        help='Overfit the network on a single batch')

    # Logging etc.
    parser.add_argument('--description', type=str, default='',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log directory')
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the training process')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')
    parser.add_argument('--seen', type=int, default=1,
                        help='Flag for using seen classes, only work for Grasp-Anything dataset') 


    # Resume training
    parser.add_argument('--resume-checkpoint', type=str, default='',
                        help='Path to model to resume training')

    args = parser.parse_args()
    return args

def run():
    args = parse_args()
   
    # Set-up output directories
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    dataset = Dataset(args.dataset_path,
                      output_size=args.input_size,
                      ds_rotate=args.ds_rotate,
                      random_rotate=False,
                      random_zoom=False,
                      include_depth=args.use_depth,
                      include_rgb=args.use_rgb,
                      seen=args.seen)
    logging.info('Dataset size is {}'.format(dataset.length))

    # Creating data indices for training and validation splits
    indices = list(range(dataset.length))
    split = int(np.floor(args.split * dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    logging.info('Training size: {}'.format(len(train_indices)))
    logging.info('Validation size: {}'.format(len(val_indices)))

    # Creating data samplers and loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    logging.info("Training batch size: {}".format(args.batch_size))
    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    logging.info('Done')

    # Debugging
    logging.info('Loading network...')
    net = make_model(args)

    # Set up Pytorch Lightning Module
    if args.resume_checkpoint == '':
        model = LGraspModule(dataset=dataset,
                             max_epochs=args.epochs,
                             base_lr=args.base_lr,
                             weight_decay=args.weight_decay,
                             backbone=args.backbone,
                             num_features=args.num_features,
                             arch_option=args.arch_option,
                             block_depth=args.block_depth,
                             activation=args.activation,)
    else:
        logging.info('Resuming training from checkpoint: {}'.format(args.resume_checkpoint))
        model = LGraspModule.load_from_checkpoint(args.resume_checkpoint)
        
    # Set up Pytorch Lightning Trainer
    trainer = make_trainer(args)

    # Fit model using train and val dataloader
    trainer.fit(model, train_data, val_data)

if __name__ == '__main__':
    run()
