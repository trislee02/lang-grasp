import torch
import torch.cuda.amp as amp
import pytorch_lightning as pl
import wandb
from utils.metrics import GraspAccuracy
# from .lgrasp_net import LGraspNet
from .lgrasp_seg_net import LGraspNet
from pytorch_lightning.utilities import grad_norm
from inference.post_process import post_process_output


class LGraspModule(pl.LightningModule):
    def __init__(self, 
                 dataset=None,
                 max_epochs=100, 
                 base_lr=4e-3,
                 weight_decay=1e-4,
                 backbone='clip_vitl16_384',
                 num_features=256,
                 activation='lrelu',):
        super().__init__()
        self.model = LGraspNet(
            backbone=backbone,
            features=num_features,
            crop_size=224,
            activation=activation,
        )
        self.loss_fn = self.model.compute_loss
        self.base_lr = base_lr
        self.weight_decay = weight_decay

        if dataset:
            self.dataset = dataset
            self.val_accuracy = GraspAccuracy(dataset=dataset)
            self.train_accuracy = GraspAccuracy(dataset=dataset)
        
        self.epochs = max_epochs
        self.enabled = False #True mixed precision will make things complicated and leading to NAN error
        self.scaler = amp.GradScaler(enabled=self.enabled)

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, didx, rot, zoom_factor = batch
        xc = (x[0], x[1]) # x[0] is a Tensor [batch_size, c, h, w], x[1] is a Tuple of `batch_size`` prompts
        yc = [yy for yy in y]

        with amp.autocast(enabled=self.enabled):
            lossd = self.loss_fn(xc, yc)
            loss = lossd['loss']
        self.log("train_loss", loss)

        if batch_idx % 10 == 0:
            plot_img = [wandb.Image(x[0][0], caption=x[1][0]), 
                        wandb.Image(lossd['pred']['pos'][0][0], caption=f"{x[1][0]}-pred_pos"),
                        wandb.Image(lossd['pred']['cos'][0][0], caption=f"{x[1][0]}-pred_cos"),
                        wandb.Image(lossd['pred']['sin'][0][0], caption=f"{x[1][0]}-pred_sin"),
                        wandb.Image(lossd['pred']['width'][0][0], caption=f"{x[1][0]}-pred_width"),
                        wandb.Image(lossd['gt']['pos'][0][0], caption=f"{x[1][0]}-gt_pos"),
                        wandb.Image(lossd['gt']['cos'][0][0], caption=f"{x[1][0]}-gt_cos"),
                        wandb.Image(lossd['gt']['sin'][0][0], caption=f"{x[1][0]}-gt_sin"),
                        wandb.Image(lossd['gt']['width'][0][0], caption=f"{x[1][0]}-gt_width"),
                        wandb.Image(lossd['features']['logit'][0][0], caption=f"{x[1][0]}-logit"),
                        wandb.Image(lossd['features']['images_features'][0][0], caption=f"{x[1][0]}-images_features_channel_0"),
                        wandb.Image(lossd['features']['images_features'][0][1], caption=f"{x[1][0]}-images_features_channel_1"),
                        wandb.Image(lossd['features']['images_features'][0][2], caption=f"{x[1][0]}-images_features_channel_2"),
                        wandb.Image(lossd['features']['images_features'][0][3], caption=f"{x[1][0]}-images_features_channel_3"),
                        wandb.Image(lossd['features']['images_features'][0][4], caption=f"{x[1][0]}-images_features_channel_4"),
                        wandb.Image(lossd['features']['images_features'][0][5], caption=f"{x[1][0]}-images_features_channel_5"),
                        wandb.Image(lossd['features']['images_features'][0][6], caption=f"{x[1][0]}-images_features_channel_6"),
                        wandb.Image(lossd['features']['images_features'][0][7], caption=f"{x[1][0]}-images_features_channel_7"),
                        wandb.Image(lossd['features']['images_features'][0][8], caption=f"{x[1][0]}-images_features_channel_8"),
                        wandb.Image(lossd['features']['images_features'][0][9], caption=f"{x[1][0]}-images_features_channel_9")]
                
            wandb_logger = self.logger.experiment
            wandb_logger.log({"plot": plot_img})
                                 
        return loss

    def training_epoch_end(self, outs):
        pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass
        # print(self.model.scratch.head_block_pos_1.depthwise.depthwise.weight)
        # print(self.model.scratch.head_block_pos_2.depthwise.depthwise.weight)
        # print(self.model.scratch.head_block_pos_3.depthwise.depthwise.weight)

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # norms_grcnn = grad_norm(self.model.grcnn, norm_type=2)
        # norms_scratch = grad_norm(self.model.scratch, norm_type=2)
        # norms_pretrained = grad_norm(self.model.pretrained, norm_type=2)
        # print("Gradient norms: ", norms)
        # wandb_logger = self.logger.experiment
        # wandb_logger.log(norms_grcnn)
        # wandb_logger.log(norms_scratch)
        # wandb_logger.log(norms_pretrained)
        pass

    def validation_step(self, batch, batch_idx):
        x, y, didx, rot, zoom_factor = batch
        xc = (x[0], x[1]) # x[0] is a Tensor [batch_size, c, h, w], x[1] is a Tuple of `batch_size`` prompts
        yc = [yy for yy in y]
        lossd = self.loss_fn(xc, yc)
        loss = lossd['loss']
        self.log("val_loss", loss)

        # Update the accuracy metric
        didx = didx.item()
        rot = rot.item()
        zoom_factor = zoom_factor.item()
        self.val_accuracy.update(lossd, didx, rot, zoom_factor)

        return loss
    
    def validation_epoch_end(self, outs):
        # Log the accuracy metric
        self.log("val_accuracy", self.val_accuracy.accuracy()) 
        print("\nValidation accuracy: ", self.val_accuracy.accuracy())
        self.val_accuracy.reset()

    def configure_optimizers(self):
        params_list = [
            # {"params": self.model.lseg_net.pretrained.parameters(), "lr": self.base_lr},
            # {"params": self.model.lseg_net.scratch.parameters(), "lr": self.base_lr},
            {"params": self.model.srb.parameters(), "lr": self.base_lr},
            {"params": self.model.grcnn.parameters(), "lr": self.base_lr},
        ]

        opt = torch.optim.Adam(
                params_list,
                lr=self.base_lr,
                weight_decay=self.weight_decay,
        )
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda x: pow(1.0 - x / self.epochs, 0.9), # verbose=True
        )

        return [opt], [sch]