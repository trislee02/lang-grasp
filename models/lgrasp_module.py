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

        plot_img = [wandb.Image(x[0][0], caption=x[1][0]), 
                    wandb.Image(self.model.out_logits_per_image[0], caption=f"{x[1][0]}-logits")]       
        plot_img.extend([wandb.Image(self.model.out_image_features[0][i], caption=f"{x[1][0]}-image features channel {i}") for i in range(10)])
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
            {"params": self.model.pretrained.parameters(), "lr": self.base_lr},
            {"params": self.model.lseg.pretrained.parameters(), "lr": self.base_lr},
            {"params": self.model.lseg.scratch.parameters(), "lr": self.base_lr * 10},
        ]
        if hasattr(self.model, "scratch"):
            print("Found output scratch")
            params_list.append(
                {"params": self.model.scratch.parameters(), "lr": self.base_lr * 10}
            )
        if hasattr(self.model, "grcnn"):
            print("Found grcnn")
            params_list.append(
                {"params": self.model.grcnn.parameters(), "lr": self.base_lr}
            )
        if hasattr(self.model, "auxlayer"):
            print("Found auxlayer")
            params_list.append(
                {"params": self.model.auxlayer.parameters(), "lr": self.base_lr * 10}
            )
        if hasattr(self.model, "scale_inv_conv"):
            print(self.model.scale_inv_conv)
            print("Found scaleinv layers")
            params_list.append(
                {
                    "params": self.model.scale_inv_conv.parameters(),
                    "lr": self.base_lr * 10,
                }
            )
            params_list.append(
                {"params": self.model.scale2_conv.parameters(), "lr": self.base_lr * 10}
            )
            params_list.append(
                {"params": self.model.scale3_conv.parameters(), "lr": self.base_lr * 10}
            )
            params_list.append(
                {"params": self.model.scale4_conv.parameters(), "lr": self.base_lr * 10}
            )

        opt = torch.optim.Adam(
                params_list,
                lr=self.base_lr,
                weight_decay=self.weight_decay,
        )
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda x: pow(1.0 - x / self.epochs, 0.9), verbose=True
        )

        return [opt], [sch]