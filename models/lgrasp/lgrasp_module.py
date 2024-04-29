import torch
import torch.cuda.amp as amp
import pytorch_lightning as pl
from utils.metrics import GraspAccuracy
from .models.lseg_net import LSegNet

class LGraspModule(pl.LightningModule):
    def __init__(self, 
                 dataset=None,
                 max_epochs=100, 
                 base_lr=4e-3,
                 backbone='clip_vitl16_384',
                 num_features=256,
                 arch_option=0,
                 block_depth=0,
                 activation='lrelu',):
        super().__init__()
        self.model = LSegNet(
            labels=[''],
            backbone=backbone,
            features=num_features,
            crop_size=224,
            arch_option=arch_option,
            block_depth=block_depth,
            activation=activation,
        )
        self.loss_fn = self.model.compute_loss
        self.base_lr = base_lr
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
        x, y, _, _, _ = batch
        xc = (x[0], x[1]) # x[0] is a Tensor [batch_size, c, h, w], x[1] is a Tuple of `batch_size`` prompts
        yc = [yy for yy in y]
        with amp.autocast(enabled=self.enabled):
            lossd = self.loss_fn(xc, yc)
            loss = lossd['loss']
        self.log("train_loss", loss)

        # Update the accuracy metric
        didx = didx.item()
        rot = rot.item()
        zoom_factor = zoom_factor.item()
        self.train_accuracy.update(lossd, didx, rot, zoom_factor)

        return loss

    def training_epoch_end(self, outs):
        # Log the accuracy metric
        self.log("val_accuracy", self.train_accuracy.accuracy()) 
        print("\nTraining accuracy: ", self.train_accuracy.accuracy())
        self.train_accuracy.reset()

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
        ]
        if hasattr(self.model, "scratch"):
            print("Found output scratch")
            params_list.append(
                {"params": self.model.scratch.parameters(), "lr": self.base_lr * 10}
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
                betas=(0.9, 0.999),
                weight_decay=1e-4,
        )
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda x: pow(1.0 - x / self.epochs, 0.9)
        )

        return [opt], [sch]