import torch
import torch.cuda.amp as amp
import pytorch_lightning as pl
from inference.post_process import post_process_output
from utils.dataset_processing import evaluation

class LGraspModule(pl.LightningModule):
    def __init__(self, 
                 model, 
                 loss_fn, 
                 dataset,
                 max_epochs=100, 
                 base_lr=4e-3):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.base_lr = base_lr
        self.dataset = dataset
        
        self.epochs = max_epochs
        self.enabled = False #True mixed precision will make things complicated and leading to NAN error
        self.scaler = amp.GradScaler(enabled=self.enabled)

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
        return loss

    def training_epoch_end(self, outs):
        # TODO: Add metric calculation
        print("Epoch end", outs)

    def validation_step(self, batch, batch_idx):
        x, y, didx, rot, zoom_factor = batch
        xc = (x[0], x[1]) # x[0] is a Tensor [batch_size, c, h, w], x[1] is a Tuple of `batch_size`` prompts
        yc = [yy for yy in y]
        lossd = self.loss_fn(xc, yc)
        loss = lossd['loss']

        # results = {
        #             'correct': 0,
        #             'failed': 0,
        #             'loss': 0,
        #             'losses': {}
        #         }

        # q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
        #                                                 lossd['pred']['sin'], lossd['pred']['width'])

        # s = evaluation.calculate_iou_match(q_out,
        #                                     ang_out,
        #                                     self.dataset.get_gtbb(didx, rot, zoom_factor),
        #                                     no_grasps=1,
        #                                     grasp_width=w_out,
        #                                     threshold=0.25
        #                                     )

        # if s:
        #     results['correct'] += 1
        # else:
        #     results['failed'] += 1

        self.log("val_loss", loss)
        return loss

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

        
        opt = torch.optim.SGD(
            params_list,
            lr=self.base_lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda x: pow(1.0 - x / self.epochs, 0.9)
        )

        return [opt], [sch]