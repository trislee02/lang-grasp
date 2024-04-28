import pytorch_lightning as pl

from models.lgrasp import LGraspModule

model = LGraspModule.load_from_checkpoint("checkpoints/model-epoch=01-val_loss=0.64.ckpt")
print(model)
model.eval()
