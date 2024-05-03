from models import LGraspModule
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

module = LGraspModule()

img_path = "data/grasp-anything/seen/image/0a5bd779e492513880bef534543ff031b169a045ed7ac809c5600c3268038f4d.jpg"
image = Image.open(img_path)
image = np.array(image)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
image = transform(image).unsqueeze(0)
image = image.cuda()
text_features, image_features = module((image, ('grasp the mug at the handle')))

fig, ax = plt.subplots(nrows=1, ncols=1)
img = image_features[0][0].detach().cpu().numpy()
img = (img - img.min()) / (img.max() - img.min())
ax.imshow(img)

# for r, row in enumerate(ax):
#     for c, col in enumerate(row):
#         img = image_features[0][r*len(row) + c].detach().cpu().numpy()
#         img = (img - img.min()) / (img.max() - img.min())
#         col.imshow(img, cmap='gray')
plt.savefig(f"logits.png")