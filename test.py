import torch
import torch.nn as nn
import clip

a = torch.tensor([[[1, 2, 3],
                   [4, 5, 6]],
                  [[7, 8, 9],
                   [10, 11, 12]]])

b = torch.tensor([[[2, 2, 2]],
                  [[3, 3, 3]]])

c = a @ b.mT
print(c)

# pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False)
# texts = clip.tokenize(["a photo of a cat", "a photo of a dog"])
# texts = texts.cuda()
# a = pretrained.encode_text(texts)

# print(a)
# print(a.shape)
# print("===================")
# b = a.unsqueeze(1)
# c = t.half() @ b.mT
# print(c)
# print(c.shape)
