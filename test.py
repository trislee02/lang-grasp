import torch
from models.lgrasp.models.lseg_net_test import LSegNet
import torch.cuda.amp as amp
from torchviz import make_dot, make_dot_from_trace

net = LSegNet(labels=['a', 'b'], arch_option=1, activation='lrelu', block_depth=1, backbone='clip_vitl16_384', num_features=256).cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

x = torch.rand(3, 3, 224, 224).cuda()
prompts = ['a', 'b', 'c']

scaler = amp.GradScaler(enabled=False)

for i in range(10):
    # pos_pred, cos_pred, sin_pred, width_pred = net(x)
    
    pos_target = torch.rand(3, 1, 224, 224).cuda()
    cos_target = torch.rand(3, 1, 224, 224).cuda()
    sin_target = torch.rand(3, 1, 224, 224).cuda()
    width_target = torch.rand(3, 1, 224, 224).cuda()

    # loss = (torch.sum(pos_pred - pos_target))**2 + (torch.sum(cos_pred - cos_target))**2 + (torch.sum(sin_pred - sin_target))**2 + (torch.sum(width_pred - width_target))**2
    optimizer.zero_grad()

    with amp.autocast(enabled=False):
        y_pred = net(x)
        loss = torch.sum(y_pred - pos_target) ** 2 
        loss = scaler.scale(loss)
    
    print("Batch: ", i, " Loss: ", loss.item())
    loss.backward()
    optimizer.step()




