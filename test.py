import torch
import torch.nn as nn

sft = nn.functional.softmax 

def forward(StateVec,ConnectMatrix,L):
    StateVec = StateVec + (2*(-1/8*StateVec - ConnectMatrix.mm(StateVec)))*0.1
    pos = L.mm(sft(StateVec, dim=0))
    return StateVec, pos

N = 6

"""Toy target"""
target = torch.randn(2,20)

"""Randomly initialise L (which ought to be inferred later)"""
L = torch.randn(2,N)
L.requires_grad_(True)


"""Produce Connectivity Matrix rho"""
rho = torch.zeros(N,N);
for i in range(N):
    for j in range(N):
        if i == j:
            rho[i, j] = 0
        elif j == i + 1:
            rho[i, j] = 1.5
        elif j == i - 1:
            rho[i, j] = 0.5
        else:
            rho[i, j] = 1

rho[-1, 0] = 1.5
rho[0, -1] = 0.5

"""Initialise state vector as states = [0.5,0,0,...]"""
states = torch.Tensor(N,1)
states[0] = 0.5
states.requires_grad_(True)

lr = 0.1 # Learning Rate
for t in range(0, target.shape[1]):
    states, pos = forward(states,rho,L)
    loss = torch.sum((pos - target[:,t].float().view([2,1]))**2)
    loss.backward()
    L.data -= L.grad.data * lr
    print(f"Loss: {loss.item()}")
    L.grad.data.zero_()