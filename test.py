import pickle
import torch

pos_grasps = torch.load(".\data\grasp-anything\seen\grasp_label\\0a5bd779e492513880bef534543ff031b169a045ed7ac809c5600c3268038f4d_0_0.pt")
print(pos_grasps)

with open(".\data\grasp-anything\seen\grasp_instructions\\0a5bd779e492513880bef534543ff031b169a045ed7ac809c5600c3268038f4d_0_0.pkl", "rb") as file:
    prompt = pickle.load(file)
print(prompt)