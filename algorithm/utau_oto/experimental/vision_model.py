"""Lightweight heatmap CNN. No position or existing-estimator inputs."""
import torch
from torch import nn

class HeatmapCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.image=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d((2,1)),
            nn.Conv2d(16,24,3,padding=1),nn.ReLU(),nn.MaxPool2d((2,1)),nn.Conv2d(24,32,3,padding=1),nn.ReLU())
        self.temporal=nn.Sequential(nn.Conv1d(32,24,5,padding=2),nn.ReLU(),nn.Conv1d(24,1,1))
    def forward(self,x):return self.temporal(self.image(x).mean(dim=2)).squeeze(1)

class FrequencyContextCNN(nn.Module):
    """Retain coarse frequency location and compare ~150ms of context."""
    def __init__(self):
        super().__init__()
        self.image=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d((2,1)),
            nn.Conv2d(16,24,3,padding=1),nn.ReLU(),nn.MaxPool2d((2,1)),
            nn.Conv2d(24,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((8,None)))
        self.temporal=nn.Sequential(nn.Conv1d(256,64,1),nn.ReLU(),
            nn.Conv1d(64,64,5,padding=2),nn.ReLU(),
            nn.Conv1d(64,48,5,padding=4,dilation=2),nn.ReLU(),
            nn.Conv1d(48,32,5,padding=8,dilation=4),nn.ReLU(),nn.Conv1d(32,1,1))
    def forward(self,x):
        z=self.image(x);return self.temporal(z.flatten(1,2)).squeeze(1)

def build_model(architecture='legacy'):
    if architecture=='legacy':return HeatmapCNN()
    if architecture=='frequency_context':return FrequencyContextCNN()
    raise ValueError('Unknown vision architecture: '+architecture)

def device_for(requested):
    if requested=='cuda' and not torch.cuda.is_available():
        from ..features import Backend
        if Backend('cuda').cupy is None:raise RuntimeError('CUDA unavailable')
        return 'cpu'
    return 'cuda' if requested!='cpu' and torch.cuda.is_available() else 'cpu'
