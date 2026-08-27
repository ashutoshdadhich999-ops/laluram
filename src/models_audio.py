import torch
import torch.nn as nn
import snntorch as snn

class AdaptiveLIFCell(nn.Module):
    def __init__(self, beta=0.8):
        super().__init__()
        self.lif = snn.Leaky(beta=beta, init_hidden=True)

    def forward(self, x):
        return self.lif(x)

class StrongAudioNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8000, 256)
        self.lif1 = AdaptiveLIFCell(beta=0.85)
        self.fc2 = nn.Linear(256, 256)
        self.lif2 = AdaptiveLIFCell(beta=0.85)
        self.out = nn.Linear(256, 8000)
        self.t_embed = nn.Embedding(50, 256)

    def forward(self, x, t):
        t_emb = self.t_embed(t)
        h = torch.relu(self.fc1(x)) + t_emb
        h = self.lif1(h)
        h = torch.relu(self.fc2(h))
        h = self.lif2(h)
        return self.out(h)

class NonSpikeAudioNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8000, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 8000)
        self.t_embed = nn.Embedding(50, 256)

    def forward(self, x, t):
        t_emb = self.t_embed(t)
        h = torch.relu(self.fc1(x)) + t_emb
        h = torch.relu(self.fc2(h))
        return self.out(h)
