# model.py
import torch, torch.nn as nn, torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,4,2,1), nn.ReLU(),
            nn.Conv2d(64,128,4,2,1), nn.ReLU(),
            nn.Conv2d(128,256,4,2,1), nn.ReLU(),
        )
    def forward(self,x): return self.conv(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(257,128,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,3,4,2,1), nn.Tanh(),
        )
    def forward(self, z): return self.deconv(z)

class CFA_GAN(nn.Module):
    def __init__(self, num_ids=1):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()
        self.age_reg = nn.Sequential(nn.Linear(256,64), nn.ReLU(), nn.Linear(64,1))
        self.id_cls = nn.Sequential(nn.Linear(256,num_ids))
    def forward(self,x,target_age):
        z = self.enc(x)
        zf = z.mean([2,3])
        zid = F.normalize(zf,dim=1)
        zage = zf.norm(dim=1,keepdim=True)
        zcat = torch.cat([zid, zage + target_age.unsqueeze(1)], dim=1)
        zcat = zcat.unsqueeze(-1).unsqueeze(-1)
        out = self.dec(zcat)
        return out, zage, zid
