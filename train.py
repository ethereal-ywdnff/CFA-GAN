# train.py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FaceAgingDataset
from model import CFA_GAN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = FaceAgingDataset()
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

model = CFA_GAN(num_ids=1).to(device)
opt = optim.Adam(model.parameters(), lr=1e-4)
l_age = nn.MSELoss()
l_id = nn.CrossEntropyLoss()

for epoch in range(30):
    for img, age, pid in loader:
        img, age, pid = img.to(device), age.to(device), pid.to(device)
        target_age = age + (torch.rand_like(age)*20-10).to(device)
        fake, zage, zid = model(img, target_age)
        # age regression
        age_pred = model.age_reg(zage)
        # ID classification
        id_pred = model.id_cls(zid)
        loss = l_age(age_pred.squeeze(), target_age) + l_id(id_pred, pid)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"Epoch {epoch} loss {loss.item():.4f}")
    torch.save(model.state_dict(), f"checkpoints/cfa_epoch{epoch}.pth")
