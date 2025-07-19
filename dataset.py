# dataset.py
import os, torch, pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

class FaceAgingDataset(Dataset):
    def __init__(self, csv_path='data/data.csv', img_dir='data/images', transform=transform):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['filename'])).convert('RGB')
        img = self.transform(img)
        age = torch.tensor(float(row['age']))
        pid = torch.tensor(int(row['person_id']))
        return img, age, pid
