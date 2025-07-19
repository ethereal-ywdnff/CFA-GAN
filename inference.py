# inference.py
import os, torch
from PIL import Image
from torchvision import transforms
from model import CFA_GAN

# 📂 设置路径
ckpt_path = 'checkpoints/cfa_epoch29.pth'  # 选择你要加载的检查点
img_dir = 'data/images'
save_dir = 'outputs/aged'
os.makedirs(save_dir, exist_ok=True)

# 1️⃣ 加载模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CFA_GAN(num_ids=1).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# 2️⃣ 设置图像预处理
preprocess = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
denorm = transforms.Compose([
    transforms.Normalize([-1,-1,-1],[2,2,2]),
    transforms.Lambda(lambda t: torch.clamp(t, 0, 1)),
    transforms.ToPILImage()
])

# 3️⃣ 批量处理 images/ 所有图片
for fn in os.listdir(img_dir):
    if not fn.lower().endswith(('.png', '.jpg', '.jpeg')): continue
    img = Image.open(os.path.join(img_dir, fn)).convert('RGB')
    x = preprocess(img).unsqueeze(0).to(device)
    
    # 默认将所有人物统一变为例如 60 岁
    real_age = 30  # 可改成你图片真实年龄，或者跳过从csv读取
    target_age = torch.tensor([60.0], device=device)
    with torch.no_grad():
        fake, _, _ = model(x, target_age - real_age)
    
    # 4️⃣ 保存图像
    out = denorm(fake.squeeze().cpu())
    out.save(os.path.join(save_dir, f'{os.path.splitext(fn)[0]}_age60.png'))

print("✅ 生成完毕，变老图保存于", save_dir)
