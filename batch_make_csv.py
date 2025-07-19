# batch_make_csv.py
import glob, os, csv

# 📁 指定图像目录
img_dir = 'CFA-GAN/data/images'
# 📄 输出 CSV 路径
csv_path = 'CFA-GAN/data/data.csv'

# 获取所有 png/jpg 图片列表
files = glob.glob(os.path.join(img_dir, '*.[pj][pn]g'))

# 假定所有图片属于同一人物，person_id=0，可按需调整
person_id = 0
# 默认年龄，可手动修改或后续填充
default_age = 25

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'age', 'person_id'])
    for fp in sorted(files):
        fn = os.path.basename(fp)
        writer.writerow([fn, default_age, person_id])

print(f"✅ 已生成 {csv_path}，共 {len(files)} 条记录")
