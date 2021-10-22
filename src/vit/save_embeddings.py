import os
import numpy as np
import pandas as pd

import torch
from torchvision import transforms as pth_transforms

import utils
import vision_transformer as vits
from PIL import Image

class CustomDataSet(torch.utils.data.dataset.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = [dirr for dirr in os.listdir(main_dir) if dirr != 'metadata.txt']

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

train_transform = pth_transforms.Compose([
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.6223, 0.4763, 0.6009), (0.2012, 0.2190, 0.1722)),
])

model = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
embed_dim = model.embed_dim * 5
model.cuda()
model.eval()
utils.load_pretrained_weights(model, '/home/ag23peby/self-learning-transformer-mil/checkpoint.pth', 'teacher', 'vit_base', 16)

output_path = '/work/scratch/ag23peby/embeddings'
camelyon_csv_path = os.path.join(output_path, 'Camelyon16.csv')
normal_csv = os.path.join(output_path, '0-normal.csv')
tumor_csv = os.path.join(output_path, '1-tumor.csv')

if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(os.path.join(output_path, '0-normal')):
    os.makedirs(os.path.join(output_path, '0-normal'))
if not os.path.exists(os.path.join(output_path, '1-tumor')):
    os.makedirs(os.path.join(output_path, '1-tumor'))

if os.path.isfile(camelyon_csv_path):
    camelyon_csv = pd.read_csv(camelyon_csv_path, header=None)
    already_processed = [row.split('/')[-1].split('.')[0] for row in camelyon_csv[0]]
else:
    already_processed = []

slides = []
is_training = False
for root, _, files in os.walk('/work/scratch/ag23peby/processed/mag20', topdown=False):
    if any('metadata.txt' in file_name for file_name in files):
        if is_training and 'training' in root:
            slides.append(root)
        elif not is_training and 'test' in root:
            slides.append(root)

reference_csv = pd.read_csv('/work/scratch/ag23peby/processed/mag20/testing/reference.csv', header=None)

"""
for slide in slides:
    my_dataset = CustomDataSet(slide, transform=train_transform)
    slide_id = os.path.basename(slide)
    slide_path = [row for row in camelyon_csv[0] if slide_id in row]
    if len(slide_path) > 1:
        print("Duplicate line in", slide_path[0])
    slide_csv = pd.read_csv(os.path.join('/media/gokberk/Disk/WSIs/Camelyon16', slide_path[0]), header=None)
    if len(slide_csv) == len(my_dataset):
        continue
    print("Problem at", slide_id)

print("Completed")
"""

for slide in slides:
    slide_id = os.path.basename(slide)
    if slide_id in already_processed:
        continue
    if is_training:
        target = 0 if 'normal' in slide_id else 1
    else:
        target = 0 if 'Normal' in reference_csv.loc[reference_csv[0] == slide_id][1].item() else 1
    folder = '0-normal' if target == 0 else '1-tumor'
    pd.DataFrame([[os.path.join('embeddings', folder, slide_id + '.csv'), target]]).to_csv(os.path.join(camelyon_csv_path), index=None, header=None, mode='a+')
    if folder == '0-normal':
        pd.DataFrame([[os.path.join('embeddings', folder, slide_id + '.csv'), target]]).to_csv(os.path.join(normal_csv), index=None, header=None, mode='a+')
    else:
        pd.DataFrame([[os.path.join('embeddings', folder, slide_id + '.csv'), target]]).to_csv(os.path.join(tumor_csv), index=None, header=None, mode='a+')
    my_dataset = CustomDataSet(slide, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=256,
        num_workers=0,
        pin_memory=True,
        shuffle=False
    )
    print(target)
    print(slide_id)
    print(len(my_dataset))
    for inp in train_loader:
        inp = inp.cuda()
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(inp, 4)
            output = [x[:, 0] for x in intermediate_output]
            output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
            output = torch.cat(output, dim=-1)
        output = output.detach().cpu().numpy()
        pd.DataFrame(output).to_csv(os.path.join(output_path, folder, slide_id + '.csv'), index=None, header=None, mode='a')
