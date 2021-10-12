import os
import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
import numpy as np

import src.vit.utils as utils
import src.vit.vision_transformer as vits
from src.dataloader.CamelyonDataset import CamelyonDataset
import csv
import pandas as pd

# ============ preparing data ... ============
train_transform = pth_transforms.Compose([
    pth_transforms.ToTensor(),
])
val_transform = pth_transforms.Compose([
    pth_transforms.ToTensor(),
])
dataset_train = CamelyonDataset('/media/gokberk/Disk/WSIs/Camelyon16/processed/mag20', transform=train_transform, is_training=True, max_bag_size=256, get_all_images=True)
dataset_val = CamelyonDataset('/media/gokberk/Disk/WSIs/Camelyon16/processed/mag20', transform=val_transform, is_training=False, max_bag_size=256, get_all_images=True)

train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=1,
    num_workers=0,
    pin_memory=True,
)
val_loader = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=1,
    num_workers=0,
    pin_memory=True,
)
print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")
print()


model = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
embed_dim = model.embed_dim * 5
model.cuda()
model.eval()
utils.load_pretrained_weights(model, 'pretrained_weights/vitb16.pth', 'teacher', 'vit_base', 16)

output_path = '/media/gokberk/Disk/WSIs/Camelyon16/embeddings'
camelyon_csv_path = os.path.join(output_path, 'Camelyon16.csv')
normal_csv = os.path.join(output_path, '0-normal.csv')
tumor_csv = os.path.join(output_path, '1-tumor.csv')

if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(os.path.join(output_path, '0-normal')):
    os.makedirs(os.path.join(output_path, '0-normal'))
if not os.path.exists(os.path.join(output_path, '1-tumor')):
    os.makedirs(os.path.join(output_path, '1-tumor'))
    
camelyon_csv = open(camelyon_csv_path, 'w')
writer_camelyon = csv.writer(camelyon_csv)

normal_csv = open(normal_csv, 'w')
writer_normal = csv.writer(camelyon_csv)

tumor_csv = open(tumor_csv, 'w')
writer_tumor = csv.writer(camelyon_csv)
import time

for (inp, target, slide_name) in train_loader:
    start = time.time()
    inp = torch.squeeze(inp)
    print(inp.shape)
    print(target)
    print(slide_name)
    folder = '0-normal' if target == 0 else '1-tumor'
    writer_camelyon.writerow([os.path.join('embeddings', folder, slide_name[0] + '.csv'), target.item()])
    if folder == '0-normal':
        writer_normal.writerow([os.path.join('embeddings', folder, slide_name[0] + '.csv'), target.item()])
        normal_csv.flush()
    else:
        writer_tumor.writerow([os.path.join('embeddings', folder, slide_name[0] + '.csv'), target.item()])
        tumor_csv.flush()
    camelyon_csv.flush()
    for i in range(0, inp.shape[0], 64):
        if i + 64 > inp.shape[0]:
            images = inp[i::].cuda()
        else:
            images = inp[i:i+64].cuda()
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(images, 4)
            output = [x[:, 0] for x in intermediate_output]
            output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
            output = torch.cat(output, dim=-1)
        output = output.detach().cpu().numpy()
        pd.DataFrame(output).to_csv(os.path.join(output_path, folder, slide_name[0] + '.csv'), index=None, header=None, mode='a')
    print(f'Time: {time.time() - start}')
        
for (inp, target, slide_name) in val_loader:
    start = time.time()
    inp = torch.squeeze(inp)
    print(inp.shape)
    print(target)
    print(slide_name)
    folder = '0-normal' if target == 0 else '1-tumor'
    writer_camelyon.writerow([os.path.join('embeddings', folder, slide_name[0] + '.csv'), target.item()])
    if folder == '0-normal':
        writer_normal.writerow([os.path.join('embeddings', folder, slide_name[0] + '.csv'), target.item()])
        normal_csv.flush()
    else:
        writer_tumor.writerow([os.path.join('embeddings', folder, slide_name[0] + '.csv'), target.item()])
        tumor_csv.flush()
    camelyon_csv.flush()
    for i in range(0, inp.shape[0], 64):
        if i + 64 > inp.shape[0]:
            images = inp[i::].cuda()
        else:
            images = inp[i:i+64].cuda()
    print(f'Time: {time.time() - start}')
        
camelyon_csv.close()
normal_csv.close()
tumor_csv.close()
    
