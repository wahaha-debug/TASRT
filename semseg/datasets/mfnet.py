import os
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from semseg.augmentations_mm_ori import get_train_augmentation

from matplotlib import pyplot as plt
import torchvision.transforms.functional as F
import json

class MFNet(Dataset):
    """
    num_classes: 5
    """
    CLASSES = ['Background', 'Hand-Drill', 'BackPack', 'Fire-Extinguisher', 'Survivor']

    PALETTE = torch.tensor([         [0, 0, 0],  
        [228, 228, 179], 
        [133, 57, 181], 
        [177, 162, 67], 
        [50, 178, 200], 
            ])

    def __init__(self, root: str = '../../data/PST900', split: str = 'train', transform = None, modals = ['img', 'thermal'], case = None,
                 use_text: bool = False,
                 rgb_text_json: str = None,
                 thermal_text_json: str = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.root = root
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.files = self._get_file_names(split)

        self.use_text = use_text
        if rgb_text_json is None:
            rgb_text_json = os.path.join(self.root, 'rgb_text.json')
        if thermal_text_json is None:
            thermal_text_json = os.path.join(self.root, 'thermal_text.json')

        def _load_json(path):
            try:
                if os.path.isfile(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception:
                pass
            return {}

        self.rgb_text_map = _load_json(rgb_text_json)
        self.thermal_text_map = _load_json(thermal_text_json)

        if not self.files:
            raise Exception(f"No images found under root '{self.root}' for split '{split}'")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        item_name = str(self.files[index])

        rgb_text = self.rgb_text_map.get(item_name, "")
        thermal_text = self.thermal_text_map.get(item_name, "")
        
        rgb = os.path.join(*[self.root, 'rgb', item_name + '.png'])  
        x1 = os.path.join(*[self.root, 'thermal', item_name + '.png'])  
        lbl_path = os.path.join(*[self.root, 'labels', item_name + '.png'])  

        sample = {}
        sample['img'] = io.read_image(rgb)[:3, ...]
        if 'thermal' in self.modals:
            sample['thermal'] = self._open_img(x1)

        label = io.read_image(lbl_path)[0, ...].unsqueeze(0)
        sample['mask'] = label

        if self.transform:
            try:
                sample = self.transform(sample)
            except Exception as e:
                raise RuntimeError(f"Augmentation failed for item '{item_name}'. RGB: {rgb}, TH: {x1}, MASK: {lbl_path}. Error: {e}")

        label = sample['mask']
        del sample['mask']

        label = self.encode(label.squeeze().numpy()).long()
        sample_list = [sample[k] for k in self.modals]

        if self.use_text:
            text_dict = {'rgb_text': rgb_text, 'thermal_text': thermal_text}
            return sample_list, label, text_dict
        return sample_list, label

    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def encode(self, label: Tensor) -> Tensor:
        return torch.from_numpy(label)

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = os.path.join(self.root, 'test.txt') if split_name == 'val' else os.path.join(self.root, 'train.txt')
        file_names = []
        with open(source) as f:
            files = f.readlines()
        for item in files:
            file_name = item.strip()
            if ' ' in file_name:
                file_name = file_name.split(' ')[0]
            file_names.append(file_name)
        return file_names


if __name__ == '__main__':
    traintransform = get_train_augmentation((480, 640), seg_fill=255)

