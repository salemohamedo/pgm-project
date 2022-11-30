"""
PyTorch dataset classes for loading the datasets.
"""

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
import os
import json
import numpy as np
from collections import OrderedDict
from tqdm.auto import tqdm


class InterventionalPongDataset(data.Dataset):

    VAR_INFO = OrderedDict({
        'background': 'categ_2',
        'ball-vel-dir': 'angle',
        'ball-vel-magn': 'continuous_1',
        'ball-x': 'continuous_1',
        'ball-y': 'continuous_1',
        'paddle-left-y': 'continuous_1',
        'paddle-right-y': 'continuous_1',
        'score-left': 'categ_5',
        'score-right': 'categ_5'
    })

    def __init__(self, data_folder, split='train', single_image=False, return_latents=False, triplet=False, seq_len=2, causal_vars=None, **kwargs):
        super().__init__()
        filename = split
        if triplet:
            filename += '_triplets'
        self.split_name = filename
        data_file = os.path.join(data_folder, f'{filename}.npz')
        if split.startswith('val') and not os.path.isfile(data_file):
            self.split_name = self.split_name.replace('val', 'test')
            print('[!] WARNING: Could not find a validation dataset. Falling back to the standard test set. Do not use it for selecting the best model!')
            data_file = os.path.join(data_folder, f'{filename.replace("val", "test")}.npz')
        assert os.path.isfile(data_file), f'Could not find ComplexInterventionalPong dataset at {data_file}'

        arr = np.load(data_file)
        self.imgs = torch.from_numpy(arr['images'])
        self.latents = torch.from_numpy(arr['latents'])
        self.targets = torch.from_numpy(arr['targets'])
        self.keys = [key.replace('_', '-') for key in arr['keys'].tolist()]
        self._clean_up_data(causal_vars)

        self.single_image = single_image
        self.return_latents = return_latents
        self.triplet = triplet
        self.encodings_active = False
        self.seq_len = seq_len if not (single_image or triplet) else 1

    def _clean_up_data(self, causal_vars=None):
        if len(self.imgs.shape) == 5:
            self.imgs = self.imgs.permute(0, 1, 4, 2, 3)  # Push channels to PyTorch dimension
        else:
            self.imgs = self.imgs.permute(0, 3, 1, 2)

        all_latents, all_targets = [], []
        target_names = [] if causal_vars is None else causal_vars
        keys_var_info = list(InterventionalPongDataset.VAR_INFO.keys())
        for key in keys_var_info:
            if key not in self.keys:
                InterventionalPongDataset.VAR_INFO.pop(key)
        for i, key in enumerate(self.keys):
            if key.endswith('-proj'):
                continue
            latent = self.latents[...,i]
            target = self.targets[...,i]
            if key == 'ball-vel-magn' and latent.unique().shape[0] == 1:
                if key in InterventionalPongDataset.VAR_INFO:
                    InterventionalPongDataset.VAR_INFO.pop(key)
                continue
            if InterventionalPongDataset.VAR_INFO[key].startswith('continuous'):
                if key.endswith('-x') or key.endswith('-y'):
                    latent = latent / 16.0 - 1.0
                else:
                    latent = latent - 2.0
            if causal_vars is not None:
                if key in causal_vars:
                    all_targets.append(target)
            elif target.sum() > 0:
                all_targets.append(target)
                target_names.append(key)
            all_latents.append(latent)
        self.latents = torch.stack(all_latents, dim=-1)
        self.targets = torch.stack(all_targets, dim=-1)
        self.target_names_l = target_names
        print(f'Using the causal variables {self.target_names_l}')

    @torch.no_grad()
    def encode_dataset(self, encoder, batch_size=512):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        encoder.eval()
        encoder.to(device)
        encodings = None
        for idx in tqdm(range(0, self.imgs.shape[0], batch_size), desc='Encoding dataset...', leave=False):
            batch = self.imgs[idx:idx+batch_size].to(device)
            batch = self._prepare_imgs(batch)
            if len(batch.shape) == 5:
                batch = batch.flatten(0, 1)
            batch = encoder(batch)
            if len(self.imgs.shape) == 5:
                batch = batch.unflatten(0, (-1, self.imgs.shape[1]))
            batch = batch.detach().cpu()
            if encodings is None:
                encodings = torch.zeros(self.imgs.shape[:-3] + batch.shape[-1:], dtype=batch.dtype, device='cpu')
            encodings[idx:idx+batch_size] = batch
        self.imgs = encodings
        self.encodings_active = True
        return encodings

    def load_encodings(self, filename):
        self.imgs = torch.load(filename)
        self.encodings_active = True

    def _prepare_imgs(self, imgs):
        if self.encodings_active:
            return imgs
        else:
            imgs = imgs.float() / 255.0
            imgs = imgs * 2.0 - 1.0
            return imgs

    def label_to_img(self, label):
        return (label + 1.0) / 2.0

    def num_labels(self):
        return -1

    def num_vars(self):
        return self.targets.shape[-1]

    def target_names(self):
        return self.target_names_l

    def get_img_width(self):
        return self.imgs.shape[-2]

    def get_inp_channels(self):
        return self.imgs.shape[-3]

    def get_causal_var_info(self):
        return InterventionalPongDataset.VAR_INFO

    def __len__(self):
        return self.imgs.shape[0] - self.seq_len + 1

    def __getitem__(self, idx):
        returns = []

        if self.triplet:
            img_pair = self.imgs[idx]
            pos = self.latents[idx]
            target = self.targets[idx]
        else:
            img_pair = self.imgs[idx:idx+self.seq_len]
            pos = self.latents[idx:idx+self.seq_len]
            target = self.targets[idx:idx+self.seq_len-1]

        if self.single_image:
            img_pair = img_pair[0]
            pos = pos[0]
        else:
            returns += [target]
        img_pair = self._prepare_imgs(img_pair)
        returns = [img_pair] + returns

        if self.return_latents:
            returns += [pos]

        return tuple(returns) if len(returns) > 1 else returns[0]