import numpy as np
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sns
import os


@torch.no_grad()
def visualize_reconstruction(model, image, label, dataset):
    """ Plots the reconstructions of a VAE """
    reconst, *_ = model(image[None])
    reconst = reconst.squeeze(dim=0)

    if dataset.num_labels() > 1:
        soft_img = dataset.label_to_img(torch.softmax(reconst, dim=0))
        hard_img = dataset.label_to_img(torch.argmax(reconst, dim=0))
        if label.dtype == torch.long:
            true_img = dataset.label_to_img(label)
            diff_img = (hard_img != true_img).any(dim=-1, keepdims=True).long() * 255
        else:
            true_img = label
            soft_reconst = soft_img.float() / 255.0 * 2.0 - 1.0
            diff_img = (label - soft_reconst).clamp(min=-1, max=1)
    else:
        soft_img = reconst
        hard_img = reconst
        true_img = label
        diff_img = (label - reconst).clamp(min=-1, max=1)

    imgs = [image, true_img, soft_img, hard_img, diff_img]
    titles = ['Original image', 'GT Labels', 'Soft prediction', 'Hard prediction', 'Difference']
    imgs = [t.permute(1, 2, 0) if (t.shape[0] in [3,4] and t.shape[-1] != 3) else t for t in imgs]
    imgs = [t.detach().cpu().numpy() for t in imgs]
    imgs = [((t + 1.0) * 255.0 / 2.0).astype(np.int32) if t.dtype == np.float32 else t for t in imgs]
    imgs = [t.astype(np.uint8) for t in imgs]

    fig, axes = plt.subplots(1, len(imgs), figsize=(10, 3))
    for np_img, title, ax in zip(imgs, titles, axes):
        ax.imshow(np_img)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    return fig


@torch.no_grad()
def plot_target_assignment(prior, dataset=None):
    """ Plots the probability matrix of latent-to-causal variable assignments """
    target_probs = prior.get_target_assignment().detach().cpu().numpy()
    fig = plt.figure(figsize=(max(6, target_probs.shape[1]), max(6, target_probs.shape[0]/2.5)))
    if dataset is not None:
        target_names = dataset.target_names()
        if len(target_names) == target_probs.shape[1]-1:
            target_names = target_names + ['No variable']
    else:
        target_names = [f'Block {i+1}' for i in range(target_probs.shape[1])]
    sns.heatmap(target_probs, annot=True,
                yticklabels=[f'Dim {i+1}' for i in range(target_probs.shape[0])],
                xticklabels=target_names)
    plt.xlabel('Blocks/Causal Variable')
    plt.ylabel('Latent dimensions')
    plt.title('Soft assignment of latent variable to block')
    plt.tight_layout()
    return 


@torch.no_grad()
def plot_target_classification(results):
    """ Plots the classification accuracies of the target classifier """
    results = {key.split('.')[-1]: results[key] for key in results.keys() if key.startswith('training_step.target_classifier')}
    if len(results) == 0:
        return None
    else:
        key_to_block = lambda key: key.split('_')[-2].replace('block','').replace('[','').replace(']','')
        key_to_class = lambda key: key.split('_class')[-1].replace('[','').replace(']','')
        blocks = sorted(list(set([key_to_block(key) for key in results])))
        classes = sorted(list(set([key_to_class(key) for key in results])))
        target_accs = np.zeros((len(blocks), len(classes)), dtype=np.float32)
        for key in results:
            target_accs[blocks.index(key_to_block(key)), classes.index(key_to_class(key))] = results[key].value / results[key].cumulated_batch_size

        fig = plt.figure(figsize=(max(4, len(classes)/1.25), max(4, len(blocks)/1.25)))
        sns.heatmap(target_accs, annot=True,
                    yticklabels=blocks,
                    xticklabels=classes)
        plt.xlabel('Variable classes/targets')
        plt.ylabel('Variable blocks')
        plt.title('Classification accuracy of blocks to causal variables')
        plt.tight_layout()
        return 


def create_pos_grid(shape, device, stack_dim=-1):
    pos_x, pos_y = torch.meshgrid(torch.linspace(-1, 1, shape[0], device=device),
                                  torch.linspace(-1, 1, shape[1], device=device))
    pos = torch.stack([pos_x, pos_y], dim=stack_dim)
    return pos


def get_act_fn(act_fn_name):
    """ Map activation function string to activation function """
    act_fn_name = act_fn_name.lower()
    if act_fn_name == 'elu':
        act_fn_func = nn.ELU
    elif act_fn_name == 'silu':
        act_fn_func = nn.SiLU
    elif act_fn_name == 'leakyrelu':
        act_fn_func = lambda: nn.LeakyReLU(negative_slope=0.05, inplace=True)
    elif act_fn_name == 'relu':
        act_fn_func = nn.ReLU
    else:
        assert False, f'Unknown activation function \"{act_fn_name}\"'
    return act_fn_func


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """ Learning rate scheduler with Cosine annealing and warmup """

    def __init__(self, optimizer, warmup, max_iters, min_factor=0.05, offset=0):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.min_factor = min_factor
        self.offset = offset
        super().__init__(optimizer)
        if isinstance(self.warmup, list) and not isinstance(self.offset, list):
            self.offset = [self.offset for _ in self.warmup]

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        if isinstance(lr_factor, list):
            return [base_lr * f for base_lr, f in zip(self.base_lrs, lr_factor)]
        else:
            return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        lr_factor = lr_factor * (1 - self.min_factor) + self.min_factor
        if isinstance(self.warmup, list):
            new_lr_factor = []
            for o, w in zip(self.offset, self.warmup):
                e = max(0, epoch - o)
                l = lr_factor * ((e * 1.0 / w) if e <= w and w > 0 else 1)
                new_lr_factor.append(l)
            lr_factor = new_lr_factor
        else:
            epoch = max(0, epoch - self.offset)
            if epoch <= self.warmup and self.warmup > 0:
                lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class SineWarmupScheduler(object):
    """ Warmup scheduler used for KL divergence, if chosen """

    def __init__(self, warmup, start_factor=0.1, end_factor=1.0, offset=0):
        super().__init__()
        self.warmup = warmup
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.offset = offset

    def get_factor(self, step):
        step = step - self.offset
        if step >= self.warmup:
            return self.end_factor
        elif step < 0:
            return self.start_factor
        else:
            v = self.start_factor + (self.end_factor - self.start_factor) * 0.5 * (1 - np.cos(np.pi * step / self.warmup))
            return v


def log_matrix(matrix, trainer, name, current_epoch=None, log_dir=None):
    """ Saves a numpy array to the logging directory """
    if log_dir is None:
        log_dir = trainer.logger.log_dir
    filename = os.path.join(log_dir, name + '.npz')

    new_epoch = trainer.current_epoch if current_epoch is None else current_epoch
    new_epoch = np.array([new_epoch])
    new_val = matrix[None]
    if os.path.isfile(filename):
        prev_data = np.load(filename)
        epochs, values = prev_data['epochs'], prev_data['values']
        epochs = np.concatenate([epochs, new_epoch], axis=0)
        values = np.concatenate([values, new_val], axis=0)
    else:
        epochs = new_epoch
        values = new_val
    np.savez_compressed(filename, epochs=epochs, values=values)