import numpy as np
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr
from collections import OrderedDict, defaultdict
import wandb
import torch.nn.functional as F


def gaussian_log_prob(mean, log_std, samples):
    """ Returns the log probability of a specified Gaussian for a tensor of samples """
    if len(samples.shape) == len(mean.shape)+1:
        mean = mean[...,None]
    if len(samples.shape) == len(log_std.shape)+1:
        log_std = log_std[...,None]
    return - log_std - 0.5 * np.log(2*np.pi) - 0.5 * ((samples - mean) / log_std.exp())**2


class AutoregLinear(nn.Module):

    def __init__(self, num_vars, inp_per_var, out_per_var, diagonal=False, 
                       no_act_fn_init=False, 
                       init_std_factor=1.0, 
                       init_bias_factor=1.0,
                       init_first_block_zeros=False):
        """
        Autoregressive linear layer, where the weight matrix is correspondingly masked.
        Parameters
        ----------
        num_vars : int
                   Number of autoregressive variables/steps.
        inp_per_var : int
                      Number of inputs per autoregressive variable.
        out_per_var : int
                      Number of outputs per autoregressvie variable.
        diagonal : bool
                   If True, the n-th output depends on the n-th input.
                   If False, the n-th output only depends on the inputs 1 to n-1
        """
        super().__init__()
        self.linear = nn.Linear(num_vars * inp_per_var, 
                                num_vars * out_per_var)
        mask = torch.zeros_like(self.linear.weight.data)
        init_kwargs = {}
        if no_act_fn_init:  # Set kaiming to init for linear act fn
            init_kwargs['nonlinearity'] = 'leaky_relu'
            init_kwargs['a'] = 1.0
        for out_var in range(num_vars):
            out_start_idx = out_var * out_per_var
            out_end_idx = (out_var+1) * out_per_var
            inp_end_idx = (out_var+(1 if diagonal else 0)) * inp_per_var
            if inp_end_idx > 0:
                mask[out_start_idx:out_end_idx, :inp_end_idx] = 1.0
                if out_var == 0 and init_first_block_zeros:
                    self.linear.weight.data[out_start_idx:out_end_idx, :inp_end_idx].fill_(0.0)
                else:
                    nn.init.kaiming_uniform_(self.linear.weight.data[out_start_idx:out_end_idx, :inp_end_idx], **init_kwargs)
        self.linear.weight.data.mul_(init_std_factor)
        self.linear.bias.data.mul_(init_bias_factor)
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.linear(x, self.linear.weight * self.mask, self.linear.bias)

@torch.no_grad()
def visualize_reconstruction(model, image, label, dataset):
    """ Plots the reconstructions of a VAE """
    image2 = image[None].to(model.device)
    z_mean, z_logstd = model.encoder(image2)
    z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
    reconst = model.decoder(z_sample)
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


def log_matrix(matrix, epoch, name, log_dir):
    """ Saves a numpy array to the logging directory """

    filename = os.path.join(log_dir, name + '.npz')

    new_epoch = epoch
    new_epoch = np.array([new_epoch])
    new_val = matrix[None]
    if os.path.isfile(filename):
        prev_data = np.load(filename, allow_pickle=True)
        epochs, values = prev_data['epochs'], prev_data['values']
        epochs = np.concatenate([epochs, new_epoch], axis=0)
        values = np.concatenate([values, new_val], axis=0)
    else:
        epochs = new_epoch
        values = new_val
    np.savez_compressed(filename, epochs=epochs, values=values)



def _log_heatmap(target_names, values, epoch, tag, split, title=None, xticks=None, yticks=None, xlabel=None, ylabel=None):
    if ylabel is None:
        ylabel = 'Target dimension'
    if xlabel is None:
        xlabel = 'True causal variable'
    if yticks is None:
        yticks = target_names+['No variable']
        if values.shape[0] > len(yticks):
            yticks = [f'Dim {i+1}' for i in range(values.shape[0])]
        if len(yticks) > values.shape[0]:
            yticks = yticks[:values.shape[0]]
    if xticks is None:
        xticks = target_names
    fig = plt.figure(figsize=(min(6, values.shape[1]/1.25), min(6, values.shape[0]/1.25)))
    sns.heatmap(values, annot=True,
                    yticklabels=yticks,
                    xticklabels=xticks,
                    fmt='3.2f')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.tight_layout()

    if epoch is not None:
        wandb.log({tag + split: wandb.Image(fig)}, step=epoch)
    else:
        wandb.log({tag + split: wandb.Image(fig)})
    plt.close(fig)

    if values.shape[0] == values.shape[1] + 1:  # Remove 'lambda_sparse' variables
        values = values[:-1]

    if values.shape[0] == values.shape[1]:
        avg_diag = np.diag(values).mean()
        max_off_diag = (values - np.eye(values.shape[0]) * 10).max(axis=-1).mean()
        if epoch is not None:
            wandb.log({f'corr_{tag}_diag{split}': avg_diag}, step=epoch)
            wandb.log({f'corr_{tag}_max_off_diag{split}': max_off_diag}, step=epoch)
            print(f"Epoch {epoch} | corr_{tag}_diag{split}: {avg_diag:0.4f} | corr_{tag}_max_off_diag{split}: {max_off_diag:0.4f} ")
        else:
            wandb.log({f'corr_{tag}_diag{split}': avg_diag})
            wandb.log({f'corr_{tag}_max_off_diag{split}': max_off_diag})
            print(f"corr_{tag}_diag{split}: {avg_diag:0.4f} | corr_{tag}_max_off_diag{split}: {max_off_diag:0.4f} ")

def log_R2_statistic(target_names, encoder, epoch, split, logdir, test_labels, norm_dists):
    avg_pred_dict = OrderedDict()
    for i, var_key in enumerate(encoder.causal_var_info):
        var_info = encoder.causal_var_info[var_key]
        gt_vals = test_labels[...,i]
        
        if var_info.startswith('continuous'):
            avg_pred_dict[var_key] = gt_vals.mean(dim=0, keepdim=True).expand(gt_vals.shape[0],)
        elif var_info.startswith('angle'):
            avg_angle = torch.atan2(torch.sin(gt_vals).mean(dim=0, keepdim=True), 
                                        torch.cos(gt_vals).mean(dim=0, keepdim=True)).expand(gt_vals.shape[0],)
            avg_angle = torch.where(avg_angle < 0.0, avg_angle + 2*np.pi, avg_angle)
            avg_pred_dict[var_key] = torch.stack([torch.sin(avg_angle), torch.cos(avg_angle)], dim=-1)
        elif var_info.startswith('categ'):
            gt_vals = gt_vals.long()
            mode = torch.mode(gt_vals, dim=0, keepdim=True).values
            avg_pred_dict[var_key] = F.one_hot(mode, int(var_info.split('_')[-1])).float().expand(gt_vals.shape[0], -1)
        else:
            assert False, f'Do not know how to handle key \"{var_key}\" in R2 statistics.'
    _, _, avg_norm_dists = encoder.calculate_loss_distance(avg_pred_dict, test_labels, keep_sign=True)
    
    r2_matrix = []
    for var_key in encoder.causal_var_info:
        ss_res = (norm_dists[var_key] ** 2).mean(dim=0)
        ss_tot = (avg_norm_dists[var_key] ** 2).mean(dim=0, keepdim=True)
        r2 = 1 - ss_res / ss_tot
        r2_matrix.append(r2)
    r2_matrix = torch.stack(r2_matrix, dim=-1).cpu().detach().numpy()
    log_matrix(r2_matrix, epoch, 'r2_matrix_' + split, logdir)
    _log_heatmap(target_names, r2_matrix, epoch, tag='r2_matrix_', split=split, title='R^2 Matrix', xticks=[key for key in encoder.causal_var_info])

    return avg_norm_dists, r2_matrix

def log_pearson_statistic(target_names, encoder, epoch, split, logdir, pred_dict, avg_gt_norm_dists):
    avg_pred_dict = OrderedDict()
    for i, var_key in enumerate(encoder.causal_var_info):
        var_info = encoder.causal_var_info[var_key]
        pred_vals = pred_dict[var_key]
        if var_info.startswith('continuous'):
            pred_vals = pred_vals.squeeze(dim=-1)
            avg_pred_dict[var_key] = pred_vals.mean(dim=0, keepdim=True).expand(pred_vals.shape[0], -1)
        elif var_info.startswith('angle'):
            angles = torch.atan(pred_vals[...,0] / pred_vals[...,1])
            avg_angle = torch.atan2(torch.sin(angles).mean(dim=0, keepdim=True), 
                                        torch.cos(angles).mean(dim=0, keepdim=True)).expand(pred_vals.shape[0], -1)
            avg_angle = torch.where(avg_angle < 0.0, avg_angle + 2*np.pi, avg_angle)
            avg_pred_dict[var_key] = avg_angle
        elif var_info.startswith('categ'):
            pred_vals = pred_vals.argmax(dim=-1)
            mode = torch.mode(pred_vals, dim=0, keepdim=True).values
            avg_pred_dict[var_key] = mode.expand(pred_vals.shape[0], -1)
        else:
            assert False, f'Do not know how to handle key \"{var_key}\" in Pearson statistics.'
    _, _, avg_pred_norm_dists = encoder.calculate_loss_distance(pred_dict, gt_vec=torch.stack([avg_pred_dict[key] for key in avg_pred_dict], dim=-1), keep_sign=True)

    pearson_matrix = []
    for var_key in encoder.causal_var_info:
        var_info = encoder.causal_var_info[var_key]
        pred_dist, gt_dist = avg_pred_norm_dists[var_key], avg_gt_norm_dists[var_key]
        nomin = (pred_dist * gt_dist[:,None]).sum(dim=0)
        denom = torch.sqrt((pred_dist**2).sum(dim=0) * (gt_dist[:,None]**2).sum(dim=0))
        p = nomin / denom.clamp(min=1e-5)
        pearson_matrix.append(p)
    pearson_matrix = torch.stack(pearson_matrix, dim=-1).cpu().detach().numpy()

    log_matrix(pearson_matrix, epoch, 'pearson_matrix_' + split, logdir)
    _log_heatmap(target_names, pearson_matrix, epoch, tag='pearson_matrix_', split=split, title='Pearson Matrix', xticks=[key for key in encoder.causal_var_info])

def log_Spearman_statistics(target_names, encoder, epoch, split, logdir, pred_dict, test_labels):
    spearman_matrix = []
    for i, var_key in enumerate(encoder.causal_var_info):
        var_info = encoder.causal_var_info[var_key]
        gt_vals = test_labels[...,i]
        pred_val = pred_dict[var_key]
        if var_info.startswith('continuous'):
            spearman_preds = pred_val.squeeze(dim=-1)  # Nothing needs to be adjusted
        elif var_info.startswith('angle'):
            spearman_preds = F.normalize(pred_val, p=2.0, dim=-1)
            gt_vals = torch.stack([torch.sin(gt_vals), torch.cos(gt_vals)], dim=-1)
        elif var_info.startswith('categ'):
            spearman_preds = pred_val.argmax(dim=-1).float()
        else:
            assert False, f'Do not know how to handle key \"{var_key}\" in Spearman statistics.'

        gt_vals = gt_vals.cpu().detach().numpy()
        spearman_preds = spearman_preds.cpu().detach().numpy()
        results = torch.zeros(spearman_preds.shape[1],)
        for j in range(spearman_preds.shape[1]):
            if len(spearman_preds.shape) == 2:
                if np.unique(spearman_preds[:,j]).shape[0] == 1:
                    results[j] = 0.0
                else:
                    results[j] = spearmanr(spearman_preds[:,j], gt_vals).correlation
            elif len(spearman_preds.shape) == 3:
                num_dims = spearman_preds.shape[-1]
                for k in range(num_dims):
                    if np.unique(spearman_preds[:,j,k]).shape[0] == 1:
                        results[j] = 0.0
                    else:
                        results[j] += spearmanr(spearman_preds[:,j,k], gt_vals[...,k]).correlation
                results[j] /= num_dims
                
        spearman_matrix.append(results)
        
    spearman_matrix = torch.stack(spearman_matrix, dim=-1).cpu().detach().numpy()

    log_matrix(spearman_matrix, epoch, 'spearman_matrix_' + split, logdir)
    _log_heatmap(target_names, spearman_matrix, epoch, tag='spearman_matrix_', split=split, title='Spearman\'s Rank Correlation Matrix', xticks=[key for key in encoder.causal_var_info])


@torch.no_grad()
def visualize_triplet_reconstruction(model, img_triplet, labels, sources, dataset=None, *args, **kwargs):
    """ Plots the triplet predictions against the ground truth for a VAE/Flow """
    sources = sources[0].to(model.device)
    labels = labels[-1]
    triplet_rec = model.generate_triplet(img_triplet[None], sources[None])
    triplet_rec = triplet_rec.squeeze(dim=0)
    if labels.dtype == torch.long:
        triplet_rec = triplet_rec.argmax(dim=0)
        diff_img = (triplet_rec != labels).long() * 255
    else:
        diff_img = ((triplet_rec - labels).clamp(min=-1, max=1) + 1) / 2.0
    triplet_rec = dataset.label_to_img(triplet_rec)
    labels = dataset.label_to_img(labels)
    vs = [img_triplet, labels, sources, triplet_rec, diff_img]
    vs = [e.squeeze(dim=0) for e in vs]
    vs = [t.permute(0, 2, 3, 1) if (len(t.shape) == 4 and t.shape[0] in [3,4] and t.shape[-1] != 3) else t for t in vs]
    vs = [t.permute(1, 2, 0) if (len(t.shape) == 3 and t.shape[0] in [3,4] and t.shape[-1] != 3) else t for t in vs]
    vs = [e.detach().cpu().numpy() for e in vs]
    img_triplet, labels, sources, triplet_rec, diff_img = vs
    img_triplet = (img_triplet + 1.0) / 2.0
    s1 = np.where(sources == 0)[0]
    s2 = np.where(sources == 1)[0]

    fig, axes = plt.subplots(1, 5, figsize=(8, 3))
    for i, (img, title) in enumerate(zip([img_triplet[0], img_triplet[1], triplet_rec, labels, diff_img], 
                                         ['Image 1', 'Image 2', 'Reconstruction', 'GT Label', 'Difference'])):
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    targets = dataset.target_names()
    fig.suptitle(f'Image 1: {[targets[i] for i in s1]}, Image 2: {[targets[i] for i in s2]}')
    plt.tight_layout()
    return fig


class ImageLog:
    """ Class for creating visualizations for logging """
    def __init__(self, exmp_inputs, dataset):
        super().__init__()
        self.imgs = exmp_inputs[0]
        if len(exmp_inputs) > 2 and len(exmp_inputs[1].shape) == len(self.imgs.shape):
            self.labels = exmp_inputs[1]
            self.extra_inputs = exmp_inputs[2:]
        else:
            self.labels = self.imgs
            self.extra_inputs = exmp_inputs[1:]
        self.dataset = dataset

    def visualize(self, model, split, epoch):
        def log_fig(tag, fig):
            if fig is None:
                return
            wandb.log({f'{split}_{tag}': wandb.Image(fig)}, step=epoch)
            plt.close(fig)
        # TODO: need to fix it
        # if hasattr(model, 'transition_prior'):
        #     log_fig('transition_prior', plot_target_assignment(model.transition_prior, dataset=self.dataset))

        if self.imgs is not None:
            images = self.imgs.to(model.device)
            labels = self.labels.to(model.device)
            if len(images.shape) == 5:
                full_imgs, full_labels = images, labels
                images = images[:,0]
                labels = labels[:,0]
            else:
                full_imgs, full_labels = None, None

            for i in range(min(4, images.shape[0])):
                log_fig(f'reconstruction_{i}', visualize_reconstruction(model, images[i], labels[i], self.dataset))
            
        # TODO: need to fix it
        if hasattr(model, 'transition_prior'):
            if full_imgs is not None:
                for i in range(min(4, full_imgs.shape[0])):
                    log_fig(f'triplet_visualization_{i}', visualize_triplet_reconstruction(model, full_imgs[i], full_labels[i], [e[i] for e in self.extra_inputs], dataset=self.dataset))
