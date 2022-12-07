import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from models.utils import SineWarmupScheduler, CosineWarmupScheduler
from models.CITRIS_encoder_decoder import Encoder, Decoder, SimpleEncoder, SimpleDecoder
from models.autoregressive_prior import CausalAssignmentNet, AutoregressivePrior
from models.intervention_classifier import InterventionClassifier
import wandb
import os
from models.utils import log_R2_statistic, log_Spearman_statistics
import numpy as np
from collections import OrderedDict
from models.utils import CosineWarmupScheduler, ImageLog
from models.causal_model import CausalNet


class CITRISVAE(torch.nn.Module):
    def __init__(self, args, causal_var_info, device):
        super(CITRISVAE, self).__init__()
        self.device = device
        self.causal_var_info = causal_var_info
        self.args = args

        # Load pretrained causal model for triplet evaluation
        if os.path.exists(args.pretrained_causal_model_path):
            self.pretrained_causal_model = CausalModel(causal_var_info=causal_var_info, 
                        img_width=args.img_width, 
                        c_in=args.c_in, 
                        c_hid=args.c_hid, 
                        is_mlp=False, 
                        device=device)
            self.pretrained_causal_model.causal_net.load_state_dict(torch.load(args.pretrained_causal_model_path))

        # VAE Encoder, Decoder
        if self.args.img_width == 32:
            self.encoder = SimpleEncoder(num_input_channels=self.args.c_in,
                                             base_channel_size=self.args.c_hid,
                                             latent_dim=self.args.num_latents)
            self.decoder = SimpleDecoder(num_input_channels=self.args.c_in,
                                             base_channel_size=self.args.c_hid,
                                             latent_dim=self.args.num_latents)
        else:
            self.encoder = Encoder(num_latents=self.args.num_latents,
                                          c_hid=self.args.c_hid,
                                          c_in=self.args.c_in,
                                          width=self.args.img_width,
                                          act_fn=lambda: nn.SiLU(),
                                          variational=True)
            self.decoder = Decoder(num_latents=self.args.num_latents,
                                          c_hid=self.args.c_hid,
                                          c_out=self.args.c_in,
                                          width=self.args.img_width,
                                          num_blocks=self.args.decoder_num_blocks,
                                          act_fn=lambda: nn.SiLU())

        # CausalAssignmentNet
        self.causal_assignment_net = CausalAssignmentNet(
            n_latents=self.args.num_latents,
            n_causal_vars=self.args.num_causal_vars + 1,
            lambda_reg=self.args.lambda_reg)

        # Transition prior
        self.transition_prior = AutoregressivePrior(
            causal_assignment_net=self.causal_assignment_net, 
            n_latents=self.args.num_latents,
            n_causal_vars=self.args.num_causal_vars + 1, 
            n_hid_per_latent=16,
            imperfect_interventions=self.args.imperfect_interventions,
            lambda_reg=self.args.lambda_reg)

        # Intervention Classifier
        self.intervention_classifier = InterventionClassifier(
            causal_assignment_net=self.causal_assignment_net,
            n_latents=self.args.num_latents,
            n_causal_vars=self.args.num_causal_vars,
            hidden_dim=self.args.c_hid,
            momentum=self.args.classifier_momentum,
            use_norm=self.args.classifier_use_normalization)

        # remove causal_assignment_net params since they are included in classifier params
        transition_prior_params = [p for n, p in self.transition_prior.named_parameters() if "causal_assignment_net" not in n and p.requires_grad]

        # Optimizer for training the model
        # self.optimizer = optim.AdamW([{'params': self.intervention_classifier.parameters(), 'lr': self.args.classifier_lr, 'weight_decay': 1e-4},
        #                               {'params': self.encoder.parameters()},
        #                               {'params': self.decoder.parameters()},
        #                               {'params': transition_prior_params}
        #                               ], lr=self.args.lr, weight_decay=0.0)

        self.optimizer = optim.AdamW([{'params': self.intervention_classifier.parameters()},
                                      {'params': self.encoder.parameters()},
                                      {'params': self.decoder.parameters()},
                                      {'params': transition_prior_params} 
                                      ], lr=self.args.lr, weight_decay=0.0)


        # Learning rate schedular for  model optimizer
        self.lr_scheduler = CosineWarmupScheduler(self.optimizer, warmup=self.args.warmup, max_iters=self.args.max_iters)
        # Learning rate schedular for KL
        self.kld_scheduler = SineWarmupScheduler(self.args.kld_warmup)

    def calculate_loss(self, imgs, target, x_rec, z_mean, z_logstd, z_sample, kld_factor):
        
        b, seq_len, c, h, w = imgs.shape
        z_sample = z_sample.view(b, seq_len, -1)
        z_mean = z_mean.view(b, seq_len, -1)
        z_logstd = z_logstd.view(b, seq_len, -1)
        x_rec = x_rec.view(b, seq_len-1, c, h, w)

        # KL divergence between every pair of frames
        kld_t1_all = self.transition_prior.compute_kl_loss(z_t=z_mean[:, :-1].view(b*(seq_len-1), -1), 
                                                     intrv=target.view(b*(seq_len-1), -1), 
                                                     z_t1_mean=z_mean[:, 1:].view(b*(seq_len-1), -1), 
                                                     z_t1_logstd=z_logstd[:, 1:].view(b*(seq_len-1), -1), 
                                                     z_t1_samples=z_sample[:, 1:].view(b*(seq_len-1), -1))

        kld_t1_all = kld_t1_all.view(b, seq_len-1).sum(dim=1)
        # reconstruction loss
        rec_loss = F.mse_loss(x_rec, imgs[:, 1:], reduction='none').sum(dim=[-3, -2, -1])

        # Combine to vae losses
        loss = (kld_factor * (kld_t1_all * self.args.beta_t1) + rec_loss.sum(dim=1)).mean()
        loss = loss / (seq_len - 1)

        # target classifier loss
        loss_model, loss_z = self.intervention_classifier(z_samples=z_sample, intrv_targets=target)
        loss = loss + (loss_model + loss_z) * self.args.beta_classifier

        return loss, rec_loss.mean(), loss_model, loss_z, kld_t1_all.mean() / (seq_len-1)

    def train(self, train_data_loader, val_data_loader, correlation_dataset, num_epochs, dataset_train):

        image_logger = ImageLog(exmp_inputs=next(iter(val_data_loader)), dataset=dataset_train)
        iteration = 0
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.intervention_classifier.to(self.device)
        self.transition_prior.to(self.device)
        for epoch in range(num_epochs):
            self.encoder.train()
            self.decoder.train()
            self.transition_prior.train()
            self.intervention_classifier.train()
            loss_avg, rec_loss_avg, loss_model_avg, loss_model_z_avg, kld_t1_avg = 0., 0., 0., 0., 0.
            for batch in train_data_loader:
                imgs, target = batch
                imgs = imgs.to(self.device)
                target = target.to(self.device)

                b, seq_len, c, h, w = imgs.shape

                # Encode
                z_mean, z_logstd = self.encoder(imgs.view(b*seq_len, c, h, w))
                z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()

                # Decode
                x_rec = self.decoder(z_sample.view(b, seq_len, -1)[:, 1:].view(b*(seq_len-1), -1))

                # Calculate Loss
                kld_factor = self.kld_scheduler.get_factor(iteration)
                loss, rec_loss, loss_model, loss_model_z, kld_t1 = self.calculate_loss(imgs, target, x_rec, z_mean, z_logstd, z_sample, kld_factor)

                self.optimizer.zero_grad()
                loss.backward() 
                self.optimizer.step()
                self.lr_scheduler.step()
                iteration += 1  

                loss_avg += loss.item() * imgs.shape[0]
                rec_loss_avg += rec_loss.item() * imgs.shape[0]
                loss_model_avg += loss_model.item() * imgs.shape[0]
                loss_model_z_avg += loss_model_z.item() * imgs.shape[0]
                kld_t1_avg += kld_t1.item() * imgs.shape[0]

            loss_avg /= len(train_data_loader.dataset)
            rec_loss_avg /= len(train_data_loader.dataset)
            loss_model_avg /= len(train_data_loader.dataset)
            loss_model_z_avg /= len(train_data_loader.dataset)
            kld_t1_avg /= len(train_data_loader.dataset)

            # Do triplet evaluation on validation set
            val_avg_loss, val_avg_norm_dist = 0., 0.
            if epoch % self.args.check_triplet_every_n_epoch == 0:
                val_avg_loss, val_avg_norm_dist = self.evaluate_with_triplet(val_data_loader, split="val", epoch=epoch)

            wandb.log({f'train_loss_avg': loss_avg}, step=epoch)
            wandb.log({f'train_rec_loss': rec_loss_avg}, step=epoch)
            wandb.log({f'train_classifier_loss': loss_model_avg}, step=epoch)
            wandb.log({f'train_classifier_loss_z': loss_model_z_avg}, step=epoch)
            wandb.log({f'train_kld': kld_t1_avg}, step=epoch)
            wandb.log({f'triplet_val_loss': val_avg_loss}, step=epoch)
            wandb.log({f'triplet_val_norm_dist': val_avg_norm_dist}, step=epoch)

            print(f"Epoch: {epoch}, loss_avg: {loss_avg: .4f} | rec_loss {rec_loss_avg: .4f} | classifier_loss: {loss_model_avg: .4f} | classifier_loss_z: {loss_model_z_avg: .4f} | kld: {kld_t1_avg: .4f} | triplet_val_loss: {val_avg_loss: .4f} | triplet_val_norm_dist: {val_avg_norm_dist: .4f}")

            # Evaluate and create Correlation matrix
            if epoch % self.args.check_correlation_every_n_epoch == 0:
                print("evalute-Correlation with probing")
                self.evaluate_correlation(correlation_dataset, "val", self.args.logdir, epoch)
                # Visualize Reconstruction
                image_logger.visualize(self, split="val", epoch=epoch)
                

    def enocode_dataset(self, dataset, train_percentage=0.5, stocastic=True):
        all_encs, all_latents = [], []
        loader = data.DataLoader(dataset, batch_size=256, drop_last=False, shuffle=False)
        self.encoder.eval()
        with torch.no_grad():
            for batch in loader:
                inps, *_, latents = batch
                inputs = inps.to(self.device)
                z_mean, z_logstd = self.encoder(inputs)
                if stocastic:
                    encs = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
                else:
                    encs = z_mean
                all_encs.append(encs.cpu())
                all_latents.append(latents)
        all_encs = torch.cat(all_encs, dim=0)
        all_latents = torch.cat(all_latents, dim=0)
        self.encoder.train()

        # Normalize for stable gradient signals 
        all_encs = (all_encs - all_encs.mean(dim=0, keepdim=True)) / all_encs.std(dim=0, keepdim=True).clamp(min=1e-2)

        # Create new tensor dataset for training and testing
        full_dataset = data.TensorDataset(all_encs, all_latents)
        train_size = int(train_percentage * all_encs.shape[0])
        test_size = all_encs.shape[0] - train_size
        train_dataset, test_dataset = data.random_split(full_dataset, lengths=[train_size, test_size], 
                                                        generator=torch.Generator().manual_seed(42))

        
        if hasattr(self, 'transition_prior'):
            target_assignment = self.transition_prior.get_target_assignment(hard=True)
        else:
            target_assignment = torch.eye(all_encs.shape[-1])
        
        return train_dataset, test_dataset, target_assignment

    def evaluate_correlation(self, dataset, split, logdir, epoch=None):
        train_dataset, test_dataset, target_assignment = self.enocode_dataset(dataset, train_percentage=0.5)

        causal_model = CausalModel(self.causal_var_info, self.args.img_width, self.args.num_latents*2, self.args.c_hid, is_mlp=True, device=self.device)

        train_loader = data.DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=512)

        causal_model.create_optimizer(lr=self.args.probe_lr, weight_decay=0., warmup=0, max_iters=-1)
        causal_model.train(self.args.probe_num_epochs, train_loader, test_loader=None, target_assignment=target_assignment, prepare_input_fn=self._prepare_input)

        trained_causal_net = causal_model.causal_net
        trained_causal_net.eval()

        test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=len(test_dataset))
        test_inps, test_labels = next(iter(test_loader))
        test_exp_inps, test_exp_labels = self._prepare_input(test_inps, target_assignment.cpu(), test_labels, flatten_inp=False)
        pred_dict = trained_causal_net(test_exp_inps.to(self.device))
        for key in pred_dict:
            pred_dict[key] = pred_dict[key].cpu()
        _, dists, norm_dists = causal_model.calculate_loss_distance(pred_dict, test_exp_labels)
        
        target_names = dataset.target_names()
        avg_norm_dists, r2_matrix = log_R2_statistic(target_names, causal_model, epoch, split, logdir, test_labels, norm_dists)
        log_Spearman_statistics(target_names, causal_model, epoch, split, logdir, pred_dict, test_labels)

        return r2_matrix
         

    def _prepare_input(self, inps, target_assignment, latents, flatten_inp=True):
        ta = target_assignment.detach()[None,:,:].expand(inps.shape[0], -1, -1)
        inps = torch.cat([inps[:,:,None] * ta, ta], dim=-2).permute(0, 2, 1)
        latents = latents[:,None].expand(-1, inps.shape[1], -1)
        if flatten_inp:
            inps = inps.flatten(0, 1)
            latents = latents.flatten(0, 1)
        return inps, latents

    def evaluate_with_triplet(self, dataloader, split, epoch=None):
        self.encoder.eval()
        self.decoder.eval()
        self.transition_prior.eval()

        self.pretrained_causal_model.causal_net.to(self.device)
        self.pretrained_causal_model.causal_net.eval()

        avg_loss = 0.
        avg_norm_dist = 0.
        for batch in dataloader:
            imgs, source, latents = batch
            # get pair of images
            if imgs.shape[1] > 2:
                imgs = imgs[:, :2]

            with torch.no_grad():
                imgs = imgs.to(self.device)
                source = source.to(self.device)
                latents = latents.to(self.device)

                # Here seq_len is 2
                b, seq_len, c, h, w = imgs.shape
                # Encode
                z_mean, _ = self.encoder(imgs.view(b*seq_len, c, h, w))

                # Use z_mean as sample for deterministic decoding
                z_mean = z_mean.view(b, seq_len, -1)

                target_assignment = self.transition_prior.get_target_assignment(hard=True)
                if source.shape[-1] + 1 == target_assignment.shape[-1]:  # No-variables missing
                    source = torch.cat([source, source[..., -1:] * 0.0], dim=-1)
                elif target_assignment.shape[-1] > source.shape[-1]:
                    target_assignment = target_assignment[..., :source.shape[-1]]

                mask = (target_assignment[None, :, :] * (1 - source[:, None, :])).sum(dim=-1)
                triplet_samples = mask * z_mean[:, 0] + (1 - mask) * z_mean[:, 1]

                generated_triplet = self.decoder(triplet_samples)

                if latents is not None and self.pretrained_causal_model is not None: 
                    v_dict = self.pretrained_causal_model.causal_net(generated_triplet)
                    losses, dists, norm_dists = self.pretrained_causal_model.calculate_loss_distance(v_dict, latents[:,-1])

                    for key in dists:
                        # For val evaluation during training
                        if epoch is not None:
                            wandb.log({f'triplet_{split}_{key}_dist': dists[key].mean()}, step=epoch)
                            wandb.log({f'triplet_{split}_{key}_norm_dist': norm_dists[key].mean()}, step=epoch)
                        else:
                            wandb.log({f'triplet_{split}_{key}_dist': dists[key].mean()})
                            wandb.log({f'triplet_{split}_{key}_norm_dist': norm_dists[key].mean()})
                    loss, norm_dist = 0., 0.
                    for key in losses:
                        loss += losses[key]
                    for key in norm_dists:
                        norm_dist += norm_dists[key].mean()
                else:
                    return 0.0, 0.0

                avg_loss += loss
                avg_norm_dist += norm_dist

        avg_loss /= len(dataloader.dataset)
        avg_norm_dist /= len(dataloader.dataset)

        return avg_loss, avg_norm_dist


################################ Causal Model ################################
class CausalModel(nn.Module):
    def __init__(self, causal_var_info, img_width, c_in, c_hid, is_mlp, device, angle_reg_weight=None, checkpoint_dir=None):

        super().__init__()
        self.device = device
        self.causal_var_info = causal_var_info
        self.causal_net = CausalNet(causal_var_info, img_width, c_in, c_hid, is_mlp)
        self.checkpoint_dir = checkpoint_dir
        self.angle_reg_weight = angle_reg_weight

        
    def create_optimizer(self, lr, weight_decay, warmup, max_iters):
        self.optimizer = optim.AdamW(self.causal_net.parameters(), lr=lr, weight_decay=weight_decay)

        self.lr_scheduler = None
        if max_iters != -1:
            self.lr_scheduler = CosineWarmupScheduler(self.optimizer, warmup=warmup, max_iters=max_iters)

    def calculate_loss_distance(self, pred_dict, gt_vec, keep_sign=False, is_training=False):
        # Function for calculating the loss and distance between predictions (pred_dict) and
        # ground truth (gt_vec) for every causal variable in the dictionary.
        losses = OrderedDict()
        dists = OrderedDict()
        norm_dists = OrderedDict()
        for i, var_key in enumerate(pred_dict):
            var_info = self.causal_var_info[var_key]
            gt_val = gt_vec[...,i]
            if var_info.startswith('continuous'):
                # MSE loss
                losses[var_key] = F.mse_loss(pred_dict[var_key].squeeze(dim=-1),
                                             gt_val, reduction='none')
                dists[var_key] = (pred_dict[var_key].squeeze(dim=-1) - gt_val)
                if not keep_sign:
                    dists[var_key] = dists[var_key].abs()
                norm_dists[var_key] = dists[var_key] / float(var_info.split('_')[-1])
            elif var_info.startswith('angle'):
                # Cosine similarity loss
                vec = torch.stack([torch.sin(gt_val), torch.cos(gt_val)], dim=-1)
                cos_sim = F.cosine_similarity(pred_dict[var_key], vec, dim=-1)
                losses[var_key] = 1 - cos_sim
                if is_training and self.angle_reg_weight is not None:
                    norm = pred_dict[var_key].norm(dim=-1, p=2.0)
                    losses[var_key + '_reg'] = self.angle_reg_weight * (2 - norm) ** 2
                dists[var_key] = torch.where(cos_sim > (1-1e-7), torch.zeros_like(cos_sim), torch.acos(cos_sim.clamp_(min=-1+1e-7, max=1-1e-7)))
                dists[var_key] = dists[var_key] / np.pi * 180.0  # rad to degrees
                norm_dists[var_key] = dists[var_key] / 180.0
            elif var_info.startswith('categ'):
                # Cross entropy loss
                gt_val = gt_val.long()
                pred = pred_dict[var_key]
                if len(pred.shape) > 2:
                    pred = pred.flatten(0, -2)
                    gt_val = gt_val.flatten(0, -1)
                losses[var_key] = F.cross_entropy(pred, gt_val, reduction='none')
                if len(pred_dict[var_key]) > 2:
                    losses[var_key] = losses[var_key].reshape(pred_dict[var_key].shape[:-1])
                    gt_val = gt_val.reshape(pred_dict[var_key].shape[:-1])
                dists[var_key] = (gt_val != pred_dict[var_key].argmax(dim=-1)).float()
                norm_dists[var_key] = dists[var_key]
            else:
                assert False, f'Do not know how to handle key \"{var_key}\" in calculating distances and losses.'
        for var_key in losses:
            losses[var_key] = losses[var_key].mean()
        return losses, dists, norm_dists

    
    def evaluate(self, test_loader):
        avg_loss = 0
        avg_norm_dist = 0
        self.causal_net.to(self.device)
        self.causal_net.eval()
        with torch.no_grad():
            for batch in test_loader:
                inps, labels = batch
                inps = inps.to(self.device)
                labels = labels.to(self.device)
                v_dict = self.causal_net(inps)
                losses, dists, norm_dists = self.calculate_loss_distance(v_dict, labels)

                loss, norm_dist = 0., 0.
                for key in losses:
                    loss += losses[key]
                for key in norm_dists:
                    norm_dist += norm_dists[key].mean()

                avg_loss += loss.item() * inps.shape[0]
                avg_norm_dist += norm_dist * inps.shape[0]

            avg_loss /= len(test_loader.dataset)
            avg_norm_dist /= len(test_loader.dataset)

        return avg_loss, avg_norm_dist

    def train(self, num_epochs, trainloader, test_loader=None, target_assignment=None, prepare_input_fn=None):
        if target_assignment is not None:
            target_assignment = target_assignment.to(self.device)
        self.causal_net.to(self.device)
        best_dist = 999999999.0
        for epoch in range(num_epochs):
            train_avg_loss = 0
            train_avg_norm_dist = 0
            self.causal_net.train()
            for batch in trainloader:
                inps, labels = batch
                inps = inps.to(self.device)
                labels = labels.to(self.device)

                # Handle Encoded dataset for training of evaluation
                if prepare_input_fn is not None and target_assignment is not None:
                    inps, labels = prepare_input_fn(inps, target_assignment, labels)

                v_dict = self.causal_net(inps)
                losses, dists, norm_dists = self.calculate_loss_distance(v_dict, labels, is_training=True)

                loss, norm_dist = 0., 0.
                for key in losses:
                    loss += losses[key]
                for key in norm_dists:
                    norm_dist += norm_dists[key].mean()

                self.optimizer.zero_grad()
                loss.backward() 
                self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                train_avg_loss += loss.item() * inps.shape[0]
                train_avg_norm_dist += norm_dist * inps.shape[0]
            train_avg_loss /= len(trainloader.dataset)
            train_avg_norm_dist /= len(trainloader.dataset)

            # Do not use in the training of linear probe evaluation
            if target_assignment is None:
                test_avg_loss, test_avg_norm_dist = self.evaluate(test_loader)

                if self.checkpoint_dir is not None and ((epoch == 0) or (train_avg_norm_dist < best_dist)):
                    best_dist = train_avg_norm_dist
                    PATH = os.path.join(self.checkpoint_dir, f"causal_model_best.pt")
                    torch.save(self.causal_net.state_dict(), PATH)

                if self.checkpoint_dir is not None:
                    PATH = os.path.join(self.checkpoint_dir, f"causal_model_last.pt")
                    torch.save(self.causal_net.state_dict(), PATH)

                print(f'Epoch {epoch:3d} | train_avg_loss {train_avg_loss:.4f} | train_avg_norm_dist {train_avg_norm_dist:.4f} | val_avg_loss {test_avg_loss:.4f} | val_avg_norm_dist {test_avg_norm_dist:.4f}')
                wandb.log({f'train_avg_loss': train_avg_loss}, step=epoch)
                wandb.log({f'train_avg_norm_dist': train_avg_norm_dist}, step=epoch)
                wandb.log({f'val_avg_loss': test_avg_loss}, step=epoch)
                wandb.log({f'val_avg_norm_dist': test_avg_norm_dist}, step=epoch)
            else:
                print(f'Epoch {epoch:3d} | train_avg_loss {train_avg_loss:.4f} | train_avg_norm_dist {train_avg_norm_dist:.4f}')
