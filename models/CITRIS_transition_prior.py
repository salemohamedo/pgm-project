import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

import sys
sys.path.append('../../')
# from models.shared.utils import kl_divergence, gaussian_log_prob
# from models.shared.modules import MultivarLinear, AutoregLinear

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


def kl_divergence(mean1, log_std1, mean2=None, log_std2=None): 
    """ Returns the KL divergence between two Gaussian distributions """
    if mean2 is None:
        mean2 = torch.zeros_like(mean1)
    if log_std2 is None:
        log_std2 = torch.zeros_like(log_std1)

    var1, var2 = (2*log_std1).exp(), (2*log_std2).exp()
    KLD = (log_std2 - log_std1) + (var1 + (mean1 - mean2) ** 2) / (2 * var2) - 0.5
    return KLD


class AutoregressiveConditionalPrior(nn.Module):
    """
    The autoregressive base model for the autoregressive transition prior.
    The model is inspired by MADE and uses linear layers with masked weights.
    """

    def __init__(self, num_latents, num_blocks, c_hid, c_out, imperfect_interventions=True):
        """
        Parameters
        ----------
        num_latents : int
                      Number of latent dimensions.
        num_blocks : int
                     Number of blocks to group the latent dimensions into. In other words,
                     it is the number of causal variables plus 1 (psi(0) - unintervened information).
        c_hid : int
                Hidden dimensionality to use in the network.
        c_out : int
                Output dimensionality per latent dimension (2 for Gaussian param estimation)
        imperfect_interventions : bool
                                  Whether interventions may be imperfect or not. If not, we mask the
                                  conditional information on interventions for a slightly faster
                                  convergence.
        """
        super().__init__()
        # Input layer for z_t
        self.context_layer = nn.Linear(num_latents, num_latents * c_hid)
        # Input layer for I_t
        self.target_layer = nn.Linear(num_blocks, num_latents * c_hid)
        # Autoregressive input layer for z_t+1
        self.init_layer = AutoregLinear(num_latents, 2, c_hid, diagonal=False)
        # Autoregressive main network with masked linear layers
        self.net = nn.Sequential(
                nn.SiLU(),
                AutoregLinear(num_latents, c_hid, c_hid, diagonal=True),
                nn.SiLU(),
                AutoregLinear(num_latents, c_hid, c_out, diagonal=True)
            )
        self.num_latents = num_latents
        self.imperfect_interventions = imperfect_interventions
        self.register_buffer('target_mask', torch.eye(num_blocks))

    def forward(self, z_samples, z_previous, target_samples, target_true):
        """
        Given latent variables z^t+1, z^t, intervention targets I^t+1, and 
        causal variable assignment samples from psi, estimate the prior 
        parameters of p(z^t+1|z^t, I^t+1). This is done by running the
        autoregressive prior for each causal variable to estimate 
        p(z_psi(i)^t+1|z^t, I_i^t+1), and stacking the i-dimension.


        Parameters
        ----------
        z_samples : torch.FloatTensor, shape [batch_size, num_latents]
                    The values of the latent variables at time step t+1, i.e. z^t+1.
        z_previous : torch.FloatTensor, shape [batch_size, num_latents]
                     The values of the latent variables at time step t, i.e. z^t.
        target_samples : torch.FloatTensor, shape [batch_size, num_latents, num_blocks]
                         The sampled one-hot vectors of psi for assigning latent variables to 
                         causal variables.
        target_true : torch.FloatTensor, shape [batch_size, num_blocks]
                      The intervention target vector I^t+1.
        """
        target_samples = target_samples.permute(0, 2, 1)  # shape: [batch_size, num_blocks, num_latents]
        
        # Transform z^t into a feature vector. Expand over number of causal variables to run the prior i-times.
        context_feats = self.context_layer(z_previous)
        context_feats = context_feats.unsqueeze(dim=1)  # shape: [batch_size, 1, num_latents * c_hid]

        # Transform I^t+1 into feature vector, where only the i-th element is shown to the respective masked split.
        target_inp = target_true[:,None] * self.target_mask[None]  # shape: [batch_size, num_blocks, num_latents] 
        target_inp = target_inp - (1 - self.target_mask[None]) # Set -1 for masked values
        target_feats = self.target_layer(target_inp)

        # Mask z^t+1 according to psi samples
        masked_samples = z_samples[:,None] * target_samples # shape: [batch_size, num_blocks, num_latents]
        masked_samples = torch.stack([masked_samples, target_samples*2-1], dim=-1)
        masked_samples = masked_samples.flatten(-2, -1) # shape: [batch_size, num_blocks, 2*num_latents]
        init_feats = self.init_layer(masked_samples)

        if not self.imperfect_interventions:
            # Mask out context features when having perfect interventions
            context_feats = context_feats * (1 - target_true[...,None])

        # Sum all features and use as input to feature network (division by 2 for normalization)
        feats = (target_feats + init_feats + context_feats) / 2.0
        pred_params = self.net(feats)

        # Return prior parameters with first dimension stacking the different causal variables
        pred_params = pred_params.unflatten(-1, (self.num_latents, -1))  # shape: [batch_size, num_blocks, num_latents, c_out]
        return pred_params


class TransitionPrior(nn.Module):
    """
    The full transition prior promoting disentanglement of the latent variables across causal factors.
    """

    def __init__(self, num_latents, num_blocks, c_hid,
                 imperfect_interventions=False,
                 autoregressive_model=False,
                 lambda_reg=0.01,
                 gumbel_temperature=1.0):
        """
        Parameters
        ----------
        num_latents : int
                      Number of latent dimensions.
        num_blocks : int
                     Number of blocks to group the latent dimensions into. In other words,
                     it is the number of causal variables plus 1 (psi(0) - unintervened information).
        c_hid : int
                Hidden dimensionality to use in the prior network.
        imperfect_interventions : bool
                                  Whether interventions may be imperfect or not.
        autoregressive_model : bool
                               If True, an autoregressive prior model is used.
        lambda_reg : float
                     Regularizer for promoting intervention-independent information to be modeled
                     in psi(0)
        gumbel_temperature : float
                             Temperature to use for the Gumbel Softmax sampling.
        """
        super().__init__()
        self.num_latents = num_latents
        self.imperfect_interventions = imperfect_interventions
        self.gumbel_temperature = gumbel_temperature
        self.num_blocks = num_blocks
        self.autoregressive_model = autoregressive_model
        self.lambda_reg = lambda_reg
        assert self.lambda_reg >= 0 and self.lambda_reg < 1.0, 'Lambda regularizer must be between 0 and 1, excluding 1.'

        # Gumbel Softmax parameters of psi. Note that we model psi(0) in the last dimension for simpler implementation
        self.target_params = nn.Parameter(torch.zeros(num_latents, num_blocks + 1))
        if self.lambda_reg <= 0.0:  # No regularizer -> no reason to model psi(0)
            self.target_params.data[:,-1] = -9e15

        # For perfect interventions, we model the prior's parameters under intervention as a simple parameter vector here.
        if not self.imperfect_interventions:
            self.intv_prior = nn.Parameter(torch.zeros(num_latents, num_blocks, 2).uniform_(-0.5, 0.5))
        else:
            self.intv_prior = None

        # Prior model creation
        self.prior_model = AutoregressiveConditionalPrior(num_latents, num_blocks+1, 16, 2,
                                                            imperfect_interventions=self.imperfect_interventions)


    def _get_prior_params(self, z_t, target=None, target_prod=None, target_samples=None, z_t1=None):
        """
        Abstracting the execution of the networks for estimating the prior parameters.

        Parameters
        ----------
        z_t : torch.FloatTensor, shape [batch_size, num_latents]
              Latents at time step t, i.e. the input to the prior
        target : torch.FloatTensor, shape [batch_size, num_blocks]
                 The intervention targets I^t+1
        target_prod : torch.FloatTensor, shape [batch_size, num_latents, num_blocks]
                      The true targets multiplied with the target sample mask, where masked
                      intervention targets are replaced with -1 to distinguish it from 0s.
        target_samples : torch.FloatTensor, shape [batch_size, num_latents, num_blocks]
                         The sampled one-hot vectors of psi for assigning latent variables to 
                         causal variables.
        z_t1 : torch.FloatTensor, shape [batch_size, num_latents]
               Latents at time step t+1, i.e. the latents for which the prior parameters are estimated.
        """
        if self.autoregressive_model:
            prior_params = self.prior_model(z_samples=z_t1, 
                                            z_previous=z_t, 
                                            target_samples=target_samples,
                                            target_true=target)
            prior_params = prior_params.unbind(dim=-1)
        else:
            net_inp = z_t
            context = self.context_layer(net_inp).unflatten(-1, (self.num_latents, -1))
            net_inp_exp = net_inp.unflatten(-1, (self.num_latents, -1))
            if self.imperfect_interventions:
                if target_prod is None:
                    target_prod = net_inp_exp.new_zeros(net_inp_exp.shape[:-1] + (self.num_blocks,))
                net_inp_exp = torch.cat([net_inp_exp, target_prod], dim=-1)
            block_inp = self.inp_layer(net_inp_exp)
            prior_params = self.out_layer(context + block_inp)
            prior_params = prior_params.chunk(2, dim=-1)
            prior_params = [p.flatten(-2, -1) for p in prior_params]
        return prior_params

    def kl_divergence(self, z_t, target, z_t1_mean, z_t1_logstd, z_t1_sample):
        """
        Calculating the KL divergence between this prior's estimated parameters and
        the encoder on x^t+1 (CITRIS-VAE). Since this prior is in general much more
        computationally cheaper than the encoder/decoder, we marginalize the KL
        divergence over the target assignments for each latent where possible.

        Parameters
        ----------
        z_t : torch.FloatTensor, shape [batch_size, num_latents]
              Latents at time step t, i.e. the input to the prior
        target : torch.FloatTensor, shape [batch_size, num_blocks]
                 The intervention targets I^t+1
        z_t1_mean : torch.FloatTensor, shape [batch_size, num_latents]
                    The mean of the predicted Gaussian encoder(x^t+1)
        z_t1_logstd : torch.FloatTensor, shape [batch_size, num_latents]
                      The log-standard deviation of the predicted Gaussian encoder(x^t+1)
        z_t1_sample : torch.FloatTensor, shape [batch_size, num_latents]
                      A sample from the encoder distribution encoder(x^t+1), i.e. z^t+1
        """
        if len(target.shape) == 1:
            target_oh = F.one_hot(target, num_classes=self.num_blocks)
        else:
            target_oh = target

        # Sample a latent-to-causal assignment from psi
        target_probs = torch.softmax(self.target_params, dim=-1)
        target_samples = F.gumbel_softmax(self.target_params[None].expand(target.shape[0], -1, -1), 
                                       tau=self.gumbel_temperature, hard=True)
        full_target_samples = target_samples
        target_samples, no_target = target_samples[:,:,:-1], target_samples[:,:,-1]
        # Add I_0=0, i.e. no interventions on the noise/intervention-independent variables
        target_exp = torch.cat([target_oh, target_oh.new_zeros(target_oh.shape[0], 1)], dim=-1)

        if self.autoregressive_model:
            # Run autoregressive model
            prior_params = self._get_prior_params(z_t, target_samples=full_target_samples, target=target_exp, z_t1=z_t1_sample)
            kld_all = self._get_kld(z_t1_mean[:,None], z_t1_logstd[:,None], prior_params)
            # Regularize psi(0)
            if self.lambda_reg > 0.0:
                target_probs = torch.cat([target_probs[:,-1:], target_probs[:,:-1] * (1 - self.lambda_reg)], dim=-1)
            # Since to predict the parameters of z^t+1_i, we do not involve whether the target sample of i has been a certain value,
            # we can marginalize it over all possible target samples here.
            kld = (kld_all * target_probs.permute(1, 0)[None]).sum(dim=[1,2])

        
        return kld


    def _get_kld(self, true_mean, true_logstd, prior_params):
        # Function for cleaning up KL divergence calls
        kld = kl_divergence(true_mean, true_logstd, prior_params[0], prior_params[1])
        return kld


    def get_target_assignment(self, hard=False):
        # Returns psi, either 'hard' (one-hot, e.g. for triplet eval) or 'soft' (probabilities, e.g. for debug)
        if not hard:
            return torch.softmax(self.target_params, dim=-1)
        else:
            return F.one_hot(torch.argmax(self.target_params, dim=-1), num_classes=self.target_params.shape[-1])