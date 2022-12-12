import torch.nn as nn
import torch
import torch.nn.functional as F
from models.utils import gaussian_log_prob

def kl_divergence_normal(mu1, mu2, log_std1, log_std2):
    return log_std2 - log_std1 + (torch.exp(2*log_std1) + torch.square(mu1 - mu2))/(2*torch.exp(2*log_std2)) - 0.5

def kl_divergence_normal(mu1, mu2, log_std1, log_std2):
    return log_std2 - log_std1 + (torch.exp(2*log_std1) + torch.square(mu1 - mu2))/(2*torch.exp(2*log_std2)) - 0.5

class MaskedLinear(nn.Linear):
    def __init__(self, n_blocks, block_in_dim, block_out_dim, diag=False):
        super().__init__(in_features=n_blocks*block_in_dim, out_features=n_blocks*block_out_dim, bias=True)
        mask = torch.zeros_like(self.weight)
        for i in range(n_blocks):
            block_out_start = i * block_out_dim
            block_out_end = (i + 1) * block_out_dim
            cumulative_block_end = i + 1 if diag else i
            cumulative_block_end *= block_in_dim
            mask[block_out_start:block_out_end + 1, :cumulative_block_end] = 1
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        return nn.functional.linear(x, self.weight*self.mask, self.bias)

class CausalAssignmentNet(nn.Module):
    def __init__(self, lambda_reg, n_latents, n_causal_vars, temp=1.0):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(n_latents, n_causal_vars))
        if lambda_reg <= 0:
            self.params.data[:,-1] = -9e15
        self.temp = temp

    def forward(self, batch_size, seq_len=None):
        if seq_len:
            exp_dims = [batch_size, seq_len, -1, -1]
        else:
            exp_dims = [batch_size, -1, -1]
        return nn.functional.gumbel_softmax(self.params.expand(exp_dims), 
                                       tau=self.temp, hard=True)
    
    def get_softmax_dist(self):
        return torch.softmax(self.params, dim=-1)

class AutoregressivePrior(nn.Module):
    def __init__(self, causal_assignment_net, n_latents, n_causal_vars, n_hid_per_latent, lambda_reg, imperfect_interventions):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.imperfect_interventions = imperfect_interventions
        self.causal_assignment_net = causal_assignment_net
        ## Encodes z_t
        self.context_encoder = nn.Linear(n_latents, n_hid_per_latent*n_latents)
        ## Encodes I_t
        self.intervention_encoder = nn.Linear(n_causal_vars, n_hid_per_latent*n_latents)
        ## Autoregressively encodes z_t+1
        self.sample_encoder = MaskedLinear(n_blocks=n_latents, block_in_dim=2, block_out_dim=n_hid_per_latent, diag=False)
        ## Autoregressively decodes p(z_t+1|z_t, I_t)
        self.decoder = nn.Sequential(
                nn.SiLU(),
                MaskedLinear(n_blocks=n_latents, block_in_dim=n_hid_per_latent, block_out_dim=n_hid_per_latent, diag=True),
                nn.SiLU(),
                MaskedLinear(n_blocks=n_latents, block_in_dim=n_hid_per_latent, block_out_dim=2, diag=True),
            )
        self.register_buffer('intrv_mask', torch.eye(n_causal_vars).unsqueeze(0))

    def forward(self, z_t, z_t1, intrv, causal_assignments):
        """
            z_t : torch.FloatTensor, shape [batch_size, n_latents]
            z_t1 : torch.FloatTensor, shape [batch_size, n_latents]
            intrv : torch.FloatTensor, shape [batch_size, n_causal_vars]
            causal_assignments : torch.FloatTensor, shape [batch_size, n_causal_vars, n_latents]
        """
        context_emb = self.context_encoder(z_t).unsqueeze(1) ## [batch_size, 1, n_latents*n_hid_per_latent]
        ## With perfect interventions, we don't need context for intervened variable
        if not self.imperfect_interventions:
            context_emb = context_emb * (1 - intrv.unsqueeze(-1))

        ## Transform intervention vector to matrix where off-diagonal elements are masked (set to -1)
        ## e.g. [0, 1] -> [[0, -1]
        ##                 [-1, 1]]
        masked_intrv = (intrv.unsqueeze(1) * self.intrv_mask) - (1 - self.intrv_mask)
        intrv_emb = self.intervention_encoder(masked_intrv) ## [batch_size, n_causal_vars, n_latents*n_hid_per_latent]

        ## Mask z_t1 samples according to causal assignments
        masked_z_t1 = z_t1.unsqueeze(1) * causal_assignments
        masked_z_t1 = torch.stack([masked_z_t1, 2*causal_assignments-1], dim=-1)
        masked_z_t1 = masked_z_t1.flatten(2)
        sample_emb = self.sample_encoder(masked_z_t1)

        ## Note, try separate mu, logstd decoders once working. 
        combined_emb = (context_emb + intrv_emb + sample_emb) / 2.0
        pred_params = self.decoder(combined_emb)
        pred_params = pred_params.unflatten(-1, (z_t.shape[-1], 2)) ## [batch_size, n_causal_vars, num_latents, 2]
        pred_mu, pred_logstd = pred_params.unbind(dim=-1)
        return pred_mu, pred_logstd

    def compute_kl_loss(self, z_t, intrv, z_t1_mean, z_t1_logstd, z_t1_samples):
        ## Sample latent-to-causal assignments
        batch_size = intrv.shape[0]
        causal_assignments = self.causal_assignment_net(batch_size) # [batch_size, n_latents, n_causal_vars]
        causal_assignments = causal_assignments.permute(0, 2, 1)

        ## Add extra variable to intervention one hots to capture noise
        intrv = torch.cat([intrv, intrv.new_zeros(intrv.shape[0], 1)], dim=-1)

        ## Predict mu, sigma params of p(z_t1|z_t, I_t)
        prior_mu, prior_logstd = self(z_t=z_t, z_t1=z_t1_samples, intrv=intrv, causal_assignments=causal_assignments)

        ## Compute KL(q(z_t1|x_t1)||p(z_t1|z_t, I_t))
        kl = kl_divergence_normal(mu1=z_t1_mean.unsqueeze(1), mu2=prior_mu, log_std1=z_t1_logstd.unsqueeze(1), log_std2=prior_logstd)

        ## Marginalize over all possible assignments
        causal_assign_probs = self.causal_assignment_net.get_softmax_dist()

        ## Note add psi lambda reg?
        if self.lambda_reg > 0.0:
            causal_assign_probs = torch.cat([causal_assign_probs[:,-1:], causal_assign_probs[:,:-1] * (1 - self.lambda_reg)], dim=-1)

        kl = (kl * causal_assign_probs.permute(1, 0).unsqueeze(0))
        return kl.sum(dim=[1,2])

    def sample_based_nll(self, z_t, z_t1, target):
        batch_size, num_samples, _ = z_t.shape

        target_oh = target
        
        # Sample a latent-to-causal assignment from psi
        target_samples = self.causal_assignment_net(batch_size*num_samples)
        # Add sample dimension and I_0=0 to the targets
        target_exp = target_oh[:,None].expand(-1, num_samples, -1).flatten(0, 1)
        target_exp = torch.cat([target_exp, target_exp.new_zeros(batch_size * num_samples, 1)], dim=-1)
        # target_prod = target_exp[:,None,:] * target_samples - (1 - target_samples)

        # # Obtain estimated prior parameters for p(z^t1|z^t,I^t+1)
        # prior_params = self._get_prior_params(z_t.flatten(0, 1), 
        #                                       target_samples=target_samples, 
        #                                       target=target_exp,
        #                                       target_prod=target_prod,
        #                                       z_t1=z_t1.flatten(0, 1))
        prior_params = self(z_t=z_t.flatten(0, 1), z_t1=z_t1.flatten(
            0, 1), intrv=target_exp, causal_assignments=target_samples.permute(0, 2, 1))
        prior_mean, prior_logstd = [p.unflatten(0, (batch_size, num_samples)) for p in prior_params]
        # prior_mean - shape [batch_size, num_samples, num_blocks, num_latents]
        # prior_logstd - shape [batch_size, num_samples, num_blocks, num_latents]
        # z_t1 - shape [batch_size, num_samples, num_latents]
        z_t1 = z_t1[:,:,None,:]  # Expand by block dimension
        nll = -gaussian_log_prob(prior_mean[:,:,None,:,:], prior_logstd[:,:,None,:,:], z_t1[:,None,:,:,:])
        # We take the mean over samples, both over the z^t and z^t+1 samples.
        nll = nll.mean(dim=[1, 2])  # shape [batch_size, num_blocks, num_latents]
        # Marginalize over target assignment

        causal_assign_probs = self.causal_assignment_net.get_softmax_dist()

        nll = nll * causal_assign_probs.permute(1, 0)[None]
        nll = nll.sum(dim=[1, 2])  # shape [batch_size]

        if self.lambda_reg > 0.0 and self.training:
            # target_params_soft = torch.softmax(self.target_params, dim=-1)
            nll = nll + self.lambda_reg * \
                (1-causal_assign_probs[:, -1]).mean(dim=0)
        return nll

    ## TODO: Rewrite
    def get_target_assignment(self, hard=False):
        # Returns psi, either 'hard' (one-hot, e.g. for triplet eval) or 'soft' (probabilities, e.g. for debug)
        if not hard:
            return torch.softmax(self.causal_assignment_net.params, dim=-1)
        else:
            return F.one_hot(torch.argmax(self.causal_assignment_net.params, dim=-1), num_classes=self.causal_assignment_net.params.shape[-1])

if __name__ == '__main__':
    x = torch.rand(4)
    ml = MaskedLinear(n_blocks=2, block_in_dim=2, block_out_dim=3, diag=True)
    print(ml.weight)
    print(ml.bias)
    print(ml.mask)
    print(x)
    print(ml(x))

    causal_ass_net = CausalAssignmentNet(32, 8, 4 + 1, 1)
    prior = AutoregressivePrior(causal_ass_net, 8, 4 + 1, 3)

    prior.compute_kl_loss(
        z_t=torch.randn(32, 8), 
        intrv=nn.functional.one_hot(torch.randint(0, 4, size=(32,))), 
        z_t1_mean=torch.randn(32, 8), 
        z_t1_logstd=torch.randn(32, 8), 
        z_t1_samples=torch.randn(32, 8))
