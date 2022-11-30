import torch.nn as nn
import torch

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
    def __init__(self, n_latents, n_causal_vars, temp=1.0):
        super().__init__()
        self.params = nn.Parameter(torch.randn(n_latents, n_causal_vars))
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
    def __init__(self, causal_assignment_net, n_latents, n_causal_vars, n_hid_per_latent):
        super().__init__()
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

        ## Note add psi lambda reg?

        ## Marginalize over all possible assignments
        causal_assign_probs = self.causal_assignment_net.get_softmax_dist()
        kl = (kl * causal_assign_probs.permute(1, 0).unsqueeze(0))
        return kl.sum(dim=[1,2])

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