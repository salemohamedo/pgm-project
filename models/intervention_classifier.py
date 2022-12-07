import torch
import torch.nn as nn
from copy import deepcopy
from models.autoregressive_prior import CausalAssignmentNet

class InterventionClassifier(nn.Module):
    def __init__(self, causal_assignment_net, n_latents, n_causal_vars, hidden_dim, momentum, use_norm):
        super().__init__()
        norm = lambda c: (nn.LayerNorm(c) if use_norm else nn.Identity())
        self.causal_assignment_net = causal_assignment_net
        self.n_causal_vars = n_causal_vars
        self.momentum = momentum

        self.intrv_classifier = nn.Sequential(
            nn.Linear(3*n_latents, 2*hidden_dim),
            norm(2*hidden_dim),
            nn.SiLU(),
            nn.Linear(2*hidden_dim, self.n_causal_vars)
        )

        ## Exponentially moving classifier
        self.exp_intrv_classifier = deepcopy(self.intrv_classifier)
        for n, p in self.exp_intrv_classifier.named_parameters():
            p.requires_grad = False

        ## Store running marginals p(I_t_i = 1) for all causal vars
        self.register_buffer('intrv_marginal', torch.zeros(n_causal_vars).fill_(0.5))

        ## Used to calculate moving average
        self.n_training_steps = 0.
        

    def _update_moving_net(self):
        with torch.no_grad():
            for p_new, p_old in zip(self.intrv_classifier.parameters(), self.exp_intrv_classifier.parameters()):
                p_old.data.mul_(self.momentum).add_(p_new.data * (1 - self.momentum)) 

    def _update_intrv_marginals(self, intrv_targets):
        if self.n_training_steps < 1e6: ## following authors, we should have good estimate by this point
            new_marginals = intrv_targets.flatten(0, 1).mean(dim=0, dtype=float)
            self.intrv_marginal = (self.intrv_marginal*self.n_training_steps + new_marginals)
            self.intrv_marginal /= self.n_training_steps + 1
            self.n_training_steps += 1

    def forward(self, z_samples, intrv_targets):
        """
            z_samples : torch.FloatTensor, shape [batch_size, seq_len, n_latents]
            intrv_targets : torch.FloatTensor, shape [batch_size, seq_len-1,  n_causal_vars]
        """
        if self.training:
            self._update_moving_net()
            self._update_intrv_marginals(intrv_targets)

        batch_size, seq_len, _ = z_samples.shape
        seq_len -= 1

        ## We want to consider one assignment for noise latents, n_causal_vars (one for each var), 
        ## one where we consider all latents
        n_assignments = self.n_causal_vars + 2
        intrv_targets = intrv_targets.unsqueeze(2).expand(-1, -1,  n_assignments, -1).flatten(0, 2)
        z_samples = z_samples.unsqueeze(2).expand(-1, -1,  n_assignments, -1)

        ## Sample latent-to-causal assignments
        causal_assignment = self.causal_assignment_net(batch_size=batch_size, seq_len=seq_len)
        # causal_assignment = nn.functional.gumbel_softmax(transition_prior.target_params[None].expand(batch_size, seq_len, -1, -1), 
        #                                      tau=1.0, hard=True)
        ## Add extra variable assigned to all latents
        causal_assignment = torch.cat([causal_assignment, causal_assignment.new_ones(causal_assignment.shape[:-1] + (1,))], dim=-1)
        causal_assignment = causal_assignment.permute(0, 1, 3, 2) ## [batch_size, seq_len, n_assignments, n_latents]

        ## Calculate cross entropy loss of p(I_t+1|z_t, z_t1_i, mask_i) for all assignments i
        z_samples_no_grad = z_samples.detach()
        causal_assignment_no_grad = causal_assignment.detach()
        loss1_input = torch.cat([z_samples_no_grad[:,:-1,:], z_samples_no_grad[:,1:,:] * causal_assignment_no_grad, causal_assignment_no_grad], dim=-1)
        loss1_input = loss1_input.flatten(0, 2)
        out = self.intrv_classifier(loss1_input)
        loss1 = nn.functional.binary_cross_entropy_with_logits(out, intrv_targets, reduction='none')
        loss1 = n_assignments * seq_len * loss1.mean()

        ## Binary cross entropy loss mask
        loss_mask = torch.cat(
            (torch.eye(self.n_causal_vars), 
            torch.zeros(1, self.n_causal_vars), 
            torch.ones(1, self.n_causal_vars)))
        loss_mask = loss_mask.unsqueeze(0).expand(batch_size*seq_len, -1, -1).flatten(0, 1).bool()
        loss_mask = loss_mask.to(intrv_targets.device)

        ## Calculate 2-part cross entropy loss: 
        ## 1. p(I_t+1_i|z_t, z_t1_i, mask_i) for all assignments i
        ## 2. for all assignments i not equal to j encourage p(I_t+1_j|z_t, z_t1_i, mask_i) to equal empirical marginal p(I_t+1_j)
        loss2_input = torch.cat([z_samples[:,:-1,:], z_samples[:,1:,:] * causal_assignment, causal_assignment], dim=-1)
        loss2_input = loss2_input.flatten(0, 2)
        loss2_outs = self.exp_intrv_classifier(loss2_input)
        
        loss2_targets = torch.where(loss_mask, intrv_targets, self.intrv_marginal.unsqueeze(0).float())
        loss_z = nn.functional.binary_cross_entropy_with_logits(loss2_outs, loss2_targets, reduction='none')
        loss_mask = loss_mask.float()
        loss_z = loss_z * (n_assignments * loss_mask + (1 - loss_mask)) ## Weight self-prediction higher, following authors
        loss2 = n_assignments * seq_len * loss_z.mean()

        return loss1, loss2

if __name__ == '__main__':
    causal_ass_net = CausalAssignmentNet(8, 4 + 1, 1)
    intrv_classifier = InterventionClassifier(causal_ass_net, 8, 4, 3, 0.6)

    print(intrv_classifier(z_samples=torch.randn(4, 3, 8), intrv_targets=torch.randn(4, 2, 4)))

