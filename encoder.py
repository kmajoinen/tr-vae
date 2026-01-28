import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


OUT_DIM = {2: 39, 4: 35, 6: 31}

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, vae=False, tr_proj=False):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.project_tr = tr_proj

        self.prev_mu = None
        self.prev_var = None
        self.prev_std = None
        self.cur_mu = None
        self.cur_logvar = None

        self.eps_mu = 0.03
        self.eps_cov = 0.001

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]

        self.vae = vae
        print(f"VAE: {self.vae}")

        if self.vae:
            self.fc_mu = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
            self.ln_mu = nn.LayerNorm(self.feature_dim)

            self.fc_logvar = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
            self.ln_logvar = nn.LayerNorm(self.feature_dim)
        else:
            self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
            self.ln = nn.LayerNorm(self.feature_dim)


        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        if not self.vae:
            h_fc = self.fc(h)
            self.outputs['fc'] = h_fc

            h_norm = self.ln(h_fc)
            self.outputs['ln'] = h_norm

            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

            return out
        else:
            h_fc_mu = self.fc_mu(h)
            #self.outputs['fc'] = h_fc

            h_norm_mu = self.ln_mu(h_fc_mu)
            #self.outputs['ln'] = h_norm

            h_fc_logvar = self.fc_logvar(h)
            #self.outputs['fc'] = h_fc

            h_norm_logvar = self.ln_logvar(h_fc_logvar)
            #self.outputs['ln'] = h_norm

            if self.project_tr:
                # print("Projecting TR...")
                self.cur_mu = h_norm_mu.detach()
                self.cur_logvar = h_norm_logvar.detach()
                # h_norm_mu, h_norm_logvar = self.project_trust_region_fixed(h_norm_mu, h_norm_logvar)

                # print("Mean proj", h_norm_mu.mean(), "Mean ", self.cur_mu.mean())
                # print("LogVar Proj", h_norm_logvar.mean(), "Logvar ", self.cur_logvar.mean())

            return [h, h_norm_mu, h_norm_logvar]

    def get_tr_loss(self, mu_proj, logvar_proj):

        std = torch.exp(0.5*logvar_proj)
        sqrt = torch.diag_embed(std)
        cur_std = torch.exp(0.5*self.cur_logvar)
        cur_sqrt = torch.diag_embed(cur_std)
        mean_diff, _ = self.gaussian_wasserstein_commutative(self.cur_mu, mu_proj, cur_sqrt, sqrt)
        # if self.cur_logvar is None:
        #     self.cur_logvar = logvar_proj
        #     self.cur_mu = mu_proj
        # std     = torch.exp(0.5 * self.cur_logvar)
        # std_proj = torch.exp(0.5 * logvar_proj)

        # mean_part = ((self.cur_mu - mu_proj.detach()) ** 2).sum(dim=1)
        # cov_part  = ((std - std_proj.detach()) ** 2).sum(dim=1)

        # tr_loss = mean_part + cov_part
        tr_loss = mean_diff

        return tr_loss.mean()

    def projection_loss(self,mu, logvar, mu_proj, logvar_proj):

        std = torch.exp(0.5*logvar_proj)
        sqrt = torch.diag_embed(std)
        cur_std = torch.exp(0.5*logvar)
        cur_sqrt = torch.diag_embed(cur_std)
        mean_diff, _ = self.gaussian_wasserstein_commutative(mu, mu_proj, cur_sqrt, sqrt)
        # if self.cur_logvar is None:
        #     self.cur_logvar = logvar_proj
        #     self.cur_mu = mu_proj
        # std     = torch.exp(0.5 * self.cur_logvar)
        # std_proj = torch.exp(0.5 * logvar_proj)

        # mean_part = ((self.cur_mu - mu_proj.detach()) ** 2).sum(dim=1)
        # cov_part  = ((std - std_proj.detach()) ** 2).sum(dim=1)

        # tr_loss = mean_part + cov_part
        tr_loss = mean_diff
        return tr_loss.mean()

    def project_trust_region_simple(self, mu, logvar):
        if self.prev_mu is None:
            # First iteration: no previous distribution
            self.prev_mu = mu.detach()
            self.prev_var = torch.exp(logvar.detach())
            return mu, logvar

        mu_old = self.prev_mu
        var_old = self.prev_var

        # Convert to std
        var     = torch.exp(logvar)
        std     = torch.sqrt(var)
        std_old = torch.sqrt(var_old)

        # ------------------ W2 mean part ------------------
        delta = mu - mu_old                             # (B,D)
        mean_part = (delta.pow(2) / (var_old + 1e-12)).sum(dim=-1)  # (B,)

        # mask: only project samples that exceed the trust region
        mask_mean = mean_part > self.eps_mu

        # omega multiplier
        omega = torch.zeros_like(mean_part)
        omega[mask_mean] = torch.sqrt(mean_part[mask_mean] / self.eps_mu) - 1.0
        omega = torch.clamp(omega, min=0.0)[:, None]  # (B,1)

        mu_proj = (mu + omega * mu_old) / (1.0 + omega + 1e-12)

        # ------------------ W2 covariance part ------------------
        cov_part = (std - std_old).pow(2).sum(dim=-1)  # (B,)
        mask_cov = cov_part > self.eps_cov

        eta = torch.zeros_like(cov_part)
        eta[mask_cov] = torch.sqrt(cov_part[mask_cov] / self.eps_cov) - 1.0
        eta = torch.clamp(eta, min=0.0)[:, None]  # (B,1)

        std_proj = (std + eta * std_old) / (1.0 + eta + 1e-12)
        logvar_proj = 2.0 * torch.log(std_proj + 1e-12)

        # Save current output as "old" for next iteration
        self.prev_mu  = mu.detach()
        self.prev_var = var.detach()

        return mu_proj, logvar_proj

    def project_trust_region_simple2(self, mu, logvar):

        # Initialize previous distribution on first call
        eps_mean = self.eps_mu
        eps_cov = self.eps_cov
        if self.prev_mu is None:
            self.prev_mu  = mu.detach()
            self.prev_std = torch.exp(0.5 * logvar.detach())
            return mu, logvar

        # -------------------------------------------------
        # Convert to std
        # -------------------------------------------------
        std     = torch.exp(0.5 * logvar)
        mu_old  = self.prev_mu
        std_old = self.prev_std

        # ------------------ W2 mean part ------------------
        mean_part = ((mu - mu_old) ** 2).sum(dim=-1)   # (B,)
        mask_mean = mean_part > eps_mean

        # Lagrange multiplier ω
        omega = torch.zeros_like(mean_part)
        if mask_mean.any():
            omega[mask_mean] = torch.sqrt(mean_part[mask_mean] / eps_mean) - 1.0
        omega = torch.clamp(omega, min=0.0).unsqueeze(-1)   # (B,1)

        # Project mean
        mu_proj = (mu + omega * mu_old) / (1.0 + omega + 1e-12)

        # ------------------ W2 covariance part ------------------
        cov_part = ((std - std_old) ** 2).sum(dim=-1)   # (B,)
        mask_cov = cov_part > eps_cov

        # Lagrange multiplier η
        eta = torch.zeros_like(cov_part)
        if mask_cov.any():
            eta[mask_cov] = torch.sqrt(cov_part[mask_cov] / eps_cov) - 1.0
        eta = torch.clamp(eta, min=0.0).unsqueeze(-1)   # (B,1)

        # Project std
        std_proj = (std + eta * std_old) / (1.0 + eta + 1e-12)

        # Convert std → logvar
        logvar_proj = 2.0 * torch.log(std_proj + 1e-12)

        # ------------------ Update stored "previous" dist ------------------
        self.prev_mu  = mu.detach()
        self.prev_std = std.detach()

        return mu_proj, logvar_proj

    def project_trust_region_fixed(self, mu, logvar):

        # print("Using TR projection")
        # Initialize previous distribution on first call
        std = torch.exp(0.5*logvar)
        # print(f"Projection shapes in: {mu.shape}, {logvar.shape}")
        if self.prev_mu is None or self.prev_mu.shape != mu.shape:
            self.prev_mu = mu.detach()
            self.prev_std = std.detach()
            return mu, logvar

        mean = mu          # THIS is the correct sqrt(cov)
        sqrt = torch.diag_embed(std)            # B×D×D diagonal matrix

        old_mean = self.prev_mu
        old_std = self.prev_std
        old_sqrt = torch.diag_embed(old_std)    # B×D×D

        # ---------------------------------------------------------------
        # 1. W2 mean and cov parts (Bosch commutative, diagonal version)
        # ---------------------------------------------------------------
        mean_part = ((mean - old_mean) ** 2).sum(dim=1)     # (B,)

        cov = sqrt @ sqrt.transpose(-1, -2)                 # gives diag(std^2)
        cov_old = old_sqrt @ old_sqrt.transpose(-1, -2)

        cov_part = torch.diagonal(
            cov_old + cov - 2 * (old_sqrt @ sqrt),
            dim1=-2, dim2=-1
        ).sum(-1)   # (B,)

        # ---------------------------------------------------------------
        # 2. Mean projection
        # ---------------------------------------------------------------
        mask_mean = mean_part > self.eps_mu

        if mask_mean.any():
            omega = torch.ones_like(mean_part)
            omega[mask_mean] = torch.sqrt(mean_part[mask_mean] / self.eps_mu) - 1.0
            # omega = torch.max(omega, min=0.0).unsqueeze(-1)
            omega = torch.max(-omega, omega).unsqueeze(-1)

            mean_proj = (mean + omega * old_mean) / (1.0 + omega + 1e-16)
            mean_proj = torch.where(mask_mean[:, None], mean_proj, mean)
        else:
            mean_proj = mean

        # ---------------------------------------------------------------
        # 3. Covariance projection
        # ---------------------------------------------------------------
        mask_cov = cov_part > self.eps_cov

        if mask_cov.any():
            eta = torch.ones_like(cov_part)
            eta[mask_cov] = torch.sqrt(cov_part[mask_cov] / self.eps_cov) - 1.0
            eta = torch.max(-eta, eta).unsqueeze(-1).unsqueeze(-1)

            sqrt_proj = (sqrt + eta * old_sqrt) / (1.0 + eta + 1e-16)
            sqrt_proj = torch.where(mask_cov[:, None, None], sqrt_proj, sqrt)
        else:
            sqrt_proj = sqrt

        # Convert back to diag logvar
        std_proj = torch.diagonal(sqrt_proj, dim1=-2, dim2=-1)
        logvar_proj = 2.0 * torch.log(std_proj + 1e-12)

        # ---------------------------------------------------------------
        # 4. Save for next iteration
        # ---------------------------------------------------------------
        self.prev_mu = mu.detach()
        self.prev_std = std.detach()
        # print(f"Projection shapes out: {mean_proj.shape}, {logvar_proj.shape}")
        return mean_proj, logvar_proj

    def get_raw_mu_logvar(self):
        return self.cur_mu, self.cur_logvar

    def get_W2_dist(self, mu1, mu2, logvar1, logvar2):

        std1 = torch.exp(0.5*logvar1)
        sqrt1 = torch.diag_embed(std1)
        std2 = torch.exp(0.5*logvar2)
        sqrt2 = torch.diag_embed(std2)
        mean_diff, logvar_diff = self.gaussian_wasserstein_commutative(mu1, mu2, sqrt1, sqrt2)
        return mean_diff, logvar_diff

    def tr_projection(self, mu1, mu2, logvar1, logvar2, eps_mu, eps_cov):
        # print(f"Projection shapes in: {mu.shape}, {logvar.shape}")
        # std = torch.exp(0.5*logvar)
        # if self.prev_mu is not None:
        #     print("Shapes: ", mu.shape, self.prev_mu.shape)
        # if self.prev_mu is None or self.prev_mu.shape != mu.shape:
        #     self.prev_mu = mu.detach()
        #     # self.prev_var = torch.exp(0.5*logvar.detach())
        #     # self.prev_std = torch.exp(0.5 * logvar.detach())
        #     self.prev_std = std.detach()
        #     return mu, logvar

        # print("Projecting TR")
        # print(f"eps_mu {eps_mu} - eps_cov {eps_cov}")
        mean = mu1
        # sqrt = torch.exp(0.5*logvar)
        std = torch.exp(0.5*logvar1)
        sqrt = torch.diag_embed(std)


        old_mean = mu2
        # old_sqrt = self.prev_var
        old_std  = torch.exp(0.5*logvar2)
        old_sqrt = torch.diag_embed(old_std)

        batch_shape = mean.shape[:-1]

        mean_part, cov_part = self.gaussian_wasserstein_commutative(mean, old_mean, sqrt, old_sqrt)

        proj_mean = self.mean_projection(mean, old_mean, mean_part, eps_mu)

        cov_mask = cov_part > eps_cov

        if cov_mask.any():
            # gradient issue with ch.where, it executes both paths and gives NaN gradient.
            eta = torch.ones(batch_shape, dtype=sqrt.dtype, device=sqrt.device)
            eta[cov_mask] = torch.sqrt(cov_part[cov_mask] / eps_cov) - 1.
            # eta = torch.max(-eta, eta)
            eta = torch.clamp(eta, min=0.0)[..., None, None]

            # new_sqrt = (sqrt + torch.einsum('i,ijk->ijk', eta, old_sqrt)) / (1. + eta + 1e-16)[..., None, None]
            new_sqrt = (sqrt + eta * old_sqrt) / (1.0 + eta + 1e-16)

            proj_sqrt = torch.where(cov_mask[..., None, None], new_sqrt, sqrt)
        else:
            proj_sqrt = sqrt

        std_proj = torch.diagonal(proj_sqrt, dim1=-2, dim2=-1)       # <-- FIXED
        logvar_proj = 2.0 * torch.log(std_proj + 1e-12)
        # self.prev_mu  = mu.detach()
        # self.prev_var = sqrt.detach()
        # self.prev_mu  = mu.detach()
        # self.prev_std = std.detach()
        # print(f"Projection shapes out: {proj_mean.shape}, {logvar_proj.shape}")
        return proj_mean, logvar_proj

    def project_trust_region(self, mu, logvar):
        # print(f"Projection shapes in: {mu.shape}, {logvar.shape}")
        std = torch.exp(0.5*logvar)
        # if self.prev_mu is not None:
        #     print("Shapes: ", mu.shape, self.prev_mu.shape)
        if self.prev_mu is None or self.prev_mu.shape != mu.shape:
            self.prev_mu = mu.detach()
            # self.prev_var = torch.exp(0.5*logvar.detach())
            # self.prev_std = torch.exp(0.5 * logvar.detach())
            self.prev_std = std.detach()
            return mu, logvar

        # print("Projecting TR")
        mean = mu
        # sqrt = torch.exp(0.5*logvar)
        std = torch.exp(0.5*logvar)
        sqrt = torch.diag_embed(std)


        old_mean = self.prev_mu
        # old_sqrt = self.prev_var
        old_std  = self.prev_std
        old_sqrt = torch.diag_embed(old_std)

        batch_shape = mean.shape[:-1]

        ####################################################################################################################
        # precompute mean and cov part of W2, which are used for the projection.
        # Both parts differ based on precision scaling.
        # If activated, the mean part is the maha distance and the cov has a more complex term in the inner parenthesis.
        mean_part, cov_part = self.gaussian_wasserstein_commutative(mean, old_mean, sqrt, old_sqrt)


        ####################################################################################################################
        # project mean (w/ or w/o precision scaling)
        proj_mean = self.mean_projection(mean, old_mean, mean_part)

        ####################################################################################################################
        # project covariance (w/ or w/o precision scaling)

        cov_mask = cov_part > self.eps_cov

        if cov_mask.any():
            # gradient issue with ch.where, it executes both paths and gives NaN gradient.
            eta = torch.ones(batch_shape, dtype=sqrt.dtype, device=sqrt.device)
            eta[cov_mask] = torch.sqrt(cov_part[cov_mask] / self.eps_cov) - 1.
            # eta = torch.max(-eta, eta)
            eta = torch.clamp(eta, min=0.0)[..., None, None]

            # new_sqrt = (sqrt + torch.einsum('i,ijk->ijk', eta, old_sqrt)) / (1. + eta + 1e-16)[..., None, None]
            new_sqrt = (sqrt + eta * old_sqrt) / (1.0 + eta + 1e-16)

            proj_sqrt = torch.where(cov_mask[..., None, None], new_sqrt, sqrt)
        else:
            proj_sqrt = sqrt

        std_proj = torch.diagonal(proj_sqrt, dim1=-2, dim2=-1)       # <-- FIXED
        logvar_proj = 2.0 * torch.log(std_proj + 1e-12)
        # self.prev_mu  = mu.detach()
        # self.prev_var = sqrt.detach()
        self.prev_mu  = mu.detach()
        self.prev_std = std.detach()
        # print(f"Projection shapes out: {proj_mean.shape}, {logvar_proj.shape}")
        return proj_mean, logvar_proj

    def mean_projection(self, mean, old_mean, maha, eps_mu):
        batch_shape = mean.shape[:-1]
        mask = maha > eps_mu

        if mask.any():
            # omega = torch.ones(batch_shape, dtype=mean.dtype, device=mean.device)
            omega = torch.ones_like(maha)
            # omega[mask] = torch.sqrt(maha[mask] / self.eps_mu) - 1.
            omega[mask] = torch.sqrt(maha[mask] / eps_mu) - 1.0
            # omega = torch.max(-omega, omega)[..., None]
            omega = torch.max(-omega, omega)[..., None]


            m = (mean + omega * old_mean) / (1 + omega + 1e-16)
            proj_mean = torch.where(mask[..., None], m, mean)
        else:
            proj_mean = mean

        return proj_mean

    def gaussian_wasserstein_commutative(self, mean, mean_other, sqrt, sqrt_other):
        def torch_batched_trace(x):
            return torch.diagonal(x, dim1=-2, dim2=-1).sum(-1)
        def mean_distance(mean, mean_other, std_other=None):
            mean_part = ((mean_other - mean) ** 2).sum(1)
            return mean_part


        # mean_part = mean_distance(mean, mean_other)
        mean_part = ((mean - mean_other) ** 2).sum(dim=1)

        # cov = sqrt.pow(2)
        # cov_other = sqrt_other.pow(2)
        cov       = sqrt @ sqrt.transpose(-1, -2)                    # <-- FIXED
        cov_other = sqrt_other @ sqrt_other.transpose(-1, -2)

        # cov_part = torch_batched_trace(cov_other + cov - 2 * sqrt_other @ sqrt)
        term = cov_other + cov - 2 * (sqrt_other @ sqrt)             # <-- FIXED
        cov_part = torch.diagonal(term, dim1=-2, dim2=-1).sum(-1)    # <-- FIXED

        return mean_part, cov_part

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, vae):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, vae, tr_proj
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, vae, tr_proj
    )
