import torch

from model.base import BaseModule
from model.encoder import MelEncoder
from model.postnet import PostNet
from model.diffusion_single import Diffusion as DiffSingle
from model.diffusion_multi import Diffusion as DiffMulti
from model.utils import sequence_mask, fix_len_compatibility, mse_loss


class FwdDiffusion(BaseModule):
    def __init__(self, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size, dim):
        super(FwdDiffusion, self).__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.dim = dim
        self.encoder = MelEncoder(n_feats, channels, filters, heads, layers, 
                                  kernel, dropout, window_size)
        self.postnet = PostNet(dim)

    @torch.no_grad()
    def forward(self, x, mask):
        x, mask = self.relocate_input([x, mask])
        z = self.encoder(x, mask)
        z_output = self.postnet(z, mask)
        return z_output

    def compute_loss(self, x, y, mask):
        x, y, mask = self.relocate_input([x, y, mask])
        z = self.encoder(x, mask)
        z_output = self.postnet(z, mask)
        loss = mse_loss(z_output, y, mask, self.n_feats)
        return loss


class DiffVCSingle(BaseModule):
    def __init__(self, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size, enc_dim, dec_dim, beta_min, beta_max):
        super(DiffVCSingle, self).__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.encoder = FwdDiffusion(n_feats, channels, filters, heads, layers,
                                    kernel, dropout, window_size, enc_dim)
        self.decoder = DiffSingle(n_feats, dec_dim, beta_min, beta_max)

    def load_encoder(self, enc_path):
        enc_dict = torch.load(enc_path, map_location=lambda loc, storage: loc)
        self.encoder.load_state_dict(enc_dict, strict=False)

    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, mode='ml', t_fwd=1.0, t_bwd=1.0):
        x, x_lengths = self.relocate_input([x, x_lengths])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        mean = self.encoder(x, x_mask)
        mean_x = self.decoder.compute_diffused_mean(x, x_mask, mean, t_fwd)

        b = x.shape[0]
        max_length = int(x_lengths.max())
        max_length_new = fix_len_compatibility(max_length)
        x_mask_new = sequence_mask(x_lengths, max_length_new).unsqueeze(1).to(x.dtype)
        mean_new = torch.zeros((b, self.n_feats, max_length_new), dtype=x.dtype, 
                                device=x.device)
        mean_x_new = torch.zeros((b, self.n_feats, max_length_new), dtype=x.dtype, 
                                  device=x.device)
        for i in range(b):
            mean_new[i, :, :x_lengths[i]] = mean[i, :, :x_lengths[i]]
            mean_x_new[i, :, :x_lengths[i]] = mean_x[i, :, :x_lengths[i]]

        z = mean_x_new
        z += torch.randn_like(mean_x_new, device=mean_x_new.device)

        y = self.decoder(z, x_mask_new, mean_new, t_bwd, n_timesteps, mode)
        return mean_x, y[:, :, :max_length]

    def compute_loss(self, x, x_lengths):
        x, x_lengths  = self.relocate_input([x, x_lengths])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        mean = self.encoder(x, x_mask).detach()
        diff_loss = self.decoder.compute_loss(x, x_mask, mean)
        return diff_loss


class DiffVCMulti(BaseModule):
    def __init__(self, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size, enc_dim, spk_dim, use_ref_t, dec_dim, 
                 beta_min, beta_max):
        super(DiffVCMulti, self).__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.enc_dim = enc_dim
        self.spk_dim = spk_dim
        self.use_ref_t = use_ref_t
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.encoder = FwdDiffusion(n_feats, channels, filters, heads, layers,
                                    kernel, dropout, window_size, enc_dim)
        self.decoder = DiffMulti(n_feats, dec_dim, spk_dim, use_ref_t, 
                                 beta_min, beta_max)

    def load_encoder(self, enc_path):
        enc_dict = torch.load(enc_path, map_location=lambda loc, storage: loc)
        self.encoder.load_state_dict(enc_dict, strict=False)

    @torch.no_grad()
    def forward(self, x, x_lengths, x_ref, x_ref_lengths, c, n_timesteps, 
                mode='ml', t_fwd=1.0, t_bwd=1.0):
        x, x_lengths = self.relocate_input([x, x_lengths])
        x_ref, x_ref_lengths, c = self.relocate_input([x_ref, x_ref_lengths, c])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        x_ref_mask = sequence_mask(x_ref_lengths).unsqueeze(1).to(x_ref.dtype)
        mean = self.encoder(x, x_mask)
        mean_x = self.decoder.compute_diffused_mean(x, x_mask, mean, t_fwd)
        mean_ref = self.encoder(x_ref, x_ref_mask)

        b = x.shape[0]
        max_length = int(x_lengths.max())
        max_length_new = fix_len_compatibility(max_length)
        x_mask_new = sequence_mask(x_lengths, max_length_new).unsqueeze(1).to(x.dtype)
        mean_new = torch.zeros((b, self.n_feats, max_length_new), dtype=x.dtype, 
                                device=x.device)
        mean_x_new = torch.zeros((b, self.n_feats, max_length_new), dtype=x.dtype, 
                                  device=x.device)
        for i in range(b):
            mean_new[i, :, :x_lengths[i]] = mean[i, :, :x_lengths[i]]
            mean_x_new[i, :, :x_lengths[i]] = mean_x[i, :, :x_lengths[i]]

        z = mean_x_new
        z += torch.randn_like(mean_x_new, device=mean_x_new.device)

        y = self.decoder(z, x_mask_new, mean_new, x_ref, x_ref_mask, mean_ref, c, 
                         t_bwd, n_timesteps, mode)
        return mean_x, y[:, :, :max_length]

    def compute_loss(self, x, x_lengths, x_ref, c):
        x, x_lengths, x_ref, c = self.relocate_input([x, x_lengths, x_ref, c])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        mean = self.encoder(x, x_mask).detach()
        mean_ref = self.encoder(x_ref, x_mask).detach()
        diff_loss = self.decoder.compute_loss(x, x_mask, mean, x_ref, mean_ref, c)
        return diff_loss
