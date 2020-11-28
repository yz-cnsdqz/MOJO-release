import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from models.mlp import MLP
from models.rnn import RNN
from utils.torch import *
import pdb
import scipy.io as sio
import math

class VAEDCT(nn.Module):
    def __init__(self, nx, ny, nz, horizon, specs):
        super(VAEDCT, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.horizon = horizon
        self.rnn_type = rnn_type = specs.get('rnn_type', 'lstm')
        self.x_birnn = x_birnn = specs.get('x_birnn', True)
        self.e_birnn = e_birnn = specs.get('e_birnn', True)
        self.use_drnn_mlp = specs.get('use_drnn_mlp', False)
        self.residual = specs.get('residual', False)
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 256)
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        # encode
        self.x_rnn = RNN(nx, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type)
        self.e_rnn = RNN(ny, nh_rnn, bi_dir=e_birnn, cell_type=rnn_type)
        self.q_bias = specs.get('posteriorbias', True)
        self.e_mlp = MLP(2*nh_rnn, nh_mlp, use_bias=self.q_bias)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz, bias=self.q_bias)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz, bias=self.q_bias)
        # decode
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nh_rnn, nh_mlp + [nh_rnn], activation='tanh')
        self.d_rnn = RNN(ny + nz + nh_rnn, nh_rnn, cell_type=rnn_type)
        self.d_mlp = MLP(nh_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, ny)
        self.d_rnn.set_mode('step')

        # load freq matrix
        self.freqbasis = specs.get('freqbasis', 'dct')
        if self.freqbasis == 'dct':
            dct_mat = torch.FloatTensor(sio.loadmat('dct_{:d}.mat'.format(horizon))['D'])
            self.register_buffer('dct_mat',dct_mat)
        elif self.freqbasis == 'dct_adaptive':
            dct_mat = nn.Parameter(torch.tensor(sio.loadmat('dct_50.mat')['D']))
            self.register_parameter('dct_mat',dct_mat)
        elif self.freqbasis == 'dft':
            self.d_rnn = RNN(ny + 2*nz + nh_rnn, nh_rnn, cell_type=rnn_type)
            self.d_rnn.set_mode('step')


    def encode_x(self, x):
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x

    def encode_y(self, y):
        if self.e_birnn:
            h_y = self.e_rnn(y).mean(dim=0)
        else:
            h_y = self.e_rnn(y)
        return h_y

    def encode(self, x, y):
        h_x = self.encode_x(x)
        h_y = self.encode_y(y)
        h = torch.cat((h_x.repeat(h_y.shape[0],1,1), h_y), dim=-1) #[t,b,d]
        if 'dct' in self.freqbasis:
            h = torch.einsum('wt,tbd->wbd', self.dct_mat, h) #[w, b, d]
        elif 'dft' in self.freqbasis:
            h = torch.rfft(h.permute(1,2,0), 1, onesided=False).permute(2,3,0,1)

        h = self.e_mlp(h)
        return self.e_mu(h), self.e_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z):
        h_x = self.encode_x(x)
        if 'dct' in self.freqbasis:
            z = torch.einsum('tw,wbd->tbd',self.dct_mat.T, z)
        elif 'dft' in self.freqbasis:
            z = torch.ifft(z.permute(2,3,0,1), 1).permute(2,0,1,3) #[t,b,d,2]
            z = z.reshape(z.shape[0], z.shape[1], -1)#[t,b,2d]
        if self.use_drnn_mlp:
            h_d = self.drnn_mlp(h_x)
            self.d_rnn.initialize(batch_size=z.shape[1], hx=h_d)
        else:
            self.d_rnn.initialize(batch_size=z.shape[1])
        y = []
        for i in range(self.horizon):
            y_p = x[-1] if i == 0 else y_i
            rnn_in = torch.cat([h_x, z[i], y_p], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            y_i = self.d_out(h)
            if self.residual:
                y_i += y_p
            y.append(y_i)
        y = torch.stack(y)
        return y

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(x, z), mu, logvar


    def sample_prior(self, x, mode = 'iid'):
        if mode == 'iid':
            if 'dft' in self.freqbasis:
                z = np.random.randn(self.horizon,2,x.shape[1], self.nz)
            else:
                z = np.random.randn(self.horizon,x.shape[1], self.nz)
            z = torch.FloatTensor(z).to(x.device)


        return self.decode(x, z)





class NFDiag(nn.Module):
    def __init__(self, nx, ny, nk, specs):
        super(NFDiag, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nk = nk
        self.nh = nh = specs.get('nh_mlp', [300, 200])
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 256)
        self.rnn_type = rnn_type = specs.get('rnn_type', 'gru')
        self.x_birnn = x_birnn = specs.get('x_birnn', False)
        self.fix_first = fix_first = specs.get('fix_first', False)
        self.nac = nac = nk - 1 if fix_first else nk
        self.n_freq = specs.get('n_freq', 10)
        self.x_rnn = RNN(nx, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type)
        self.mlp = MLP(nh_rnn, nh)
        self.freq_separate = specs.get('freq_separate', False)
        self.robustkl = specs.get('robustkl', False)

        if not self.freq_separate:
            self.head_A = nn.Linear(nh[-1], ny * nac)
            self.head_b = nn.Linear(nh[-1], ny * nac)
        else:
            self.head_A = nn.ModuleList([nn.Linear(nh[-1], ny * nac) for i in range(self.n_freq)])
            self.head_b = nn.ModuleList([nn.Linear(nh[-1], ny * nac) for i in range(self.n_freq)])

    def encode_x(self, x):
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x

    def encode(self, x, y):
        if self.fix_first:
            z = y
        else:
            h_x = self.encode_x(x)
            h = self.mlp(h_x)
            a = self.head_A(h).view(-1, self.nk, self.ny)[:, 0, :]
            b = self.head_b(h).view(-1, self.nk, self.ny)[:, 0, :]
            z = (y - b) / a
        return z

    def forward(self, x, z=None):
        h_x = self.encode_x(x)
        if z is None:
            z = torch.randn((self.n_freq, h_x.shape[0], self.ny), device=x.device)
        z = z.repeat_interleave(self.nk, dim=1)
        h = self.mlp(h_x)
        if self.freq_separate:
            a_list = []
            b_list = []
            if self.fix_first:
                for ff in range(self.n_freq):
                    a = self.head_A[ff](h).view(-1, self.nac, self.ny)
                    b = self.head_b[ff](h).view(-1, self.nac, self.ny)
                    a = torch.cat((ones(h_x.shape[0], 1, self.ny, device=x.device), a), dim=1).view(-1, self.ny)
                    b = torch.cat((zeros(h_x.shape[0], 1, self.ny, device=x.device), b), dim=1).view(-1, self.ny)
                    a_list.append(a)
                    b_list.append(b)
            else:
                for ff in range(self.n_freq):
                    a = self.head_A[ff](h).view(-1, self.ny)
                    b = self.head_b[ff](h).view(-1, self.ny)
                    a_list.append(a)
                    b_list.append(b)
            y = torch.exp(0.5*torch.stack(a_list,dim=0)) * z + torch.stack(b_list,dim=0)
            return y, a_list, b_list
        else:
            if self.fix_first:
                a = self.head_A(h).view(-1, self.nac, self.ny)
                b = self.head_b(h).view(-1, self.nac, self.ny)
                a = torch.cat((ones(h_x.shape[0], 1, self.ny, device=x.device), a), dim=1).view(-1, self.ny)
                b = torch.cat((zeros(h_x.shape[0], 1, self.ny, device=x.device), b), dim=1).view(-1, self.ny)
            else:
                a = self.head_A(h).view(-1, self.ny)
                b = self.head_b(h).view(-1, self.ny)
            y = torch.exp(0.5*a.unsqueeze(0)) * z + b.unsqueeze(0)
            return y, a, b


    def sample(self, x, z=None):
        return self.forward(x, z)[0]

    def get_kl(self, a, b):
        if not self.robustkl:
            KLD = 0.5 * torch.sum(-1 - a + b.pow(2) + a.exp()) # original
        else:
            KLD = 0.5 *  (torch.sqrt( 1+ torch.sum(-1 - a + b.pow(2) + a.exp()).pow(2))-1) # robust kl

        return KLD



def get_vae_model(cfg, traj_dim):
    specs = cfg.vae_specs
    model_name = specs.get('model_name', 'VAEDCT')
    if model_name == 'VAEDCT':
        return VAEDCT(traj_dim, traj_dim, cfg.nz, cfg.t_pred, specs)



def get_dlow_model(cfg, traj_dim):
    specs = cfg.dlow_specs
    model_name = specs.get('model_name', 'NFDiag')
    if model_name == 'NFDiag':
        return NFDiag(traj_dim, cfg.nz, cfg.nk, specs)
