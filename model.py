import torch
import torchaudio
from torch import nn
import numpy as np
from modules import *
from modules_GNN import *


class LinkSeg(nn.Module):
    def __init__(self, config, encoder, nb_ssm_classes=3, nb_section_labels=7):
        super(LinkSeg, self).__init__()
        self.config = config
        self.encoder = encoder
        self.hidden_size = 32
        self.output_channels = 16
        self.nb_ssm_classes = nb_ssm_classes
        self.dropout_gnn_features = .5
        self.dropout_gnn_attention = .5
        self.dropout = nn.Dropout(.2)
        self.final_projection = nn.Linear(self.output_channels, nb_ssm_classes)

        self.gnn = GCN_DENSE(in_size = self.hidden_size, hid_size = self.hidden_size, out_size = self.hidden_size, num_layers = 0, dropout = 0.1, norm = False)

        self.mlp = nn.Linear(self.hidden_size, self.hidden_size)

        self.conv = Conv_net_SSM(input_channels = 1, output_channels = self.output_channels, verbose = False, shape = 5, dropout = 0.2)
        
        self.gnn_final = EdgeGAT(in_size=self.hidden_size, feat_size=self.output_channels, heads=8, feat_dropout=self.dropout_gnn_features, attn_dropout=self.dropout_gnn_attention, num_layers=0)

        self.bound_predictor = nn.Linear(self.hidden_size*2+self.output_channels, 1)
        self.class_predictor = nn.Linear(self.hidden_size, nb_section_labels)

        self.pos_embedding = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1500, self.output_channels)))
        self.fc = nn.Sequential(nn.Linear(self.output_channels, self.output_channels))
   
    def forward(self, x):
        x_encoded = self.encoder(x)
        
        N = x_encoded.size(0)
        a = F.normalize(x_encoded, p=2, dim=-1)
        dist = torch.cdist(a, a, p=2).pow(2)
        std = torch.std(dist)
        gamma = -1/(2*std)
        A = dist.mul(gamma).exp()
        
        x_encoded = self.mlp(self.gnn(A, x_encoded)) 

        a = F.normalize(x_encoded, p=2, dim=-1)
        A = torch.cdist(a, a, p=2).pow(2)

        A_conv = self.conv(A.unsqueeze(0)).squeeze(0)
        A_conv = A_conv.permute(1, 2, 0)
        
        A = torch.ones((N, N), device=x.device)
        src, dst = torch.nonzero(A, as_tuple=True)
        
        diff = torch.abs(src-dst) 

        x_time = self.pos_embedding[diff]

        x_pairwise = x_time.reshape(N, N, -1)

        A_conv = self.fc(A_conv) + x_pairwise

        A_pred = self.final_projection(A_conv).squeeze()
        
        edge_feat = A_conv[src, dst]
        
        g = dgl.graph((src, dst), num_nodes=N)
      
        x_gnn_final = self.gnn_final(g, x_encoded, edge_feat)
  
        src_bounds = torch.arange(N-1)
        dst_bounds = torch.arange(1, N)

        link_bounds = A_conv[src_bounds, dst_bounds]
        class_pred = self.class_predictor(x_gnn_final)
        x_bound = torch.cat((x_gnn_final[src_bounds], x_gnn_final[dst_bounds], link_bounds), -1)
        bound_pred = F.sigmoid(self.bound_predictor(x_bound)).squeeze()

        return x_gnn_final, bound_pred, class_pred, A_pred



class Encoder(nn.Module):
    def __init__(
        self,
        config,
        n_mels=64,
        conv_ndim=32,
        sample_rate=22050,
        n_fft=1024,
        f_min=0,
        f_max=11025,
        dropout=0,
        hidden_dim = 32*2,
        attention_ndim = 32*2,
        attention_nlayers = 2,
        attention_nheads = 8,
        verbose=False

    ):
        super(Encoder, self).__init__()

        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                                     n_fft=n_fft,
                                                                     f_min=f_min,
                                                                     f_max=f_max,
                                                                     n_mels=n_mels, 
                                                                     hop_length=config.hop_length,
                                                                     power=2)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.frontend = GenericResFrontEnd(conv_ndim=conv_ndim, nharmonics=1, nmels=64, output_size=attention_ndim, dropout=dropout)
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 16, attention_ndim))
        # transformer
        self.transformer = Transformer(
            attention_ndim,
            attention_nlayers,
            attention_nheads,
            attention_ndim // 2,
            attention_ndim,
            dropout,
        )
        self.verbose = verbose
        self.dropout = nn.Dropout(dropout)
        self.mlp_head = nn.Linear(attention_ndim, hidden_dim)

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): (batch, time)

        Returns:
            x (torch.Tensor): (batch, n_seq_cls)

        """
        # Input preprocessing
        x = self.spec(x)
        x = self.amplitude_to_db(x)
        x = x.unsqueeze(1)
        if self.verbose:
            print('shape before cnn', x.shape)

        # Input embedding
        x = self.frontend(x)
        x += self.pos_embedding[:, : x.size(1)]
        x = self.dropout(x)
        if self.verbose:
            print('shape after cnn', x.shape)

        # transformer
        x = self.transformer(x)
        if self.verbose:
            print('shape after transformer', x.shape)
        x = x.mean(1).squeeze(1)
        if self.verbose:
            print('shape before mlp head', x.shape)
        x = self.mlp_head(x)
        if self.verbose:
            print('shape after mlp head', x.shape)
        return x


class Backbone(nn.Module):
    def __init__(self, config, verbose=False):
        super(Backbone, self).__init__()

        self.front_end = Encoder(config,
        n_mels=64,
        sample_rate=22050,
        n_fft=1024,
        conv_ndim=32,
        f_min=0,
        f_max=11025,
        dropout=0.1,
        hidden_dim = 32,
        attention_ndim = 32,
        attention_nlayers = 2,
        attention_nheads = 8,
        verbose=verbose)


    def forward(self, x):
        x = self.front_end(x)
        return x