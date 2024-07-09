from torch import nn
import dgl

    

class GCN_DENSE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers = 2, dropout = 0.2, norm=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.in_size = in_size
        self.hid_size = hid_size
        self.out_size = out_size
        self.diff = False
        self.norm = norm
        if self.norm:
            self.norm_layer = nn.LayerNorm(self.in_size)
        if self.in_size != self.out_size:
            self.fc = nn.Linear(self.in_size, self.out_size)
            self.diff = True
        self.dropout = nn.Dropout(dropout)
        self.layers.append(dgl.nn.pytorch.conv.DenseGraphConv(in_feats=self.in_size, out_feats=self.hid_size, norm='right', bias=True, activation=None))
        for k in range(num_layers):
        
            self.layers.append(dgl.nn.pytorch.conv.DenseGraphConv(in_feats=self.hid_size, out_feats=self.hid_size, norm='right', bias=True, activation=None))

        self.layers.append(dgl.nn.pytorch.conv.DenseGraphConv(in_feats=self.hid_size, out_feats=self.out_size, norm='right', bias=True, activation=None))
        
        self.activation = nn.ELU()
        

    def forward(self, A, features):
        if self.norm:
            h = self.norm_layer(features)
        else:
            h = features
        for i, layer in enumerate(self.layers):
            if i < len(self.layers)-1:
                h = self.activation(layer(A, h)) + h
                self.dropout(h)
            else:
                h = layer(A, h) + h
                self.dropout(h)
        if self.diff:
            h = h + self.fc(features)
        else:
            h = h + features
        return h



class MLP(nn.Module):
    def __init__(self, input_size, output_size, num_layers = 1, hidden_dim=128, dropout=.2, norm=True, bias=False):
        super().__init__()
        self.layers = nn.ModuleList()
        #self.activation = nn.LeakyReLU(negative_slope=0.2)
        if norm:
            self.layers += [
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.ELU(),
            #nn.LeakyReLU(negative_slope=0.2),
            #nn.GELU(),
            nn.Dropout(dropout)]
        else:
            self.layers += [
            nn.Linear(input_size, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            #nn.LayerNorm(hidden_dim),
            nn.ELU(),
            #nn.LeakyReLU(negative_slope=0.2),
            #nn.GELU(),
            nn.Dropout(dropout)]
        for k in range(num_layers):
            if norm:
                self.layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                #nn.LayerNorm(hidden_dim),
                #nn.LeakyReLU(negative_slope=0.2),
                #nn.GELU(),
                nn.ELU(),
                nn.Dropout(dropout)]
            else:
                self.layers += [
                nn.Linear(hidden_dim, hidden_dim),
                #nn.BatchNorm1d(hidden_dim),
                #nn.LayerNorm(hidden_dim),
                #nn.LeakyReLU(negative_slope=0.2),
                #nn.GELU(),
                nn.ELU(),
                nn.Dropout(dropout)]
        if bias:
            L = nn.Linear(hidden_dim, output_size, bias=True)
        else:
            L = nn.Linear(hidden_dim, output_size, bias=False)
        self.layers.append(L)
        self.norm = norm
        if self.norm:
            self.norm_layer = nn.LayerNorm(input_size)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
        return h 



class EdgeGAT(nn.Module):
    def __init__(self, in_size, feat_size, heads, feat_dropout=.1, attn_dropout=.1, num_layers=0, norm=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.ELU()
        self.in_size = in_size 
        self.feat_size = feat_size
        self.heads = heads 
        self.hidden_size = self.in_size
        self.residual = True
        self.norm = norm
        if self.norm:
            self.layer_norm = nn.LayerNorm(self.in_size)
        
        self.layers.append(dgl.nn.pytorch.conv.EdgeGATConv(in_feats=self.in_size, edge_feats=self.feat_size, out_feats=self.hidden_size, num_heads=self.heads, 
                                                        feat_drop=feat_dropout, attn_drop=attn_dropout, negative_slope=0.2, residual=self.residual, activation=None, allow_zero_in_degree=False, bias=True))
        
        for k in range(num_layers):
            self.layers.append(dgl.nn.pytorch.conv.EdgeGATConv(in_feats=self.hidden_size*heads, edge_feats=self.feat_size, out_feats=self.hidden_size, num_heads=self.heads, 
                                                        feat_drop=feat_dropout, attn_drop=attn_dropout, negative_slope=0.2, residual=self.residual, activation=None, allow_zero_in_degree=False, bias=True))
           
        self.layers.append(dgl.nn.pytorch.conv.EdgeGATConv(in_feats=self.hidden_size*heads, edge_feats=self.feat_size, out_feats=self.in_size, num_heads=self.heads, 
                                                        feat_drop=feat_dropout, attn_drop=attn_dropout, negative_slope=0.2, residual=self.residual, activation=None, allow_zero_in_degree=False, bias=True))
        
        
       


    def forward(self, g, features, edge_feat):
        if self.norm:
            h = self.layer_norm(features)
        else:
            h = features
        
        for i, layer in enumerate(self.layers):
            h = layer(g, h, edge_feat) 
            if i < len(self.layers) - 1 :
                h = h.flatten(1)
                h = self.activation(h)
            else:
                h = h.mean(1)
        return h
    