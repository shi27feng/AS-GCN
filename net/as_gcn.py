import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.autograd import Variable

from net.utils.graph import Graph


class Model(nn.Module):

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.edge_type = 2

        temporal_kernel_size = 9
        spatial_kernel_size = A.size(0) + self.edge_type
        st_kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        dims = [in_channels] + [64] * 3 + [128] * 3 + [256] * 3
        self.class_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            residual = False if i is 0 else True
            stride = 2 if ((i is 3) or (i is 6)) else 1
            self.class_layers.append(
                StgcnBlock(dims[i],
                           dims[i + 1],
                           st_kernel_size,
                           self.edge_type,
                           stride=stride,
                           residual=residual, **kwargs)
            )

        self.recon_layers = nn.ModuleList()
        dims = [255] + [128] * 4
        for i in range(len(dims) - 1):
            self.recon_layers.append(
                StgcnBlock(dims[i], dims[i + 1], st_kernel_size, self.edge_type, stride=1, **kwargs)
            )
        self.recon_layers.append(StgcnBlock(128, 128, (3, spatial_kernel_size), self.edge_type, stride=2, **kwargs))
        self.recon_layers.append(StgcnBlock(128, 128, (5, spatial_kernel_size),
                                            self.edge_type,
                                            stride=1, padding=False, residual=False, **kwargs))
        self.recon_layer_6 = StgcnReconBlock(128 + 3, 30, (1, spatial_kernel_size),
                                             self.edge_type, stride=1, padding=False,
                                             residual=False, activation=None, **kwargs)

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(self.A.size()))
                                                     for i in range(len(dims) - 1)])
            self.edge_importance_recon = nn.ParameterList([nn.Parameter(torch.ones(self.A.size()))
                                                           for i in range(len(dims) - 1)])
        else:
            self.edge_importance = [1] * (len(self.st_gcn_networks) + len(self.st_gcn_recon))
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x, x_target, x_last, A_act, lamda_act):

        N, C, T, V, M = x.size()
        x_recon = x[:, :, :, :, 0]  # [2N, 3, 300, 25]
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # [N, 2, 25, 3, 300]
        x = x.view(N * M, V * C, T)  # [2N, 75, 300]

        x_last = x_last.permute(0, 4, 1, 2, 3).contiguous().view(-1, 3, 1, 25)

        x_bn = self.data_bn(x)
        x_bn = x_bn.view(N, M, V, C, T)
        x_bn = x_bn.permute(0, 1, 3, 4, 2).contiguous()
        x_bn = x_bn.view(N * M, C, T, V)

        h = x_bn
        for i in range(len(self.class_layers)):
            h = self.class_layers[i](h, self.A * self.edge_importance[i], A_act, lamda_act)

        x_class = fn.avg_pool2d(h, h.size()[2:])
        x_class = x_class.view(N, M, -1, 1, 1).mean(dim=1)
        x_class = self.fcn(x_class)
        x_class = x_class.view(x_class.size(0), -1)

        for i in range(len(self.recon_layers)):
            h = self.recon_layers[i](h,
                                     self.A * self.edge_importance_recon[i],
                                     A_act,
                                     lamda_act)

        r6, _ = self.recon_layer_6(torch.cat((h, x_last), 1),
                                   self.A * self.edge_importance_recon[6],
                                   A_act,
                                   lamda_act)  # [N, 64, 1, 25]
        pred = x_last.squeeze().repeat(1, 10, 1) + r6.squeeze()  # [N, 3, 25]

        pred = pred.contiguous().view(-1, 3, 10, 25)
        x_target = x_target.permute(0, 4, 1, 2, 3).contiguous().view(-1, 3, 10, 25)

        return x_class, pred[::2], x_target[::2]

    def extract_feature(self, x):

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


class StgcnBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 edge_type=2,
                 t_kernel_size=1,
                 stride=1,
                 padding=True,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        if padding:
            padding = ((kernel_size[0] - 1) // 2, 0)
        else:
            padding = (0, 0)

        self.gcn = SpatialGcn(in_channels=in_channels,
                              out_channels=out_channels,
                              k_num=kernel_size[1],
                              edge_type=edge_type,
                              t_kernel_size=t_kernel_size)
        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           (kernel_size[0], 1),
                                           (stride, 1),
                                           padding),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A, B, lamda_act):

        res = self.residual(x)
        x, A = self.gcn(x, A, B, lamda_act)
        x = self.tcn(x) + res

        return self.relu(x), A


class StgcnReconBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 edge_type=2,
                 t_kernel_size=1,
                 stride=1,
                 padding=True,
                 dropout=0,
                 residual=True,
                 activation='relu'):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        if padding == True:
            padding = ((kernel_size[0] - 1) // 2, 0)
        else:
            padding = (0, 0)

        self.gcn_recon = SpatialGcnRecon(in_channels=in_channels,
                                         out_channels=out_channels,
                                         k_num=kernel_size[1],
                                         edge_type=edge_type,
                                         t_kernel_size=t_kernel_size)
        self.tcn_recon = nn.Sequential(nn.BatchNorm2d(out_channels),
                                       nn.ReLU(inplace=True),
                                       nn.ConvTranspose2d(in_channels=out_channels,
                                                          out_channels=out_channels,
                                                          kernel_size=(kernel_size[0], 1),
                                                          stride=(stride, 1),
                                                          padding=padding,
                                                          output_padding=(stride - 1, 0)),
                                       nn.BatchNorm2d(out_channels),
                                       nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels,
                                                             out_channels=out_channels,
                                                             kernel_size=1,
                                                             stride=(stride, 1),
                                                             output_padding=(stride - 1, 0)),
                                          nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)
        self.activation = activation

    def forward(self, x, A, B, lamda_act):

        res = self.residual(x)
        x, A = self.gcn_recon(x, A, B, lamda_act)
        x = self.tcn_recon(x) + res
        if self.activation == 'relu':
            x = self.relu(x)
        else:
            x = x

        return x, A


class SpatialGcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 k_num,
                 edge_type=2,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.k_num = k_num
        self.edge_type = edge_type
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * k_num,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A, B, lamda_act):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.k_num, kc // self.k_num, t, v)
        x1 = x[:, :self.k_num - self.edge_type, :, :, :]
        x2 = x[:, -self.edge_type:, :, :, :]
        x1 = torch.einsum('nkctv,kvw->nctw', (x1, A))
        x2 = torch.einsum('nkctv,nkvw->nctw', (x2, B))
        x_sum = x1 + x2 * lamda_act

        return x_sum.contiguous(), A


class SpatialGcnRecon(nn.Module):

    def __init__(self, in_channels, out_channels, k_num, edge_type=3,
                 t_kernel_size=1, t_stride=1, t_padding=0, t_outpadding=0, t_dilation=1,
                 bias=True):
        super().__init__()

        self.k_num = k_num
        self.edge_type = edge_type
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels,
                                         out_channels=out_channels * k_num,
                                         kernel_size=(t_kernel_size, 1),
                                         padding=(t_padding, 0),
                                         output_padding=(t_outpadding, 0),
                                         stride=(t_stride, 1),
                                         dilation=(t_dilation, 1),
                                         bias=bias)

    def forward(self, x, A, B, lamda_act):
        x = self.deconv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.k_num, kc // self.k_num, t, v)
        x1 = x[:, :self.k_num - self.edge_type, :, :, :]
        x2 = x[:, -self.edge_type:, :, :, :]
        x1 = torch.einsum('nkctv,kvw->nctw', (x1, A))
        x2 = torch.einsum('nkctv,nkvw->nctw', (x2, B))
        x_sum = x1 + x2 * lamda_act

        return x_sum.contiguous(), A