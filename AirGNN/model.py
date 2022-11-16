import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from sklearn.mixture import GaussianMixture
import numpy as np
import torch.nn.functional as F
kl_loss = torch.nn.KLDivLoss(reduction="none",log_target = True)
import torch.nn.functional as F
from scipy.stats import entropy


class AirGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(AirGNN, self).__init__()
        self.dropout = args.dropout
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop = AdaptiveMessagePassing(K=args.K, alpha=args.alpha, mode=args.model, args=args)
        print(self.prop)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index, y = data.x, data.adj_t, data.y
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)



class AdaptiveMessagePassing(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
                 K: int,
                 alpha: float,
                 dropout: float = 0.,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 mode: str = None,
                 node_num: int = None,
                 args=None,
                 **kwargs):

        super(AdaptiveMessagePassing, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
        self.mode = mode
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self.node_num = node_num
        self.args = args
        self._cached_adj_t = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None, mode=None) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                raise ValueError('Only support SparseTensor now')

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        add_self_loops=self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        if mode == None: mode = self.mode

        if self.K <= 0:
            return x
        hh = x

        if mode == 'MLP':
            return x

        elif mode == 'APPNP':
            x = self.appnp_forward(x=x, hh=hh, edge_index=edge_index, K=self.K, alpha=self.alpha)

        elif mode in ['AirGNN']:
            x = self.amp_forward(x=x, hh=hh, edge_index=edge_index, K=self.K,alpha = self.alpha)
        else:
            raise ValueError('wrong propagate mode')
        return x

    def appnp_forward(self, x, hh, edge_index, K, alpha):
        gamma = 1
        lambda_amp = 0.5
        alpha = 0.1
        for k in range(K):
            #y = x - gamma * 2 * (1 - lambda_amp) * self.compute_LX(x=x, edge_index=edge_index)  # Equation (9)
            # x = hh + self.proximal_L21(x=y - hh, lambda_=gamma * lambda_amp) # Equation (11) and (12)

            x = self.propagate(x=x,edge_index = edge_index, edge_weight=None, size=None)
            x = x * (1 - alpha)
            x += alpha * hh # + self.proximal_L21(x=y - hh, lambda_=gamma * lambda_amp)
            #x = hh + self.proximal_L21(x = y-hh,lambda_ = gamma*lambda_amp)
        return x


#     def compute_fast_KL(self, m1, s1, p1, m2, s2, p2):
#         '''Compute the KLDivergence between two gaussians, KL(P|Q) scaled between 0 and 1'''
#         #note: precision = p = inv(s)
#         #d = len(m1)
#         kl = 0.5*( np.log2( np.linalg.det(s2)/np.linalg.det(s1) ) - d + np.trace(np.matmul(p2, s1)) + ((m2-m1).T)@(p2)@(m2-m1) )
#         scaled_kl = 1 - np.exp(-kl)

#         return scaled_kl
    def compute_fast_KL(self, m1, s1, p1, m2, s2, p2):
        '''Compute the KLDivergence between two gaussians, KL(P|Q) scaled between 0 and 1'''
        #note: precision = p = inv(s)
       # print("*****************",m1,"****************")
        #d = len(m1)
        kl = 0.5*( torch.log2(s2/s1 ) - 1 + (p2*s1) + ((m2-m1))*(p2)*(m2-m1) )
        scaled_kl = 1 - torch.exp(-kl)

        return scaled_kl

#     def amp_forward(self, x, hh, K, edge_index):
#         lambda_amp = self.args.lambda_amp
#         gamma = 1 / (2 * (1 - lambda_amp))  ## or simply gamma = 1

#         for k in range(K):
#             y = x - gamma * 2 * (1 - lambda_amp) * self.compute_LX(x=x, edge_index=edge_index)  # Equation (9)
#             #y = x - self.compute_LX(x=x,edge_index = edge_index)
#             # gmXin = GaussianMixture(n_components=1, random_state=0).fit(hh.detach().to('cpu').numpy())
#             # gmY = GaussianMixture(n_components=1, random_state=0).fit(y.detach().to('cpu').numpy())
#             # scaled_KL = self.compute_fast_KL(m1=gmXin.means_[0], s1=gmXin.covariances_[0], p1 = gmXin.precisions_[0],m2=gmY.means_[0], s2 = gmY.covariances_[0], p2 = gmY.precisions_[0])
#             # # #x = hh + (1-scaled_KL)*(y-hh) #we use 1-kl because we want to more heavily weigh closer samples.
#             # x = hh + (scaled_KL)*(y-hh) #we use 1-kl because we want to more heavily weigh closer samples.

#             x = hh + self.proximal_L21(x=y - hh, lambda_=gamma * lambda_amp) # Equation (11) and (12)
#         return x
#     def amp_forward(self, x, hh, K, edge_index,alpha):
#         lambda_amp = self.args.lambda_amp
#         gamma = 1 / (2 * (1 - lambda_amp))  ## or simply gamma = 1
  
#         for k in range(K):
#             y = x - gamma * 2 * (1 - lambda_amp) * self.compute_LX(x=x, edge_index=edge_index)  # Equation (9)
      
#             k = np.zeros((hh.size()[0],1))
#             for i in range(hh.size()[0]):
#                 unscaled = entropy(F.softmax(hh[i]).to('cpu').detach().numpy(),F.softmax(y[i]).to('cpu').detach().numpy())
#                 scaled = 1-np.exp(-unscaled)
#                 k[i][0] = 1-np.exp(-unscaled)
#             k = np.where(k<np.percentile(k,90),0,0.1)
                
   
             ##x = hh + (1-scaled_KL)*(y-hh) #we use 1-kl because we want to more heavily weigh closer samples.
#              #x = hh + (scaled_KL)*(y-hh) #we use 1-kl because we want to more heavily weigh closer samples.
#             x = alpha * hh + (1 - alpha)*y + torch.tensor(k).to('cuda')*(y-hh)

#             #x = hh + self.proximal_L21(x=y - hh, lambda_=gamma * lambda_amp) # Equation (11) and (12)
#         return x

    def amp_forward(self, x, hh, K, edge_index,alpha):
        lambda_amp = self.args.lambda_amp
        gamma = 1 / (2 * (1 - lambda_amp))  ## or simply gamma = 1
  
        for k in range(K):
            y = x - gamma * 2 * (1 - lambda_amp) * self.compute_LX(x=x, edge_index=edge_index)  # Equation (9)
      
            # k = np.zeros((hh.size()[0],1))
            # for i in range(hh.size()[0]):
            #     unscaled = entropy(F.softmax(hh[i]).to('cpu').detach().numpy(),F.softmax(y[i]).to('cpu').detach().numpy())
            #     scaled = 1-np.exp(-unscaled)
            #     k[i][0] = 1-np.exp(-unscaled)
            # k = np.where(k<np.percentile(k,90),0,0.1)
            kl = kl_loss(F.log_softmax(hh),F.log_softmax(y))
            kl = kl.sum(dim =1)
            kl = torch.where(kl<torch.quantile(kl,0.9),0.0,0.1)
            #print("*****************",k.size())
                
   
              # x = hh + (1-scaled_KL)*(y-hh) #we use 1-kl because we want to more heavily weigh closer samples.
              # x = hh + (scaled_KL)*(y-hh) #we use 1-kl because we want to more heavily weigh closer samples.
            x = alpha * hh + (1 - alpha)*y + kl.unsqueeze(1)*(y-hh)

            #x = hh + self.proximal_L21(x=y - hh, lambda_=gamma * lambda_amp) # Equation (11) and (12)
        return x

    def proximal_L21(self, x: Tensor, lambda_):
        row_norm = torch.norm(x, p=2, dim=1)
        score = torch.clamp(row_norm - lambda_, min=0)
        index = torch.where(row_norm > 0)             #  Deal with the case when the row_norm is 0
        score[index] = score[index] / row_norm[index] # score is the adaptive score in Equation (14)
        return score.unsqueeze(1) * x

    def compute_LX(self, x, edge_index, edge_weight=None):
        x = x - self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, alpha={}, mode={}, dropout={}, lambda_amp={})'.format(self.__class__.__name__, self.K,
                                                               self.alpha, self.mode, self.dropout,
                                                               self.args.lambda_amp)