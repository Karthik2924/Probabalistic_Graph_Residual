import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
from torch_sparse import SparseTensor, matmul
import torch.nn.functional as F
from scipy import stats

kl_loss = torch.nn.KLDivLoss(reduction="none",log_target = True)
import torch.nn.functional as F
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
        self.kl_residual = []
        self.beta_vals = []
        self.ks_stat=[]
        self.ks_p = []
        self.ks = []

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
            x = self.amp_forward(x=x, hh=hh, edge_index=edge_index, K=self.K)
        elif mode in ['kl_res']:
            x = self.klres_forward(x=x, hh=hh, edge_index = edge_index, K = self.K, alpha = self.alpha)
        elif mode in ['ks']:
            x = self.ks_forward(x = x, hh = hh, edge_index = edge_index, K = self.K, alpha = self.alpha)
        else:
            raise ValueError('wrong propagate mode')
        return x

    def appnp_forward(self, x, hh, edge_index, K, alpha):
        #print("****alpha***** = ",alpha)
        for k in range(K):
            x = self.propagate(edge_index, x=x, edge_weight=None, size=None)
            x = x * (1 - alpha)
            x += alpha * hh
        return x

    def amp_forward(self, x, hh, K, edge_index):
        lambda_amp = self.args.lambda_amp
        gamma = 1 / (2 * (1 - lambda_amp))  ## or simply gamma = 1

        for k in range(K):
            y = x - gamma * 2 * (1 - lambda_amp) * self.compute_LX(x=x, edge_index=edge_index)  # Equation (9)
            x = hh + self.proximal_L21(x=y - hh, lambda_=gamma * lambda_amp) # Equation (11) and (12)
        return x
    def ks_forward(self,x,hh,K,edge_index,alpha):
        for k in range(K):
            y = self.propagate(edge_index, x = x,edge_weight = None, size = None)
            ks = list(map(stats.kstest,hh.detach().to('cpu').numpy(),y.detach().to('cpu').numpy()))
            self.ks.extend(ks)
            ks = torch.tensor(ks)[:,1].to('cuda')
            score = torch.where(ks<0.25,0.05,0.25).unsqueeze(1)
            x = score*hh + (1-score)*hh
            #ks = ks.to('cuda')
            #ks = torch.where(k<torch.quantile(ks,0.9),0,0.1)
            #x = hh + score.unsqueeze(1)*(y-hh)
        return x
    def klres_forward(self,x,hh,K,edge_index,alpha):
        lambda_amp = self.args.lambda_amp
        gamma = 1 / (2 * (1 - lambda_amp))  ## or simply gamma = 1

        for k in range(K):
            y = self.propagate(edge_index,x=x,edge_weight = None,size = None)
            kl = kl_loss(F.log_softmax(hh),F.log_softmax(y))
            kl = kl.sum(dim =1)
            kl = 1-torch.exp(-kl)

            score = torch.where(kl<0.1,0.1,0.01).unsqueeze(1)

            x = score * hh + (1 - score)*y #+ kl.unsqueeze(1)*(y-hh)

        return x
        

    def proximal_L21(self, x: Tensor, lambda_):
        # print(x.shape)
        # print(x[0])
        row_norm = torch.norm(x, p=2, dim=1)
        # print(row_norm.shape)
        # print(row_norm[0])
        score = torch.clamp(row_norm - lambda_, min=0)
        index = torch.where(row_norm > 0)             #  Deal with the case when the row_norm is 0
        score[index] = score[index] / row_norm[index] # score is the adaptive score in Equation (14)
        #self.beta_vals.extend(score.detach().to('cpu').numpy())
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




