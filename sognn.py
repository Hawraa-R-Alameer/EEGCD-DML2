from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GraphConv, DenseSAGEConv, dense_diff_pool, DenseGCNConv
from torch_geometric.nn import TopKPooling, SAGPooling, EdgePooling
from torch.nn import Linear, Dropout, PReLU, Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
from imps import *


device = torch.device('cuda', 0)

class SOGC(torch.nn.Module):
    """Self-organized Graph Construction Module
    Args:
        in_features: size of each input sample
        bn_features: size of bottleneck layer
        out_features: size of each output sample
        topk: size of top k-largest connections of each channel
    """
    def __init__(self, in_features: int, bn_features: int, out_features: int, topk: int):
        super().__init__()

        self.channels = 62 
        self.in_features = in_features
        self.bn_features = bn_features
        self.out_features = out_features
        self.topk = topk
        
        self.bnlin = Linear(in_features, bn_features)#Linear Bottleneck layer#(44*32, 32)
        self.gconv = DenseGCNConv(in_features, out_features)#(44*32, 32)
        
    def forward(self, x): 
        x = x.reshape(-1, self.channels, self.in_features)
        xa = torch.tanh(self.bnlin(x))
        adj = torch.matmul(xa, xa.transpose(2,1))#/self.bn_features 
        adj = torch.softmax(adj, 2)
        amask = torch.zeros(xa.size(0), self.channels, self.channels).to(device)
        amask.fill_(0.0)
        s, t = adj.topk(self.topk, 2)
        amask.scatter_(2, t, s.fill_(1))
        adj = adj*amask
        x = F.relu(self.gconv(x, adj)) 
        
        return x


class SOGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        drop_rate = 0.1
        topk = 10
        self.channels = 62 
        
        self.conv1 = Conv2d(1, 32, (5,5))
        self.drop1 = Dropout(drop_rate)
        self.pool1 = MaxPool2d((1,4))
        self.sogc1 = SOGC(65*32, 64, 32, topk)  
        
        self.conv2 = Conv2d(32, 64, (1,5))
        self.drop2 = Dropout(drop_rate)
        self.pool2 = MaxPool2d((1,4)) 
        self.sogc2 = SOGC(15*64, 64, 32, topk) 
        
        self.conv3 = Conv2d(64, 128, (1,5))
        self.drop3 = Dropout(drop_rate)
        self.pool3 = MaxPool2d((1,4)) 
        self.sogc3 = SOGC(2*128, 64, 32, topk) 
        
        self.drop4 = Dropout(drop_rate)

        self.linend = Linear(self.channels*32*3, 3) 

        
    def forward(self, x, return_fea=False):
        x = x.float()
        x = x.reshape(-1, 1, 5, 265) #(Batch*channels, 1, Freq_bands, Features)
        
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = self.pool1(x)
        
        x1 = self.sogc1(x)
        
        x = F.relu(self.conv2(x)) 
        x = self.drop2(x)
        x = self.pool2(x)
        
        x2 = self.sogc2(x)
        
        x = F.relu(self.conv3(x))  
        x = self.drop3(x)
        x = self.pool3(x) 
        
        x3 = self.sogc3(x)
        
        x = torch.cat([x1, x2, x3], dim=1) 
        x = self.drop4(x)

        x = x.reshape(-1, self.channels*32*3)
        if return_fea:
            return x

        x = self.linend(x)
        pred = F.softmax(x,1) 
        
        return x, pred

# I intend to implement this network to run SOGNN on DEAP -- not completed
class SOGNN2(torch.nn.Module):
    def __init__(self, n_channel=62, n_freq_band=5):
        super().__init__()

        drop_rate = 0.1
        topk = 10
        self.channels = n_channel
        self.n_freq_band = n_freq_band

        self.conv1 = Conv2d(1, 32, (n_freq_band, 5))
        self.drop1 = Dropout(drop_rate)
        self.pool1 = MaxPool2d((1, 4))
        self.sogc1 = SOGC(65 * 32, 64, 32, topk)

        self.conv2 = Conv2d(32, 64, (1, 5))
        self.drop2 = Dropout(drop_rate)
        self.pool2 = MaxPool2d((1, 4))
        self.sogc2 = SOGC(15 * 64, 64, 32, topk)

        self.conv3 = Conv2d(64, 128, (1, 5))
        self.drop3 = Dropout(drop_rate)
        self.pool3 = MaxPool2d((1, 4))
        self.sogc3 = SOGC(2 * 128, 64, 32, topk)

        self.drop4 = Dropout(drop_rate)

        self.linend = Linear(self.channels * 32 * 3, 3)

    def forward(self, x, feature_flag=False):
        x = x.float()
        x = x.reshape(x.size(0), 1, self.n_freq_band, -1)  # (Batch*channels, 1, Freq_bands, Features)

        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = self.pool1(x)

        x1 = self.sogc1(x)

        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.pool2(x)

        x2 = self.sogc2(x)

        x = F.relu(self.conv3(x))
        x = self.drop3(x)
        x = self.pool3(x)

        x3 = self.sogc3(x)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.drop4(x)

        x = x.reshape(-1, self.channels * 32 * 3)
        if feature_flag:
            return x

        x = self.linend(x)
        pred = F.softmax(x, 1)

        return x, pred


class DSBNLayer(nn.Module):
    def __init__(self, num_features):
        super(DSBNLayer, self).__init__()
        # Create a separate BN layer for each domain
        self.bn_src = nn.BatchNorm1d(num_features)
        self.bn_tar = nn.BatchNorm1d(num_features)

    def forward(self, x, domain='src'):
        # x: input features (batch_size, num_features)
        original_shape = x.shape
        B = original_shape[0]
        rest = original_shape[1:]
        x = x.reshape(B, -1)

        if domain in ['src', 'source']:
            out = self.bn_src(x)
        else:
            out = self.bn_tar(x)

        out = out.reshape(*original_shape)
        return out


class SOGC_DSBN(nn.Module):
    """Self-organized Graph Construction Module
    Args:
        in_features: size of each input sample
        bn_features: size of bottleneck layer
        out_features: size of each output sample
        topk: size of top k-largest connections of each channel
    """

    def __init__(self, in_features: int, bn_features: int, out_features: int, topk: int):
        super().__init__()

        self.channels = 62
        self.in_features = in_features
        self.bn_features = bn_features
        self.out_features = out_features
        self.topk = topk
        self.dsbn = DSBNLayer(self.channels*bn_features)
        self.bnlin = Linear(in_features, bn_features)  # Linear Bottleneck layer#(44*32, 32)
        self.gconv = DenseGCNConv(in_features, out_features)  # (44*32, 32)

    def forward(self, x, domain='src'):
        x = x.reshape(-1, self.channels, self.in_features)

        # xa = torch.tanh(self.bnlin(x)) -->
        xa = self.bnlin(x)
        xa = self.dsbn(xa, domain)
        xa = torch.tanh(xa)

        adj = torch.matmul(xa, xa.transpose(2, 1))  # /self.bn_features
        adj = torch.softmax(adj, 2)
        amask = torch.zeros(xa.size(0), self.channels, self.channels).to(device)
        amask.fill_(0.0)
        s, t = adj.topk(self.topk, 2)
        amask.scatter_(2, t, s.fill_(1))
        adj = adj * amask
        x = F.relu(self.gconv(x, adj))

        return x


class SOGNN_DSBN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        drop_rate = 0.1
        topk = 10
        self.channels = 62

        self.conv1 = Conv2d(1, 32, (5, 5))
        self.drop1 = Dropout(drop_rate)
        self.pool1 = MaxPool2d((1, 4))
        self.sogc1 = SOGC_DSBN(65 * 32, 64, 32, topk)

        self.conv2 = Conv2d(32, 64, (1, 5))
        self.drop2 = Dropout(drop_rate)
        self.pool2 = MaxPool2d((1, 4))
        self.sogc2 = SOGC_DSBN(15 * 64, 64, 32, topk)

        self.conv3 = Conv2d(64, 128, (1, 5))
        self.drop3 = Dropout(drop_rate)
        self.pool3 = MaxPool2d((1, 4))
        self.sogc3 = SOGC_DSBN(2 * 128, 64, 32, topk)

        self.drop4 = Dropout(drop_rate)
        n_fea = self.channels * 32 * 3
        self.dsbn = DSBNLayer(n_fea)
        self.linend = Linear(n_fea, 3)

    def forward(self, x, domain='src', return_fea=True):
        x = x.float()
        x = x.reshape(-1, 1, 5, 265)  # (Batch*channels, 1, Freq_bands, Features)

        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = self.pool1(x)

        x1 = self.sogc1(x, domain)

        x = F.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.pool2(x)

        x2 = self.sogc2(x, domain)

        x = F.relu(self.conv3(x))
        x = self.drop3(x)
        x = self.pool3(x)

        x3 = self.sogc3(x, domain)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.drop4(x)

        x = x.reshape(-1, self.channels * 32 * 3)
        x = self.dsbn(x, domain)
        if return_fea:
            return x

        x = self.linend(x)
        pred = F.softmax(x, 1)
        return x, pred


def main():
    device = torch.device('cuda')
    model = SOGNN_DSBN().to(device)
    x = torch.rand(16, 62, 265*5).to(device)
    x2 = torch.rand(16, 62, 265*5).to(device)
    out = model(x, domain='src', return_fea=True)
    out2 = model(x, domain='tar')
    pass


if __name__ == '__main__':
    main()
    print('Congratulations to you!')