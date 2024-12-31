from torch import nn
import torch
from torch.utils.data import DataLoader
# Define the projector module as a subclass of nn.Module
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import math

def soft_trip_fn(d_p, d_n):
    return torch.log(1 + torch.exp(d_p-d_n))

class Emotion_Loss(nn.Module):
    def __init__(self, n_classes=3, K=10, gamma=.1, embed_dim=128, device='cuda'):
        super().__init__()
        self.gamma = 1./gamma
        self.n_classes = n_classes
        self.K = K
        # self.W = nn.Parameter(torch.Tensor(embed_dim, n_classes*K)).to(device)
        self.W = nn.Parameter(torch.Tensor(embed_dim, n_classes * K))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.weight = torch.zeros(n_classes*K, n_classes*K, dtype=torch.bool).to(device)

    def forward(self, inputs, targets=None, reduction='mean'):
        centers = F.normalize(self.W, p=2, dim=0)
        x = F.normalize(inputs, p=2, dim=1)
        simInd = x.matmul(centers)
        simStruc = simInd.reshape(-1, self.n_classes, self.K)
        prob = torch.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        if targets is None:
            return simClass

        dis = 2.0 + 1e-5 - 2*simClass
        neg_ind, neutral_ind, pos_ind = 0, 1, 2
        opposite_ind = (-(targets-1))+1
        # torch.cat((opposite_ind.view(-1,1), targets.view(-1,1)), dim=1)
        # dist_self = dis[:, targets] # distance of each item in batch with its class
        r_bs = range(len(targets))
        dist_self = dis[r_bs, targets]
        # torch.cat((dis, targets.view(-1, 1), dist_self.view(-1, 1)), dim=1)
        dist_neutral = dis[:,neutral_ind] # distance of each item in batch with neutral class
        dist_opposite = dis[r_bs, opposite_ind] # distance of each item in batch with positive class

        m_neutral = targets == neutral_ind
        m_pos_neg = targets != neutral_ind

        loss = soft_trip_fn(dist_self, 0.) # term_self
        term_pos_neg = soft_trip_fn(dist_self[m_pos_neg], dist_neutral[m_pos_neg]) +\
                       soft_trip_fn(dist_neutral[m_pos_neg], dist_opposite[m_pos_neg])
                       # soft_trip_fn(dist_self[m_pos_neg],dist_opposite[m_pos_neg])

        term_neutral = soft_trip_fn(dist_self[m_neutral], dis[m_neutral,pos_ind]) +\
                       soft_trip_fn(dist_self[m_neutral], dis[m_neutral, neg_ind])

        loss[m_pos_neg] += term_pos_neg
        loss[m_neutral] += term_neutral
        if reduction == 'mean':
            return loss.mean(), simClass
        else:
            return loss, simClass


class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        # Call the parent constructor
        super(Projector, self).__init__()
        # Define the first linear layer with input_dim and 256 units
        self.linear1 = nn.Linear(input_dim, 256)
        # Define the ReLU activation function
        self.relu = nn.ReLU()
        # Define the second linear layer with 256 and output_dim units
        self.linear2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # Apply the first linear layer to the input x
        x = self.linear1(x)
        # Apply the ReLU activation function
        x = self.relu(x)
        # Apply the second linear layer to get the output
        x = self.linear2(x)
        # Return the output
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x


class Soft_Trip_Loss(nn.Module):
    def __init__(self, n_classes=3, K=10, la=20, gamma=.1, tau=0.2,
                 embed_dim=128, margin=.01, device='cuda'):
        super().__init__()
        self.la = la
        self.gamma = 1./gamma
        self.n_classes = n_classes
        self.K = K
        self.W = nn.Parameter(torch.Tensor(embed_dim, n_classes*K))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.weight = torch.zeros(n_classes*K, n_classes*K, dtype=torch.bool).to(device)
        for i in range(0, n_classes):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        self.tau = tau
        self.margin = margin

    def forward(self, inputs, targets, return_loss=True):
        centers = F.normalize(self.W, p=2, dim=0)
        x = F.normalize(inputs, p=2, dim=1)
        simInd = x.matmul(centers)
        simStruc = simInd.reshape(-1, self.n_classes, self.K)
        prob = torch.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        if not return_loss:
            return simClass
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), targets] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), targets)

        # # Calculate the mean of data points for each class
        # class_mean = torch.zeros(self.n_classes, self.K, centers.shape[0]).to(inputs.device)
        # for i in range(self.n_classes):
        #     class_mean[i] = torch.mean(x[targets == i], dim=0)
        #
        # # Calculate the distance between centers and class means
        # center_mean_dist = torch.norm(centers.unsqueeze(0) - class_mean.unsqueeze(1), dim=2)
        #
        # # Add the regularization term encouraging centers to be close to class means
        # reg_center_mean = torch.mean(center_mean_dist)
        # reg_weight = self.la * reg_center_mean
        # Calculate the diversity regularization term
        # div_term = torch.sum(torch.sqrt(2.0 + 1e-5 - 2. * simStruc), dim=(1, 2)) / (self.n_classes * self.K)
        # div_weight = self.la * torch.mean(div_term)

        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/\
                  (self.n_classes*self.K*(self.K-1.))
            return lossClassify+self.tau*reg, simClass
        else:
            return lossClassify, simClass

class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]

    def forward(self, anchor, positive, negatives):
        if len(anchor.size()) == 1:
            anchor = anchor.unsqueeze(0)

        if len(positive.size()) == 1:
            positive = positive.unsqueeze(0)

        batch_size = anchor.size(0)
        anchor, positive, negatives = self.normalize(anchor, positive, negatives)

        positive_logit = torch.sum(anchor * positive, dim=1, keepdim=True)
        negative_logits = anchor @ negatives.transpose(-1, -2)
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits / self.temperature, labels, reduction='mean')
        return loss

# Define a batch sampler that can generate batches of signal segments from the dataset based on their emotional labels, such that each batch contains one positive pair and n negative pairs
class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, n=32):
        # dataset is an instance of torcheeg.datasets.SEEDDataset
        self.dataset = dataset
        # n is the number of negative pairs per batch
        self.n = n
        # get the labels of all samples in the dataset
        self.labels = torch.tensor([sample[1] for sample in dataset])
        # get the indices of samples for each label
        self.label_to_indices = {label.item(): torch.nonzero(self.labels == label).squeeze() for
                                 label in torch.unique(self.labels)}

    def __iter__(self):
        # iterate over all samples in the dataset
        for i in range(len(self.dataset)):
            # get the label of the current sample
            label = self.labels[i].item()
            # get a random index of another sample with the same label (positive pair)
            pos_index = i
            while pos_index == i:
                pos_index = torch.randint(0, len(self.label_to_indices[label]), size=(1,)).item()
                pos_index = self.label_to_indices[label][pos_index]

            # get n random indices of samples with different labels (negative pairs)
            neg_indices = []
            while len(neg_indices) < self.n:
                neg_label = label
                while neg_label == label:
                    neg_label = torch.randint(0, len(self.label_to_indices), size=(1,)).item()
                    neg_index = torch.randint(0, len(self.label_to_indices[neg_label]), size=(1,)).item()
                    neg_index = self.label_to_indices[neg_label][neg_index]


                if neg_index not in neg_indices:
                    neg_indices.append(neg_index)
                # yield a batch of indices with one positive pair and n negative pairs

            yield [i, pos_index] + neg_indices


    def __len__(self):
        # return the number of batches generated by this sampler
        return len(self.dataset)

def main():
    a = Emotion_Loss(3)
    for n, p in a.named_parameters():
        print(n, p.shape())

    s = Soft_Trip_Loss()
    for n, p in a.named_parameters():
        print(n, s.shape())


    from dataset import get_dataset
    n_subjects = 15
    train_dataset, test_dataset = get_dataset(n_subjects, sub_i=0)
    batch_sampler = BatchSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    it = iter(train_loader)
    batch = next(it)
    X = batch[0][0].cuda()
    y = batch[1].cuda()
    from Common.utils import MyDict
    args = MyDict()
    args.lam = .5
    args.n_subj = 14  # number of training subjects
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from eeg_dml import create_model
    model = create_model(args=args)
    X_f = model(X, return_fea=True)
    proj = Projector(model.d_f, output_dim=128).cuda()
    X_e = proj(X_f)
    # loss = softmax_contrastive_loss(X_e, y)
    infoNCE = InfoNCELoss(temperature=1.)
    loss = infoNCE(X_e[0], X_e[1], X_e[2:])
    pass



if __name__ == '__main__':
    main()
    print('Congratulations to you!')