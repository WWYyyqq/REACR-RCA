import os
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# -------------------- Neural Network Architecture -------------------------
# MeanSage - Encoder
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout, num_layers, norm):
        super(GraphSAGEEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        hidden_feats = hidden_feats if num_layers > 1 else out_feats
        self.input_conv = dgl.nn.GraphConv(in_feats, hidden_feats, norm=norm)
        self.convs = []
        for _ in range(num_layers - 2):
            self.convs.append(dgl.nn.GraphConv(hidden_feats, hidden_feats, norm=norm))
        if num_layers > 1:
            self.convs.append(dgl.nn.GraphConv(hidden_feats, out_feats, norm=norm))

    def forward(self, g, features):
        h = F.leaky_relu(self.input_conv(g, features))
        h = self.dropout(h)
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
            h = self.dropout(h)
        return h

    def transform(self, g, features):

        h = F.leaky_relu(self.input_conv(g, features))
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
        return h

# MeanSage - Decoder
class GraphSAGEDecoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout, num_layers, norm):
        super(GraphSAGEDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        hidden_feats = hidden_feats if num_layers > 1 else out_feats
        self.input_conv = dgl.nn.GraphConv(in_feats, hidden_feats)
        self.convs = []
        for _ in range(num_layers - 2):
            self.convs.append(dgl.nn.GraphConv(hidden_feats, hidden_feats))
        if num_layers > 1:
            self.convs.append(dgl.nn.GraphConv(hidden_feats, out_feats))

    def forward(self, g, features):
        h = F.leaky_relu(self.input_conv(g, features))
        h = self.dropout(h)
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
            h = self.dropout(h)
        return h

    def transform(self, g, features):
        h = F.leaky_relu(self.input_conv(g, features))
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
        return h

# GraphSAGE - AE
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout=0.0, mask_rate=0.0, num_layers=2, norm='none'):
        super(GraphSAGE, self).__init__()
        self.encoder = GraphSAGEEncoder(in_feats, hidden_feats, out_feats, dropout, num_layers, norm)
        self.decoder = GraphSAGEDecoder(out_feats, hidden_feats, in_feats, dropout, num_layers, norm)
        self.mask_rate = mask_rate

    def forward(self, g, features):
        z = self.encoder(g, features)
        x_hat = self.decoder(g, z)
        return x_hat

    def transform(self, g, features):
        z = self.encoder.transform(g, features)
        x_hat = self.decoder.transform(g, z)
        return x_hat

# Linear - Encoder/Decoder
class LinearCoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout=0):
        super(LinearCoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_feats, hidden_feats)
        self.fc2 = nn.Linear(hidden_feats, out_feats)

    def forward(self, z):
        h = F.leaky_relu(self.fc1(z))
        h = self.dropout(h)
        x_hat = self.fc2(h)
        return x_hat

    def transform(self, z):
        h = F.leaky_relu(self.fc1(z))
        x_hat = self.fc2(h)
        return x_hat

# Linear - AE
class MLPAE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(MLPAE, self).__init__()
        self.encoder = LinearCoder(in_feats, hidden_feats, out_feats)
        self.decoder = LinearCoder(out_feats, hidden_feats, in_feats)

    def forward(self, g, features):
        z = self.encoder(features)
        x_hat = self.decoder(z)
        return x_hat

    def transform(self, g, features):
        z = self.encoder.transform(features)
        x_hat = self.decoder.transform(z)
        return x_hat

# GAT - Encoder/Decoder
class GATCoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads, dropout):
        super(GATCoder, self).__init__()
        self.conv1 = dgl.nn.GATConv(in_feats, hidden_feats, num_heads, dropout)
        self.conv2 = dgl.nn.GATConv(hidden_feats * num_heads, out_feats, 1, dropout)

    def forward(self, graph, feat):
        h = F.elu(self.conv1(graph, feat))
        h = h.flatten(1)
        h = self.conv2(graph, h).mean(1)
        return h

# GAT - AE
class GATAE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GATAE, self).__init__()
        self.encoder = GATCoder(in_feats, hidden_feats, out_feats, num_heads=3, dropout=0)
        self.decoder = GATCoder(out_feats, hidden_feats, in_feats, num_heads=3, dropout=0)

    def forward(self, g, features):
        z = self.encoder(g, features)
        x_hat = self.decoder(g, z)
        return x_hat
# end Neural Network Architecture -----------------------------------------



# -------------------- Collate Function  -------------------------
# Creating a DataLoader for gae pre_training.
# Sample Format: (timestamp, topo, node_feats)
def collate(samples):
    timestamps, graphs, feats = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return timestamps, batched_graph, torch.cat(feats, dim=0)

def create_dataloader(samples, batch_size, shuffle=True):
    dataset = list(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return dataloader

# Creating a DataLoader for feedback.
def collate_fd(samples):
    graphs, feats, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    feats = torch.stack([torch.tensor(step) for series in feats for step in series])
    feats = feats.view(-1, feats.shape[-3]//len(labels), feats.shape[-2], feats.shape[-1]).permute(0,2,1,3)
    shape = feats.shape
    feats = feats.reshape(-1, shape[-2], shape[-1])
    return batched_graph, feats, torch.tensor(np.array([lb.cpu().detach().numpy() for lb in labels])).view(-1)

def create_dataloader_fd(fd_samples, labels, batch_size, shuffle=False):
    dataset = [[fd_samples[i][-1][1], [step[2] for step in fd_samples[i]], labels[i]] for i in range(len(labels))]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fd)
    return dataloader
# end Collate Function  ------------------------------------------



# -------------------- Loss Function  -------------------------
# Loss_version_0: torch.nn.MSELoss()

# Loss_version_1: Calculate the MSE loss separately for different modalities of data and then sum them up.
# for gae pre_training.
class ModalLoss(nn.Module):
    def __init__(self, feat_span):
        super(ModalLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.feat_span = feat_span

    def forward(self, inputs, targets):
        h = 0
        for start, end in self.feat_span:
            loss = self.mse(inputs[:, start: end + 1], targets[:, start: end + 1])
            h += loss
        return h

    def compute(self, inputs, targets):
        h = 0
        for start, end in self.feat_span:
            loss = self.mse(inputs[start: end + 1], targets[start: end + 1])
            h += loss
        return h

# Loss_version_2
# for feedback.
class RankingLoss(nn.Module):
    def __init__(self, rank_range, margin=None):
        super(RankingLoss, self).__init__()
        if rank_range <= 1:
            raise ValueError('rank_range must be greater than 1')
        self.rank_range = rank_range
        self.margin = (1.0 / rank_range) if margin is None else float(margin)
        if self.margin < 0:
            raise ValueError('margin must be non-negative')
        self.register_buffer(
            'weights',
            torch.tensor([(1 / i) for i in range(1, rank_range)], dtype=torch.float32),
        )

    def forward(self, predicted_scores, labels):
        loss = []
        for start in range(0, len(labels), self.rank_range):
            end = start + self.rank_range
            tmp_s = predicted_scores[start:end]
            tmp_l = labels[start:end]
            positives = tmp_l > 0
            negatives = ~positives
            if not torch.any(positives) or not torch.any(negatives):
                continue
            # Service labels form a positive candidate bag.  Penalize nodes
            # ranked above the best candidate rather than forcing every replica
            # to receive the same score.  Operating on raw bounded scorer
            # outputs avoids 1/(max-min) gradient explosions near ties.
            positive_bag_score = torch.max(tmp_s[positives])
            negative_scores = torch.sort(tmp_s[negatives], descending=True).values
            violations = torch.relu(self.margin - positive_bag_score + negative_scores)
            rank_weights = self.weights[: negative_scores.numel()].to(
                device=violations.device,
                dtype=violations.dtype,
            )
            loss.append(
                torch.sum(violations * rank_weights)
                / rank_weights.sum().clamp_min(1e-12)
            )
        if not loss:
            return predicted_scores.mean() * 0.0
        return torch.mean(torch.stack(loss))
# end Loss Function  ------------------------------------------
