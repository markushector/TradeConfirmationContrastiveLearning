import torch
import torch.nn as nn
import torch.nn.functional as F

from contrastive_matching.utils.utils import get_args


class ContrastiveNetwork(nn.Module):
    """
    Weight-sharing network that converts embedded trades and confirmations to their final latent representations.
    """

    def __init__(self, dim=56, p=0.0):
        super(ContrastiveNetwork, self).__init__()
        self.args = get_args()

        self.layer1 = nn.Linear(dim, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 50)
        self.layer4 = nn.Linear(50, 50)
        self.layer5 = nn.Linear(50, 30)

        self.batch_norm1 = nn.BatchNorm1d(dim)
        self.batch_norm2 = nn.BatchNorm1d(dim)
        self.batch_norm3 = nn.BatchNorm1d(dim)
        self.batch_norm4 = nn.BatchNorm1d(dim)

        self.dropout1 = nn.Dropout(p=p)
        self.dropout2 = nn.Dropout(p=p)
        self.dropout3 = nn.Dropout(p=p)
        self.dropout4 = nn.Dropout(p=p)

        if self.args.projection_head:
            self.projection_head = nn.Linear(dim, dim)

    def forward(self, x):
        # print(f"Xshape: {x.shape}")
        # print(f"Input: {self.layer1.in_features}")
        x = self.dropout1(F.relu(self.layer1(x)))
        x = self.dropout2(F.relu(self.layer2(x)))
        x = self.dropout3(F.relu(self.layer3(x)))
        x = self.dropout4(F.relu(self.layer4(x)))

        if self.args.projection_head:
            x = F.relu(self.layer5(x))
            x = F.normalize(self.projection_head(x))
        else:
            x = self.layer5(x)

        return x


def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)


class ContrastiveLossOriginal(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """
    def __init__(self, batch_size, temperature=0.001):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    @staticmethod
    def calc_similarity_batch_original(a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch_original(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        #nominator = torch.exp(positives / self.temperature)
        n = positives / self.temperature

        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        #all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        all_losses = -n + torch.logsumexp(similarity_matrix / self.temperature, dim=1)
        #print(f"Similarity matrix {similarity_matrix[0, :]}")
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        # Simplify line 81
        return loss, torch.sum(positives)


class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """
    def __init__(self, batch_size, temperature=0.001):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

    @staticmethod
    def calc_similarity_batch(a, b):
        return F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=2)

    @staticmethod
    def calc_similarity_batch_original(a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        positives = torch.diag(similarity_matrix)

        n = positives / self.temperature

        all_losses = -n + torch.logsumexp(similarity_matrix / self.temperature, dim=1)
        #print(f"Similarity matrix {similarity_matrix[0, :]}")
        loss = torch.sum(all_losses) / self.batch_size
        # Simplify line 81
        return loss, torch.sum(positives)
