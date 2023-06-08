import torch

from contrastive_matching.dataloader.dataloader import ContrastiveDataSet, Dictionary
from contrastive_matching.model.attention_module import Attention
from contrastive_matching.model.contrastive_network import ContrastiveNetwork, ContrastiveLoss, ContrastiveLossOriginal
from contrastive_matching.utils.utils import get_args

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

from wordcloud import WordCloud
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


class VisiNet(nn.Module):
    """
    Model class putting everything together. Has methods to load data, build model and evaluate.
    """

    PATH = './saved_models'

    def __init__(self):
        super(VisiNet, self).__init__()

        # Arguments
        self.args = get_args()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Data
        self.dictionary = None
        self.data = None
        self.train_data = None
        self.dataloader = None
        self.val_data = None
        self.val_dataloader = None
        self.test_data = None
        self.test_dataloader = None
        self.unseen_data = None

        # Networks
        self.trade_attention = None
        self.conf_attention = None
        self.contrastive = None

        # Training
        self.lr = None
        self.epochs = None
        self.params = None
        self.optim = None
        self.contrastive_loss = None

    def forward(self, x, y, batch=200):

        #x = self.trade_attention(x)
        #y = self.conf_attention(y)
        #print(x.shape)
        #print(y.shape)

        if self.args.concatenate:
            x = x.view(batch, -1)
            y = y.view(batch, -1)
            #print("======================")
            #print(x.shape)
            #print(y.shape)
        else:
            x = torch.mean(x, 1)
            y = torch.mean(y, 1)

        x = self.contrastive(x)
        y = self.contrastive(y)

        return x, y

    def forward_attention_weights(self, x, y, batch=200):

        x, x_w = self.trade_attention(x, weights=True)
        y, y_w = self.conf_attention(y, weights=True)

        if self.args.concatenate:
            x = x.view(batch, -1)
            y = y.view(batch, -1)
        else:
            x = torch.mean(x, 1)
            y = torch.mean(y, 1)

        x = self.contrastive(x)
        y = self.contrastive(y)

        return x, y, x_w, y_w

    def forward_trade(self, trade):
        x = self.trade_attention(trade)
        if self.args.concatenate:
            x = x.view(self.args.batch_size, -1)
        else:
            x = torch.mean(x, 1)
        x = self.contrastive(x)
        return x

    def forward_conf(self, conf):
        x = self.conf_attention(conf)
        if self.args.concatenate:
            x = x.view(self.args.batch_size, -1)
        else:
            x = torch.mean(x, 1)
        x = self.contrastive(x)
        return x

    def load_data(self):

        print("Loading data")

        self.dictionary = Dictionary()
        self.data = ContrastiveDataSet(self.dictionary, self.device)
        self.train_data, self.val_data, self.test_data = random_split(self.data, [0.8, 0.1, 0.1])
        self.dataloader = DataLoader(self.train_data,
                                     batch_size=self.args.batch_size,
                                     shuffle=self.args.shuffle)
        self.val_dataloader = DataLoader(self.val_data,
                                         batch_size=10,
                                         shuffle=True)
        self.test_dataloader = DataLoader(self.test_data,
                                          batch_size=10,
                                          shuffle=True)

    def build_model(self):

        print("Building model")

        # Networks
        self.trade_attention = Attention(dim=self.args.embedding_dimension)
        self.conf_attention = Attention(dim=self.args.embedding_dimension)

        if self.args.concatenate:
            self.contrastive = ContrastiveNetwork(dim=self.args.embedding_dimension * self.dictionary.longest_doc_len)
        else:
            self.contrastive = ContrastiveNetwork(dim=self.args.embedding_dimension)

        # Training
        self.lr = self.args.learning_rate
        self.epochs = self.args.epochs

        self.params = self.parameters()
        self.optim = optim.Adam(self.params, lr=self.lr)
        self.contrastive_loss = ContrastiveLoss(batch_size=self.args.batch_size)

        self.losses = []

    def train_model(self):

        losses = []

        print("Initializing training. ")

        self.train()
        bar = tqdm(range(self.epochs))

        for i in range(self.epochs):
            epoch_loss = 0
            for j, (trades, confs, _, _) in enumerate(self.dataloader):

                self.optim.zero_grad()

                latent_trades, latent_confs = self.forward(trades, confs, batch=self.args.batch_size)

                loss, positives = self.contrastive_loss(latent_trades, latent_confs)

                loss.backward()

                self.optim.step()
                epoch_loss += loss.item()

                ### Remove comments to plot during training
                #if j % 10 == 0:
                #    plt.figure()
                #    plt.scatter(F.normalize(latent_trades).detach().numpy()[:, 0], F.normalize(latent_trades).detach().numpy()[:, 1])
                #    plt.scatter(F.normalize(latent_confs).detach().numpy()[:, 0], F.normalize(latent_confs).detach().numpy()[:, 1])
                #    plt.show()
                #    bar.set_description(f"Epoch: {i + 1} Loss: {epoch_loss / len(self.dataloader)} Positives: {positives}")


            losses.append(epoch_loss)
            bar.update()
            bar.set_description(f"Epoch: {i + 1} Loss: {epoch_loss/len(self.dataloader)} Positives: {positives}")

        self.losses.append(losses)

    def indecies_to_text(self, indicies) -> list:
        text = [self.dictionary.i2w[i] for i in indicies]
        return text

    def evaluate(self, data, p=0.8, batch=10):
        self.test_dataloader = DataLoader(data, batch_size=batch, shuffle=self.args.shuffle)
        self.eval()
        correct = 0
        count = 0
        accuracy = 0
        for trades, confs, text_trades, text_confs in self.test_dataloader:
            print(text_trades.shape, text_confs.shape)

            latent_trades, latent_confs, trades_weight, confs_weight = self.forward_attention_weights(trades, confs, batch=batch)
            latent_trades, latent_confs = F.normalize(latent_trades, p=2, dim=1), F.normalize(latent_confs, p=2, dim=1)

            print(trades_weight.shape, confs_weight.shape)
            print(self.indecies_to_text(list(text_trades[0, :].detach().numpy())))
            print(self.indecies_to_text(list(text_confs[0, :].detach().numpy())))
            text_trade = self.indecies_to_text(list(text_trades[0, :].detach().numpy()))
            text_conf = self.indecies_to_text(list(text_confs[0, :].detach().numpy()))
            print(trades_weight[0, :, :])
            print(trades_weight[0, :, :].sum(0))
            print(confs_weight[0, :, :].sum(0))
            print(trades_weight[0, :, :].sum(1))

            self.plot_words(text_trade, list(trades_weight[0, :, :].sum(0).detach().numpy()))
            self.plot_words(text_conf, list(confs_weight[0, :, :].sum(0).detach().numpy()))

            plt.figure()
            plt.scatter(latent_trades.detach().numpy()[:, 0],
                        latent_trades.detach().numpy()[:, 1])
            plt.scatter(latent_confs.detach().numpy()[:, 0],
                        latent_confs.detach().numpy()[:, 1])
            plt.show()

            similarity = self.calc_similarity_batch(latent_trades, latent_confs)

            similarities = similarity[:batch, batch:]

            #probabilities = F.softmax(similarities, dim=1)
            #print(probabilities)
            #chosen_probabilities = probabilities.max().detach().numpy()
            #print(chosen_probabilities)

            #print(similarities)
            armax = torch.argmax(similarities, dim=1).detach().numpy()
            print(f"Argmax: {armax}")

            for i, pred in enumerate(list(armax)):
                correct += 1 if i == pred else 0
                accuracy += 0
            count += batch
        print(f"Accuracy = {correct / count}")
        print(f"Average prediction certainty = {accuracy / correct}")

    def load_model(self, model_file):
        self.load_state_dict(torch.load(self.PATH + model_file))

    def save_model(self, model_file):
        torch.save(self.state_dict(), self.PATH + model_file)

    @staticmethod
    def calc_similarity_batch(a, b):
        print(a.shape)
        print(b.shape)
        representations = torch.cat([a, b], dim=0)
        print(representations.shape)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def plot_words(self, tokens, scores, bg_color='turquoise', cmap='inferno'):

        d = {w: f for w, f in zip(tokens, scores)}
        wordcloud = WordCloud(background_color=bg_color, colormap=cmap, prefer_horizontal=1)
        wordcloud.generate_from_frequencies(frequencies=d)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()

    def match_using_cosine_sim(self, data, batch):
        self.test_dataloader = DataLoader(data, batch_size=batch, shuffle=self.args.shuffle)
        self.eval()
        correct = 0
        count = 0
        accuracy = 0
        for trades, confs, text_trades, text_confs in self.test_dataloader:

            #t = trades.mean(2)
            #c = confs.mean(2)
            t = trades.view(batch, -1)
            c = confs.view(batch, -1)
            print("text")
            print(text_trades.shape)
            print(text_confs.shape)
            print(t.shape)
            print(c.shape)
            sims = self.calc_similarity_batch(text_trades, text_confs)
            print(sims.shape)

            similarities = sims#[:batch, batch:]
            print(similarities)
            armax = torch.argmax(similarities, dim=1).detach().numpy()
            print(f"Argmax: {armax}")

            for i, pred in enumerate(list(armax)):
                correct += 1 if i == pred else 0
                accuracy += 0
            count += batch
        print(f"Accuracy = {correct / count}")
        print(f"Average prediction certainty = {accuracy / correct}")



