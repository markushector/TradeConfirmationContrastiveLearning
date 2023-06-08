import os

import torch.nn as nn
import torch

import numpy as np
import enum

from gen.enums import Currencies, Directions


class EmbeddingLayer(nn.Module):
    """
    The embedding-layer used to embed text-documents to matrices.
    Uses pre-trained glove-embeddings.
    """

    def __init__(self, args, target_vocab: list):
        super(EmbeddingLayer, self).__init__()

        self.target_vocab: list = target_vocab
        self.args = args

        self.glove = self._get_glove_embeddings()

        self.weights_matrix = self._create_weights_matrix()
        self.embedding_layer = self._create_emb_layer()

    def forward(self, x):
        """ input: (*)
            output: (*, emb_dim)
        """
        x = self.embedding_layer(x)
        return x

    def _create_weights_matrix(self):
        emb_dim = self.args.embedding_dimension
        matrix_len = len(self.target_vocab)
        weights_matrix = np.zeros((matrix_len, emb_dim))
        words_found = 0

        for i, word in enumerate(self.target_vocab):
            #if False:
            if word in [ccy.value.lower() for ccy in Currencies] + [dire.value.lower() for dire in Directions]:
                weights_matrix[i, :] = np.random.normal(scale=0.6, size=(emb_dim,))
                print(f"Random init of currency or direction: {word}")
                continue
            try:
                weights_matrix[i] = self.glove[word]
                words_found += 1
            except KeyError:
                weights_matrix[i, :] = np.random.normal(scale=0.6, size=(emb_dim,))
                print(word)

        print(f"In dictionary of size {len(self.target_vocab)}, {words_found} words were found pre-trained in glove, "
              f"and {len(self.target_vocab) - words_found} was initialized randomly. ")

        return weights_matrix

    def _create_emb_layer(self):
        emb_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(self.weights_matrix).float())
        if not self.args.train_embedding:
            emb_layer.weight.requires_grad = False

        return emb_layer

    def _get_glove_embeddings(self) -> dict:
        glove = {}
        file = self.get_glove_file()
        with open(file, 'rt') as fi:
            full_content = fi.read().strip().split('\n')
        for i in range(len(full_content)):
            i_word = full_content[i].split(' ')[0]
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
            glove[i_word] = i_embeddings
        return glove

    def get_glove_file(self):
        emb_dim = self.args.embedding_dimension
        if emb_dim == 50:
            print(os.getcwd())
            return 'contrastive_matching/' + GloveFiles.GLOVE_50D.value
        elif emb_dim == 100:
            return GloveFiles.GLOVE_100D.value
        elif emb_dim == 200:
            return GloveFiles.GLOVE_200D.value
        elif emb_dim == 300:
            return GloveFiles.GLOVE_300D.value
        else:
            print("Unknown dimension chosen for embedding layer. ")


class GloveFiles(enum.Enum):
    GLOVE_50D = 'glove/glove.6B.50d.txt'
    GLOVE_100D = 'glove/glove.6B.100d.txt'
    GLOVE_200D = 'glove/glove.6B.200d.txt'
    GLOVE_300D = 'glove/glove.6B.300d.txt'
