import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import spacy

from contrastive_matching.model.embedding_module import EmbeddingLayer
from contrastive_matching.model.positional_encoder import PositionalEncoding
from contrastive_matching.utils.utils import get_args

#DATA_DIR = './cont_data'
DATA_DIR = './data'


class Dictionary(object):
    """
    Dictionary-class that goes through data to create dictionary, and index-to-word/word-to-index mappings.
    """

    TRADE_STRING = '_trade.txt'
    CONFIRMATION_STRING = '_trade_confirmation.txt'

    TRADE_INDEX = 0
    CONFIRMATION_INDEX = 1

    PAD = '<PAD>'

    def __init__(self, data_dir=DATA_DIR):
        self.project_root = os.getcwd()
        self.tokenizer = spacy.load("en_core_web_sm")
        # self.stopwords = set(stopwords.words('english'))
        self.stopwords = ['.', ',', '!', '?', '\n', '\n\n', '\n\n\n', ';', ':', '-', '_', '$']

        self.data_path = data_dir
        self.data_frame = self._read_contrastive_data_to_pandas()
        self.dictionary, self.i2w, self.w2i, self.longest_doc_len = self._create_dictionary()
        self.number_of_pairs = len(self.data_frame.index)

    def __getitem__(self, index):
        return self.dictionary[index]

    def __len__(self):
        return self.number_of_pairs

    def _read_contrastive_data_to_pandas(self) -> pd.DataFrame:

        os.chdir(self.data_path)
        number_of_data_pairs = int(len(os.listdir()) / 2)
        data_frame = pd.DataFrame(columns=['Trade', 'Confirmation'])

        for i in range(number_of_data_pairs):

            with open(str(i) + self.TRADE_STRING, 'r') as file:
                trade = file.read()

            with open(str(i) + self.CONFIRMATION_STRING, 'r') as file:
                confirmation = file.read()

            data_frame.loc[i] = [trade, confirmation]
        os.chdir(self.project_root)

        return data_frame

    def _create_dictionary(self):

        dictionary = []
        i2w = {}
        w2i = {}
        running_index = 0
        longest_doc_len = 0

        for i in range(len(self.data_frame.index)):
            confirmation = self.data_frame.iloc[i, self.CONFIRMATION_INDEX]
            trade = self.data_frame.iloc[i, self.TRADE_INDEX]
            conf_tokens = self.tokenizer(confirmation)
            trade_tokens = self.tokenizer(trade)
            if len(conf_tokens) > longest_doc_len:
                longest_doc_len = len(conf_tokens)

            if len(trade_tokens) > longest_doc_len:
                longest_doc_len = len(trade_tokens)

            for token in conf_tokens:
                # Split if number is decimal
                tokens = check_if_int_and_split(token.text.split('.'))
                for word in [token.lower() for token in tokens]:
                    if word not in dictionary and word not in self.stopwords:
                        dictionary.append(word)
                        i2w[running_index] = word
                        w2i[word] = running_index
                        running_index += 1

            for token in trade_tokens:
                # Split if number is decimal
                tokens = check_if_int_and_split(token.text.split('.'))
                for word in [token.lower() for token in tokens]:
                    if word not in dictionary and word not in self.stopwords:
                        dictionary.append(word)
                        i2w[running_index] = word
                        w2i[word] = running_index
                        running_index += 1

        dictionary.append(self.PAD)
        i2w[running_index] = self.PAD
        w2i[self.PAD] = running_index

        return dictionary, i2w, w2i, longest_doc_len


class ContrastiveDataSet(Dataset):
    """
    Class containing dataset, inherits Dataset from Pytorch. Fetches data-files and converts them to torch-tensors.
    """

    TRADE_STRING = '_trade.txt'
    CONFIRMATION_STRING = '_trade_confirmation.txt'

    TRADE_INDEX = 0
    CONFIRMATION_INDEX = 1

    def __init__(self, dictionary, device):
        self.project_root = os.getcwd()

        self.tokenizer = spacy.load("en_core_web_sm")
        self.dictionary = dictionary

        self.data_path = dictionary.data_path

        self.args = get_args()
        self.device = device

        self.embedding = EmbeddingLayer(args=self.args,
                                        target_vocab=self.dictionary.dictionary).to(self.device)
        self.positional_encoder = PositionalEncoding(d_model=self.args.embedding_dimension).to(self.device)

    def __getitem__(self, index):

        trade, confirmation = self._read_trade_confirmation_pair(index)
        cleaned_trade = self._clean_text([token.text.lower() for token in self.tokenizer(trade)])
        cleaned_conf = self._clean_text([token.text.lower() for token in self.tokenizer(confirmation)])
        trade_in_index = self._convert_text_to_index_tensor(cleaned_trade)
        conf_in_index = self._convert_text_to_index_tensor(cleaned_conf)
        embedded_trade = self.positional_encoder(self.embedding(trade_in_index)[:, None, :]).squeeze()
        embedded_conf = self.positional_encoder(self.embedding(conf_in_index)[:, None, :]).squeeze()
        random_trade = torch.randn_like(embedded_trade)
        random_conf = random_trade + 0.01*torch.randn_like(random_trade)
        #random_conf = torch.randn_like(embedded_conf)
        p = 0.0
        embedded_trade += p * torch.randn_like(embedded_trade)
        embedded_conf += p * torch.randn_like(embedded_conf)

        return (embedded_trade, embedded_conf, torch.tensor(trade_in_index), torch.tensor(conf_in_index))
        #return random_trade, random_conf

    def __len__(self):

        return self.dictionary.number_of_pairs

    def _read_trade_confirmation_pair(self, index: int) -> [str, str]:

        os.chdir(self.data_path)
        with open(str(index) + self.TRADE_STRING, 'r') as file:
            trade = file.read()

        with open(str(index) + self.CONFIRMATION_STRING, 'r') as file:
            confirmation = file.read()

        os.chdir(self.project_root)
        return [trade, confirmation]

    def _clean_text(self, tokens: list) -> list:
        cleaned_tokens = []
        for token in tokens:
            words = check_if_int_and_split(token.split('.'))
            for word in words:
                if word not in self.dictionary.stopwords:
                    cleaned_tokens.append(word)
        padded_tokens = cleaned_tokens[:]
        for _ in range(self.dictionary.longest_doc_len - len(cleaned_tokens)):
            padded_tokens.append(self.dictionary.PAD)
        return padded_tokens

    def _convert_text_to_index_tensor(self, tokens: list):
        index_representation = []
        for token in tokens:
            index = self.dictionary.w2i[token]
            index_representation.append(index)
        tensor = torch.tensor(index_representation).to(self.device)
        return tensor


def check_if_int_and_split(tokens: list):
    split_tokens = []
    for token in tokens:
        try:
            int(token)
            if (len(token) > 2) and len(token) % 2 == 0:
                split_integers = [token[i * 2:i * 2 + 2] for i in range(int((len(token) / 2)))]
                split_tokens += split_integers
            elif (len(token) > 2) and len(token) % 2 == 1:
                split_integers = [token[i * 2:i * 2 + 2] for i in range(int(((len(token) - 1) / 2)))]
                split_integers.append(token[-1])
                split_tokens += split_integers
            else:
                split_tokens.append(token)

        except ValueError as _:
            split_tokens.append(token)
            continue

    return split_tokens



