from contrastive_matching.model.visinet import VisiNet

import pickle


disc_data_files = ['disc_data_set.pkl',
                   'disc_train_data_set.pkl',
                   'disc_test_data_set.pkl',
                   'disc_train_dataloader.pkl',
                   'disc_test_dataloader.pkl',
                   'disc_dictionary.pkl']


def load_data(model, files):
    dir = 'disc_data_files/'
    with open(dir + files[0], 'rb') as inp:
        print(f"Loading {files[0]}")
        model.data = pickle.load(inp)
    with open(dir + files[1], 'rb') as inp:
        print(f"Loading {files[1]}")
        model.train_data = pickle.load(inp)
    with open(dir + files[2], 'rb') as inp:
        print(f"Loading {files[2]}")
        model.test_data = pickle.load(inp)
    with open(dir + files[3], 'rb') as inp:
        print(f"Loading {files[3]}")
        model.dataloader = pickle.load(inp)
    with open(dir + files[4], 'rb') as inp:
        print(f"Loading {files[4]}")
        model.test_dataloader = pickle.load(inp)
    with open(dir + files[5], 'rb') as inp:
        print(f"Loading {files[5]}")
        model.dictionary = pickle.load(inp)


model = VisiNet()
#load_data(model, disc_data_files)
model.load_data()
model.build_model()
model.train_model()

