import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-embedding', '-tE', type=bool, default=False,
                        help='Option to make word embedding layer trainable. ')
    parser.add_argument('--embedding-dimension', '-eD', default=50, type=int,
                        help='Dimension of the word embeddings in the initial embedding layer. ')
    parser.add_argument('--epochs', '-e', type=int, default=500, help='Number of training epochs. ')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001, help='Learning rate. ')
    parser.add_argument('--number-of-heads', '-nH', type=int, default=2, help='Number of heads used for attention. ')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='Device to be used, "cpu" or "cuda". ')
    parser.add_argument('--batch-size', '-bS', type=int, default=20, help='Batch size for training. ')
    parser.add_argument('--shuffle', '-sh', type=bool, default=True, help='Shuffle training samples during training. ')
    parser.add_argument('--projection-head', '-pH', type=bool, default=False,
                        help='Whether or not to use projection head. ')
    parser.add_argument('--concatenate', '-c', type=bool, default=False,
                        help='Whether output of attention module should be concatenated or summed and taken average. ')

    args = parser.parse_args(args=[])

    return args