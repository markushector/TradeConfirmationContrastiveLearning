import argparse
import os


def delete_files_in_directory(path):
    curr_dir = os.getcwd()
    os.chdir(path)
    contents_of_dir = os.listdir()
    if len(contents_of_dir) > 0:
        for file in contents_of_dir:
            os.remove(file)
        print("All files removed from data directory. ")
    else:
        print("Data directory empty, nothing to delete. ")
    os.chdir(curr_dir)


def get_argument_parser():
    parser = argparse.ArgumentParser(description='Generate financial data')
    parser.add_argument('--number', '-N', type=int, default=20, help='Number of trades to be generated')
    parser.add_argument('--cptys', '-c', type=int, default=5, help='Number of different counterparties')
    parser.add_argument('--html', type=bool, default=False, help='Option to generate html confirmations as well')
    parser.add_argument('--noise', '-n', type=bool, default=False, help='Option to generate noise')
    parser.add_argument('--error-probability', '-e', type=float, default=0.0,
                        help='Probability of errors in the generation of confirmations. ')
    parser.add_argument('--data-path', '-d', type=str, default='data',
                        help='Path to directory where to save generated data. ')
    return parser