#!/bin/python3

import numpy as np
import tensorflow as tf
import argparse

from data_preprocessing import get_and_fix_data, X_path_train, Y_path_train, X_path_test, Y_path_test
from graph_constructor import LogisticGraph, TwoLayerNN, KNNGraph


def main():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', help='Name of the algorithm to be used. Currently supported: logreg, NN, KNN')

    parser.add_argument('-K', help='Value of K for the KNN algorithm', type=int)
    parser.add_argument('-H', '--n_hidden',
                        help='Number of neurons in the hidden layer of the neural network', type=int)

    

    args = parser.parse_args()




if __name__ == '__main__':
    main()