#!/bin/python3

import argparse

from tqdm import tqdm
import numpy as np
import tensorflow as tf


from data_preprocessing import get_and_fix_data, X_path_train, Y_path_train, X_path_test, Y_path_test
from graph_constructor import LogisticGraph, TwoLayerNN, KNNGraph


def train_and_validate(algorithm, K=5, H=100, num_iters=5000, beta=0.05):
    X_train, Y_train, X_dev, Y_dev = get_and_fix_data(X_path_train, Y_path_train, .15)

    X_test, Y_test, _, _ = get_and_fix_data(X_path_test, Y_path_test, 0)
    N = X_train.shape[0]
    C = Y_train.shape[0]

    if algorithm == 'logreg' or algorithm == 'nn':
        model = LogisticGraph(N, C) if algorithm == 'logreg' else TwoLayerNN(N, C, H, 'relu', beta)
        losses = np.array([])
        with tf.Session() as sess:
            cost = np.inf
            sess.run(model.model)
            for _ in tqdm(range(num_iters)):
                _, cost = sess.run([model.train, model.loss], feed_dict={model.X: X_train, model.y: Y_train})
                losses = np.append(losses, cost)

            print('Final cost: %.5f' % cost)

            print('Train accuracy: %.3f' %
                  (100 * model.accuracy.eval(feed_dict={model.X: X_train, model.y: Y_train})) + '%')

            print('Dev/validation accuracy: %.3f' %
                     (100 * model.accuracy.eval(feed_dict={model.X: X_dev, model.y: Y_dev})) + '%')

            print('Test accuracy: %.3f' %
                  (100 * model.accuracy.eval(feed_dict={model.X: X_test, model.y: Y_test})) + '%')

    if algorithm == 'knn':
        model = KNNGraph(N, C, K)
        # To do: compute accuracy over train, dev and test set, probably iterate over X_<...>.T
        with tf.Session():
            print('Evaluating the training set accuracy')
            train_results = []
            for i in tqdm(range(X_train.shape[1])):
                train_results.append(int(Y_train[:, i:i + 1].argmax() == model.pred.eval(
                    feed_dict={model.X_tr: X_train, model.Y_tr: Y_train, model.X_te: X_train[:, i:i + 1]})))
            train_results = np.array(train_results)
            train_accuracy = np.mean(train_results)

            print('Evaluating the dev set accuracy')
            dev_results = []
            for i in tqdm(range(X_dev.shape[1])):
                dev_results.append(int(Y_dev[:, i:i + 1].argmax() == model.pred.eval(
                    feed_dict={model.X_tr: X_train, model.Y_tr: Y_train, model.X_te: X_dev[:, i:i + 1]})))
            dev_results = np.array(dev_results)
            dev_accuracy = np.mean(dev_results)

            print('Evaluating the test set accuracy')
            test_results = []
            for i in tqdm(range(X_test.shape[1])):
                test_results.append(int(Y_test[:, i:i + 1].argmax() == model.pred.eval(
                    feed_dict={model.X_tr: X_train, model.Y_tr: Y_train, model.X_te: X_test[:, i:i + 1]})))
            test_results = np.array(test_results)
            test_accuracy = np.mean(test_results)

            print('Train accuracy: %.3f' % train_accuracy)
            print('Dev/validation accuracy: %.3f' % dev_accuracy)
            print('Test accuracy: %.3f' % test_accuracy)


def main():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm',
                        choices=['knn', 'nn', 'logreg'],
                        help='Name of the algorithm to be used. Currently supported: logreg, nn, knn')

    parser.add_argument('-K', help='Value of K for the KNN algorithm', type=int, default=5)
    parser.add_argument('-H', '--n_hidden',
                        help='Number of neurons in the hidden layer of the neural network', type=int, default=100)
    parser.add_argument('-i', '--num_iters',
                        help="Number of training iterations for logreg and NN models", type=int, default=5000)
    parser.add_argument('-b', '--beta', help='Coefficient of L2 regularization', type=float, default=0.05)

    args = parser.parse_args()

    train_and_validate(args.algorithm, K=args.K, H=args.n_hidden, num_iters=args.num_iters, beta=args.beta)


if __name__ == '__main__':
    main()
