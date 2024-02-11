import os
import sys

from scipy.special import comb

sys.path.append("python/")

import metrics
import numpy as np
from data import Dataloader
from models import HeuristicModel, TwoClustersMIP

np.random.seed(42)


def predict_utility_from_avg_coefficients(X, avg_coefficients, all_mins, all_maxs):
    """
    Predict the utility of each sample in X using the average utility coefficients
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Samples to predict the utility
    avg_coefficients : np.ndarray of shape (n_features, n_clusters)
        Average utility coefficients
    all_mins : np.ndarray of shape (n_features)
        Minimum value for each feature
    all_maxs : np.ndarray of shape (n_features)
        Maximum value for each feature

    Returns
    -------
    np.ndarray of shape (n_samples, n_clusters)
        Predicted utilities
    """

    L = 5
    N_CRITERIA = 10

    def xl(i, l):
        return all_mins[i] + l * (all_maxs[i] - all_mins[i]) / L

    def step(i):
        return (all_maxs[i] - all_mins[i]) / L

    def u_k_i(k, i, j, X):
        x = X[j][i]

        if x == all_maxs[i]:
            return avg_coefficients[k][i][L]

        else:
            li = int((x - all_mins[i]) / step(i))

            # On retourne la valeur de la fonction en la valeur d'intérêt
            res = (avg_coefficients[k][i][li + 1] - avg_coefficients[k][i][li]) / (
                step(i)
            ) * (x - xl(i, li)) + avg_coefficients[k][i][li]

            return res

    # Fonction qui renvoie la valeur de la somme des utilités pour un produit
    def u_k(k, j, X):
        return sum(u_k_i(k, i, j, X) for i in range(N_CRITERIA))

    return np.stack(
        [np.array([u_k(0, j, X), u_k(1, j, X), u_k(2, j, X)]) for j in range(len(X))],
        axis=1,
    )


def evaluate_cluster_intersection(Z_pred, Z_true):
    """main function to call the ClusterIntersection metric

    Parameters
    ----------
    z_pred (np.ndarray of shape (n_elements)):
        index (in {0, 1, ..., n}) of predicted cluster for each element
    z_true (np.ndarray of shape (n_elements)):
        index (in {0, 1, ..., n}) of ground truth cluster for each element

    Returns
    -------
    float Percentage of pairs attributed regrouped within same cluster in prediction compared to ground truth
    """
    # print(z_pred.shape, z_true.shape)
    assert Z_true.shape == Z_pred.shape

    truepos_plus_falsepos = comb(np.bincount(Z_true), 2).sum()
    truepos_plus_falseneg = comb(np.bincount(Z_pred), 2).sum()
    concatenation = np.c_[(Z_true, Z_pred)]
    true_positive = sum(
        comb(np.bincount(concatenation[concatenation[:, 0] == i, 1]), 2).sum()
        for i in set(Z_true)
    )
    false_positive = truepos_plus_falsepos - true_positive
    false_negative = truepos_plus_falseneg - true_positive
    true_negative = (
        comb(len(concatenation), 2) - true_positive - false_positive - false_negative
    )
    return (true_positive + true_negative) / (
        true_positive + false_positive + false_negative + true_negative
    )


if __name__ == "__main__":
    ### First part: test of the MIP model
    # data_loader = Dataloader("data/dataset_4_test")  # Path to test dataset
    # X, Y = data_loader.load()

    # model = TwoClustersMIP(
    #     n_clusters=2, n_pieces=5
    # )  # You can add your model's arguments here, the best would be set up the right ones as default.
    # model.fit(X, Y)

    # # %Pairs Explained
    # pairs_explained = metrics.PairsExplained()
    # print("Percentage of explained preferences:", pairs_explained.from_model(model, X, Y))

    # # %Cluster Intersection
    # cluster_intersection = metrics.ClusterIntersection()

    # Z = data_loader.get_ground_truth_labels()
    # print("% of pairs well grouped together by the model:")
    # print("Cluster intersection for all samples:", cluster_intersection.from_model(model, X, Y, Z))

    ### 2nd part: test of the heuristic model
    data_loader = Dataloader("data/dataset_10")  # Path to test dataset # _test
    X, Y = data_loader.load()

    # Compute mins and maxs on the whole dataset
    A = np.concatenate((X, Y), axis=0)
    all_mins = A.min(axis=0)
    all_maxs = A.max(axis=0)
    # print(f"all_mins: {all_mins}\n all_maxs: {all_maxs}")

    indexes = np.linspace(0, len(X) - 1, num=len(X), dtype=int)
    np.random.shuffle(indexes)
    print(f"indexes size: {len(indexes)}")
    all_train_indexes = indexes[: int(len(indexes) * 0.9)]
    print(f"all_train_indexes size: {len(all_train_indexes)}")
    train_indexes = np.random.choice(all_train_indexes, 3000, replace=True)
    print(f"train_indexes size: {len(train_indexes)}")
    test_indexes = indexes[int(len(indexes) * 0.9) :]
    print(f"test_indexes size: {len(test_indexes)}")

    # print(train_indexes, test_indexes)
    X_train = X[train_indexes]
    Y_train = Y[train_indexes]
    model = HeuristicModel(all_mins, all_maxs, n_clusters=3)
    model.fit(X_train, Y_train)

    X_test = X[test_indexes]
    Y_test = Y[test_indexes]
    Z_test = data_loader.get_ground_truth_labels()[test_indexes]

    # Validation on test set
    # %Pairs Explained
    pairs_explained = metrics.PairsExplained()
    print(
        "Percentage of explained preferences:",
        pairs_explained.from_model(model, X_test, Y_test),
    )

    # %Cluster Intersection
    cluster_intersection = metrics.ClusterIntersection()
    print("% of pairs well grouped together by the model:")
    print(
        "Cluster intersection for all samples:",
        cluster_intersection.from_model(model, X_test, Y_test, Z_test),
    )

    ### 3rd part: test of the heuristic model on multiple sapled datasets and average the results

    # data_loader = Dataloader("data/dataset_10")  # Path to test dataset # _test
    # X, Y = data_loader.load()

    # # Compute mins and maxs on the whole dataset
    # A = np.concatenate((X, Y), axis=0)
    # all_mins = A.min(axis=0)
    # all_maxs = A.max(axis=0)

    # nb_iterations = 10
    # utility_coefficients = []

    # all_indexes = np.linspace(0, len(X) - 1, num=len(X), dtype=int)
    # np.random.shuffle(all_indexes)
    # all_train_indexes = all_indexes[: int(len(all_indexes) * 0.95)]
    # all_test_indexes = all_indexes[int(len(all_indexes) * 0.95) :]

    # for i in range(nb_iterations):
    #     print(f"ITERATION {i}")
    #     train_indexes = np.random.choice(all_train_indexes, 2000, replace=True)
    #     X_train = X[train_indexes]
    #     Y_train = Y[train_indexes]
    #     model = HeuristicModel(all_mins, all_maxs, n_clusters=3)
    #     model.fit(X_train, Y_train)
    #     coefficients = model.get_u_k()
    #     utility_coefficients.append(coefficients)

    # avg_utility_coefficients = np.mean(utility_coefficients, axis=0)

    # X_test = X[all_test_indexes]
    # Y_test = Y[all_test_indexes]
    # Z_true = data_loader.get_ground_truth_labels()[all_test_indexes]
    # U_X = predict_utility_from_avg_coefficients(
    #     X_test, avg_utility_coefficients, all_mins, all_maxs
    # )
    # U_Y = predict_utility_from_avg_coefficients(
    #     Y_test, avg_utility_coefficients, all_mins, all_maxs
    # )

    # # Pairs Explained
    # print(U_X.shape, U_Y.shape)
    # pairs_explained = 100 * np.sum(np.sum(U_X - U_Y > 0, axis=1) > 0) / len(U_X)
    # print(
    #     "Percentage of explained preferences:",
    #     pairs_explained,
    # )

    # # Cluster Intersection
    # Z_pred = np.argmax(U_X - U_Y, axis=0)
    # print(
    #     "Cluster intersection for all samples:",
    #     evaluate_cluster_intersection(Z_pred, Z_true),
    # )
