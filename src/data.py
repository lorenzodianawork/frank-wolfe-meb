"""
Dataset loading and MEB problem construction.

Datasets (MNIST, Fashion-MNIST, CIFAR-10) are flattened to feature vectors,
normalized to [0, 1], and split 80/20 with a fixed seed for reproducibility.
"""

import os
import pickle
import tarfile
import urllib.request

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist, fashion_mnist

SEED = 42


def load_cifar10_fast(cache_dir="./data/cifar10"):
    """Downloads CIFAR-10 from an OSF mirror (faster/more reliable than the
    original Toronto server) and caches it locally.
    """
    os.makedirs(cache_dir, exist_ok=True)
    data_dir = os.path.join(cache_dir, "cifar-10-batches-py")
    tar_path = os.path.join(cache_dir, "cifar-10-osf.tar.gz")

    if not os.path.isdir(data_dir):
        if not os.path.exists(tar_path):
            url = "https://osf.io/jbpme/download"
            urllib.request.urlretrieve(url, tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(cache_dir)

    def _load_batch(fname):
        with open(os.path.join(data_dir, fname), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        X = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y = np.array(d[b"labels"], dtype="uint8")
        return X, y

    xs, ys = zip(*[_load_batch(f"data_batch_{i}") for i in range(1, 6)])
    X_train, y_train = np.concatenate(xs), np.concatenate(ys)
    X_test, y_test = _load_batch("test_batch")

    return (X_train, y_train.reshape(-1, 1)), (X_test, y_test.reshape(-1, 1))


def load_dataset(name, test_size=0.2, seed=SEED):
    """Loads one of {'mnist', 'fashion_mnist', 'cifar10'}, merges the
    original train/test pools, flattens and normalizes to [0, 1], then
    re-splits 80/20 with a fixed seed so all solvers see identical data.
    """
    if name == "cifar10":
        (X_tr, y_tr), (X_te, y_te) = load_cifar10_fast()
    else:
        loaders = {"mnist": mnist, "fashion_mnist": fashion_mnist}
        (X_tr, y_tr), (X_te, y_te) = loaders[name].load_data()

    X_all = np.concatenate([X_tr, X_te])
    X_all = X_all.reshape(X_all.shape[0], -1).astype("float64") / 255.0
    y_all = np.concatenate([y_tr.flatten(), y_te.flatten()]).astype(str)

    return train_test_split(X_all, y_all, test_size=test_size, random_state=seed)


def build_meb_problem(X_class):
    """Builds the dual quadratic-program data (P, Q, c) for one class:
    P holds the points as columns, Q = P^T P is the Gram matrix, and
    c_i = ||a_i||^2.
    """
    P = X_class.T
    c = np.sum(P ** 2, axis=0)
    Q = P.T @ P
    return P, Q, c
