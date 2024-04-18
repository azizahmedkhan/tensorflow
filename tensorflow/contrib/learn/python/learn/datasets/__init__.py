"""Module includes reference datasets and utilities to load datasets."""

import csv
from os import path

import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.contrib.learn.python.learn.datasets import text_datasets

# Export load_iris and load_boston.
load_iris = base.load_iris
load_boston = base.load_boston

# List of all available datasets.
# Note, currently they may return different types.
DATASETS = {
    # Returns base.Dataset.
    'iris': base.load_iris,
    'boston': base.load_boston,
    # Returns base.Datasets (train/validation/test sets).
    'mnist': mnist.load_mnist,
    'dbpedia': text_datasets.load_dbpedia,
}


def load_dataset(name):
  """Loads dataset by name.

  Args:
    name: Name of the dataset to load.

  Returns:
    Features and targets for given dataset. Can be numpy or iterator.

  Raises:
    ValueError: if `name` is not found.
  """
  if name not in DATASETS:
    raise ValueError('Name of dataset is not found: %s' % name)
  return DATASETS[name]()
